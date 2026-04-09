"""SER — ReAct Worker Agent.

Implements the ReAct (Reason-Act-Observe) loop for code generation and execution.
Actions: bash(command), python(code), submit(path)
"""

import base64
import json
import os
import re
import signal
import time
import logging
import subprocess
import tempfile
from pathlib import Path
from statistics import mean
from threading import Thread
from typing import Optional

from llm import LLMClient
from config import (
    MAX_REACT_STEPS, CODE_EXEC_TIMEOUT, OBSERVATION_MAX_CHARS,
    PYTHON_BIN, LLM_TEMPERATURE,
)

# GPU preflight constants (not in new config, use defaults)
PREFLIGHT_SECS = 120
PREFLIGHT_MIN_GPU_UTIL = 15
PREFLIGHT_TRAIN_WAIT = 300
PREFLIGHT_TRAIN_MEM_THR = 2000
AGENT_USER = "root"  # no user isolation in this setup

logger = logging.getLogger(__name__)


def _pipe_reader(pipe, line_list):
    """Read lines from a pipe into a list (runs in a daemon thread)."""
    try:
        for line in pipe:
            line_list.append(line)
    except (ValueError, OSError):
        pass  # pipe closed


def truncate_output(text: str, max_chars: int = OBSERVATION_MAX_CHARS) -> str:
    """Truncate output, keeping head and tail."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... [TRUNCATED {len(text) - max_chars} chars] ...\n\n"
        + text[-half:]
    )


def _resolve_agent_user():
    """Resolve AGENT_USER to (uid, gid) or return None if user doesn't exist."""
    try:
        import pwd
        pw = pwd.getpwnam(AGENT_USER)
        return pw.pw_uid, pw.pw_gid
    except (KeyError, ImportError):
        return None


class ReActAgent:
    """Multi-step ReAct agent that executes code in a working directory."""

    def __init__(
        self,
        llm: LLMClient,
        work_dir: Path,
        python_bin: Path = PYTHON_BIN,
        max_steps: int = MAX_REACT_STEPS,
        exec_timeout: int = CODE_EXEC_TIMEOUT,
        gpu_id: int = None,
        eval_score_fn=None,
        format_check_fn=None,
        tmp_dir: Path = None,
    ):
        self.llm = llm
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.python_bin = python_bin
        self.max_steps = max_steps
        self.exec_timeout = exec_timeout
        self.gpu_id = gpu_id
        # Optional callback: eval_score_fn(submission_path) -> (score, is_lower_better)
        # If provided, submit() returns the eval score instead of ending the loop.
        self.eval_score_fn = eval_score_fn
        # Optional callback: format_check_fn(submission_path) -> str | None
        # Returns a human-readable error string if submission format is wrong, else None.
        self.format_check_fn = format_check_fn
        # Per-iteration TMPDIR — isolates temp files so orchestrator can clean up after each iter
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None
        # Once a step passes preflight with high GPU util, don't kill subsequent steps.
        # This prevents inference-only steps from being falsely flagged as "low-util training".
        self._preflight_passed = False

        # Agent user isolation: chown work_dir so agent can write to it
        self._agent_user = _resolve_agent_user()
        if self._agent_user:
            uid, gid = self._agent_user
            try:
                os.chown(str(self.work_dir), uid, gid)
            except OSError as e:
                logger.warning(f"Failed to chown {self.work_dir} to {AGENT_USER}: {e}")

    def run(self, system_prompt: str, task_prompt: str) -> dict:
        """
        Run the ReAct loop.

        Returns:
            dict with keys:
                - submission_path: Path to submission CSV (or None)
                - code: str of the main solution code
                - trajectory: list of (reason, action, observation) tuples
                - steps: int number of steps taken
                - success: bool
        """
        messages = [{"role": "user", "content": task_prompt}]
        # Inject sample images from EDA if available (multimodal)
        messages = self._inject_images_if_available(messages)
        trajectory = []
        submission_path = None
        best_submission_path = None
        best_eval_score = None
        best_is_lower = None
        solution_code = ""
        start_time = time.time()

        for step in range(1, self.max_steps + 1):
            elapsed = time.time() - start_time
            logger.info(f"Step {step}/{self.max_steps} (elapsed: {elapsed:.0f}s)")

            # Inject step budget warnings at thresholds
            if step == 4 and submission_path is None:
                messages.append({
                    "role": "user",
                    "content": "REMINDER: Step 4 reached without submission. You should submit within the next 2 steps. If your current script isn't working, simplify your approach immediately."
                })
            elif step == 6 and submission_path is None:
                messages.append({
                    "role": "user",
                    "content": "WARNING: Step 6 without submission! You MUST submit NOW. If your script failed, write a simpler version and run it. Do NOT spend more steps debugging."
                })
            elif step >= 8 and submission_path is None:
                messages.append({
                    "role": "user",
                    "content": "CRITICAL: You are past step 8 without a submission! This iteration is almost wasted. Submit ANY submission.csv NOW — even copying sample_submission.csv is better than nothing. Do this in ONE step: `cp sample_submission.csv ./submission.csv` then submit."
                })

            # Get LLM response
            response = self.llm.chat(
                messages=messages,
                system=system_prompt,
                temperature=LLM_TEMPERATURE,
            )

            # Parse reason and action
            reason, action_type, action_content = self._parse_response(response)

            if reason:
                logger.info(f"  Reason: {reason[:300]}")

            if action_type is None:
                # No action found — ask LLM to produce one
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please provide an action. Use one of: bash(command), python(code), or submit(path)."
                })
                trajectory.append((reason, "none", "No action parsed"))
                continue

            logger.info(f"  Action: {action_type}({action_content[:200]})")

            # Execute action
            if action_type == "submit":
                # Try to find the submission file - check multiple locations
                sub_path = action_content.strip()
                resolved = self._resolve_path(sub_path)
                if resolved and Path(resolved).exists():
                    submission_path = resolved
                else:
                    found = self._find_submission()
                    submission_path = found if found else resolved

                if self.eval_score_fn and submission_path and Path(submission_path).exists():
                    # Score the submission and return feedback — do NOT end the loop
                    observation = self._handle_submit_with_score(
                        submission_path, best_eval_score, best_is_lower
                    )
                    score, is_lower = self._call_eval_score(submission_path)
                    if score is not None:
                        improved = (
                            best_eval_score is None
                            or (is_lower and score < best_eval_score)
                            or (not is_lower and score > best_eval_score)
                        )
                        if improved:
                            best_submission_path = submission_path
                            best_eval_score = score
                            best_is_lower = is_lower
                    trajectory.append((reason, f"submit({action_content})", observation))
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Observation:\n{observation}"})
                    solution_code = self._collect_solution_code()
                    # Continue the loop — agent can keep improving
                else:
                    # No eval scorer or file missing: original behaviour (end loop)
                    observation = self._handle_submit(submission_path)
                    trajectory.append((reason, f"submit({action_content})", observation))
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Observation:\n{observation}"})
                    solution_code = self._collect_solution_code()
                    break

            elif action_type == "bash":
                observation = self._exec_bash(action_content)
            elif action_type == "python":
                observation = self._exec_python(action_content)
                # Track written Python files
                if "# save" in action_content.lower() or "open(" in action_content:
                    solution_code = action_content
            else:
                observation = f"Unknown action type: {action_type}"

            observation = truncate_output(observation)
            trajectory.append((reason, f"{action_type}({action_content[:200]})", observation[:2000]))

            # Feed observation back
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        # Use best scoring submission if available, else last submitted
        if best_submission_path and Path(best_submission_path).exists():
            submission_path = best_submission_path

        # If no submission or resolved path doesn't exist, search broadly
        if submission_path is None or not Path(submission_path).exists():
            found = self._find_submission()
            if found:
                submission_path = found

        # Collect solution code if not yet done
        if not solution_code:
            solution_code = self._collect_solution_code()

        # Persist full trajectory to disk for debugging
        self._save_trajectory(trajectory, system_prompt, task_prompt)

        return {
            "submission_path": str(submission_path) if submission_path else None,
            "code": solution_code,
            "trajectory": trajectory,
            "steps": len(trajectory),
            "success": submission_path is not None and Path(submission_path).exists(),
            "elapsed": time.time() - start_time,
        }

    def _inject_images_if_available(self, messages: list[dict]) -> list[dict]:
        """If sample images exist from EDA, add them to the first user message."""
        sample_dir = self.work_dir.parent / "eda" / "sample_images"
        if not sample_dir.exists():
            # Also check work_dir's sibling paths
            for candidate in [
                self.work_dir.parent.parent / "eda" / "sample_images",
                self.work_dir / "sample_images",
            ]:
                if candidate.exists():
                    sample_dir = candidate
                    break
            else:
                return messages

        images = sorted(sample_dir.glob("*.png"))[:3]  # Max 3 samples
        if not images:
            return messages

        # Build multimodal content for the first user message
        first_msg = messages[0]
        if isinstance(first_msg["content"], str):
            content_blocks = [{"type": "text", "text": first_msg["content"]}]
        else:
            content_blocks = list(first_msg["content"])

        for img_path in images:
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                content_blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                })
                content_blocks.append({
                    "type": "text",
                    "text": f"[Sample image: {img_path.name}]",
                })
            except Exception as e:
                logger.warning(f"Failed to load sample image {img_path}: {e}")

        messages[0] = {"role": "user", "content": content_blocks}
        logger.info(f"Injected {len(images)} sample images into first message")
        return messages

    def _save_trajectory(self, trajectory, system_prompt: str, task_prompt: str):
        """Save full trajectory to disk for debugging."""
        traj_file = self.work_dir / "trajectory.jsonl"
        try:
            with open(traj_file, "w") as f:
                # Write meta as first line
                meta = {
                    "type": "meta",
                    "system_prompt_len": len(system_prompt),
                    "task_prompt_len": len(task_prompt),
                    "task_prompt_summary": task_prompt[:500],
                    "gpu_id": self.gpu_id,
                    "exec_timeout": self.exec_timeout,
                    "timestamp": time.time(),
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                # Write each step
                for i, (reason, action, obs) in enumerate(trajectory):
                    entry = {
                        "step": i + 1,
                        "reason": reason,
                        "action": action,
                        "observation": obs[:2000],
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")

    def _parse_response(self, response: str) -> tuple[str, Optional[str], str]:
        """Parse LLM response into (reason, action_type, action_content).

        Strategy: find the LAST code block in the response — the actual action
        is typically at the end, after reasoning that may contain inline code snippets.
        Also check for explicit Action: markers and <tag> markers first (highest priority).
        """
        reason = ""
        action_type = None
        action_content = ""

        # Priority 1: Explicit submit markers (check first)
        submit_patterns = [
            r'Action:\s*submit\(([^)]*)\)',
            r'<submit>(.*?)</submit>',
        ]
        for pattern in submit_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return response[:match.start()].strip(), "submit", match.group(1).strip()

        # Priority 2: Find ALL code blocks, use the LAST one as the action
        code_block_pattern = r'```(bash|python|sh)\n(.*?)```'
        matches = list(re.finditer(code_block_pattern, response, re.DOTALL))

        if matches:
            last_match = matches[-1]
            lang = last_match.group(1)
            action_content = last_match.group(2).strip()
            action_type = "bash" if lang in ("bash", "sh") else "python"
            reason = response[:last_match.start()].strip()
            return reason, action_type, action_content

        # Priority 3: Explicit Action: markers
        action_markers = [
            (r'Action:\s*bash\((.*?)\)', 'bash'),
            (r'Action:\s*python\((.*?)\)', 'python'),
        ]
        for pattern, atype in action_markers:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return response[:match.start()].strip(), atype, match.group(1).strip()

        # Priority 4: XML-style tags
        tag_patterns = [
            (r'<bash>(.*?)</bash>', 'bash'),
            (r'<python>(.*?)</python>', 'python'),
        ]
        for pattern, atype in tag_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return response[:match.start()].strip(), atype, match.group(1).strip()

        # No action found
        return response.strip(), None, ""

    def _apply_tmp_dir(self, env: dict):
        """Override TMPDIR/TEMP/TMP in env to use per-iteration isolated directory."""
        if self.tmp_dir:
            tmp = str(self.tmp_dir)
            env["TMPDIR"] = tmp
            env["TEMP"] = tmp
            env["TMP"] = tmp
            # HuggingFace / torch cache dirs also go here to avoid filling /tmp
            env.setdefault("HF_HOME", str(self.tmp_dir / "hf_home"))
            env.setdefault("TORCH_HOME", str(self.tmp_dir / "torch_home"))
            env.setdefault("TRANSFORMERS_CACHE", str(self.tmp_dir / "hf_home"))
        return env

    def _exec_bash(self, command: str) -> str:
        """Execute a bash command in the working directory."""
        env = os.environ.copy()
        env["PATH"] = f"{self.python_bin.parent}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = ""
        # Bind to specific GPU if set, otherwise no GPU for CPU tasks
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""  # CPU task — no GPU

        self._apply_tmp_dir(env)

        # Agent user isolation
        popen_kwargs = {}
        if self._agent_user:
            uid, gid = self._agent_user
            popen_kwargs["user"] = uid
            popen_kwargs["group"] = gid
            env["HOME"] = str(self.work_dir)

        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                cwd=str(self.work_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,  # Create new process group for clean kill
                **popen_kwargs,
            )

            # Route training-like commands through preflight profiler
            if self._is_likely_training(command):
                return self._exec_with_preflight(proc, env)
            else:
                return self._exec_simple(proc)
        except Exception as e:
            return f"[ERROR: {str(e)}]"

    def _exec_python(self, code: str) -> str:
        """Execute Python code using the kaggle environment."""
        # Write code to temp file and execute
        code_file = self.work_dir / "_react_tmp.py"
        code_file.write_text(code)

        env = os.environ.copy()
        env["PATH"] = f"{self.python_bin.parent}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = ""
        # Bind to specific GPU if set, otherwise no GPU for CPU tasks
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""  # CPU task — no GPU

        self._apply_tmp_dir(env)

        # Agent user isolation
        popen_kwargs = {}
        if self._agent_user:
            uid, gid = self._agent_user
            popen_kwargs["user"] = uid
            popen_kwargs["group"] = gid
            env["HOME"] = str(self.work_dir)
            # Ensure code file is readable by agent
            try:
                os.chown(str(code_file), uid, gid)
            except OSError:
                pass

        try:
            proc = subprocess.Popen(
                [str(self.python_bin), str(code_file)],
                cwd=str(self.work_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
                **popen_kwargs,
            )

            # Python code is always potentially training — use preflight
            return self._exec_with_preflight(proc, env)
        except Exception as e:
            return f"[ERROR: {str(e)}]"

    # ---- Preflight Profiler Methods ----

    def _is_likely_training(self, command: str) -> bool:
        """Determine if command is likely a long-running training task."""
        indicators = [
            "python", ".py", "train", "solution", "torch",
            "tensorflow", "lightgbm", "xgboost", "sklearn",
        ]
        cmd_lower = command.lower()
        return any(ind in cmd_lower for ind in indicators)

    def _exec_simple(self, proc) -> str:
        """Fast path for simple commands (no preflight)."""
        try:
            stdout, stderr = proc.communicate(timeout=self.exec_timeout)
        except subprocess.TimeoutExpired:
            self._kill_proc_group(proc)
            return (f"[TIMEOUT: command exceeded {self.exec_timeout}s limit]\n"
                    f"IMPORTANT: Your script was killed because it took too long. "
                    f"Do NOT retry the same approach. Instead: "
                    f"(1) Use a smaller/faster model, "
                    f"(2) Reduce training data size, "
                    f"(3) Use fewer epochs/folds, or "
                    f"(4) Switch to a completely different approach.")
        output = stdout
        if stderr:
            output += "\n[STDERR]\n" + stderr
        if proc.returncode != 0:
            output += f"\n[EXIT CODE: {proc.returncode}]"
        return output or "(no output)"

    def _exec_with_preflight(self, proc, env: dict) -> str:
        """Two-phase execution: wait for training to start, then profile GPU.

        Phase A: Wait up to PREFLIGHT_TRAIN_WAIT seconds for training to begin.
                 Preprocessing (data loading, caching) is expected and NOT penalized.
                 Training is detected by GPU memory > threshold or training keywords in output.
        Phase B: Once training detected, profile GPU for PREFLIGHT_SECS seconds.
                 Low GPU util during training = data loading bottleneck → kill with advice.
        CPU tasks (gpu_id=None): skip GPU profiling entirely, just run with timeout.
        """
        # CPU tasks: skip GPU profiling — no GPU to profile
        if self.gpu_id is None:
            return self._exec_simple(proc)
        stdout_lines, stderr_lines = [], []
        t_out = Thread(target=_pipe_reader, args=(proc.stdout, stdout_lines), daemon=True)
        t_err = Thread(target=_pipe_reader, args=(proc.stderr, stderr_lines), daemon=True)
        t_out.start()
        t_err.start()

        POLL = 10  # seconds between samples
        start = time.time()
        training_detected = False
        train_start = None
        train_samples = []  # GPU samples collected during training phase

        # Combined wait + profile loop
        while True:
            elapsed = time.time() - start

            if proc.poll() is not None:
                break  # process ended

            # Phase A: waiting for training — allow up to PREFLIGHT_TRAIN_WAIT
            if not training_detected and elapsed > PREFLIGHT_TRAIN_WAIT:
                logger.info("Preflight: training not detected in %ds, running with timeout only", PREFLIGHT_TRAIN_WAIT)
                break

            # Phase B: profiling training — collect for PREFLIGHT_SECS
            if training_detected and (time.time() - train_start) >= PREFLIGHT_SECS:
                break

            time.sleep(POLL)
            sample = self._sample_gpu(env)

            if not training_detected:
                output_tail = "".join(stdout_lines)[-5000:]
                if self._is_training_active(sample, output_tail):
                    training_detected = True
                    train_start = time.time()
                    logger.info("Preflight: training phase detected, starting GPU profiling")
                    if sample:
                        train_samples.append(sample)
            else:
                if sample:
                    train_samples.append(sample)

        # Process already finished
        if proc.poll() is not None:
            t_out.join(5)
            t_err.join(5)
            return self._build_output(stdout_lines, stderr_lines, proc.returncode)

        # Training never detected → CPU task or long preprocessing, just run with timeout
        if not training_detected:
            remaining = self.exec_timeout - (time.time() - start)
            try:
                proc.wait(timeout=max(remaining, 10))
            except subprocess.TimeoutExpired:
                self._kill_proc_group(proc)
                t_out.join(5)
                t_err.join(5)
                return (
                    f"[TIMEOUT: execution exceeded {self.exec_timeout}s]\n"
                    f"Note: No GPU training was detected during execution.\n"
                    f"If this is a CPU task (LightGBM/XGBoost), consider using fewer features or data.\n"
                    f"If this should use GPU, ensure model.to('cuda') and data.to('cuda').\n\n"
                    f"Partial output:\n{''.join(stdout_lines)[-3000:]}"
                )
            t_out.join(5)
            t_err.join(5)
            return self._build_output(stdout_lines, stderr_lines, proc.returncode)

        # Training detected — analyze GPU utilization during training phase
        diagnostic = self._analyze_preflight(train_samples, "".join(stdout_lines), env)

        if diagnostic["action"] == "kill":
            if self._preflight_passed:
                # A previous step already validated high GPU utilization — this step is likely
                # inference-only (loading weights into GPU memory triggers training detection,
                # but inference util is bursty/low). Don't kill; let it run to completion.
                logger.info(
                    f"Preflight: would kill (GPU util={diagnostic['avg_util']:.0f}%) "
                    f"but preflight already passed earlier — treating as inference step, skipping kill"
                )
                diagnostic["action"] = "continue"
            else:
                logger.warning(
                    f"Preflight KILL (training phase): GPU util={diagnostic['avg_util']:.0f}%, "
                    f"reasons={diagnostic['reasons']}"
                )
                self._kill_proc_group(proc)
                t_out.join(5)
                t_err.join(5)
                return self._build_preflight_abort_msg("".join(stdout_lines), diagnostic)

        if diagnostic["action"] != "kill":
            self._preflight_passed = True  # Mark: this agent has proven GPU training works

        # Continue to full timeout
        advice = diagnostic.get("advice", "")
        remaining = self.exec_timeout - (time.time() - start)
        try:
            proc.wait(timeout=max(remaining, 10))
        except subprocess.TimeoutExpired:
            logger.warning("Full timeout after passing preflight")
            self._kill_proc_group(proc)
            t_out.join(5)
            t_err.join(5)
            return self._build_timeout_msg("".join(stdout_lines), diagnostic)

        t_out.join(5)
        t_err.join(5)
        result = self._build_output(stdout_lines, stderr_lines, proc.returncode)
        if advice:
            result = f"[GPU OPTIMIZATION HINT: {advice}]\n\n{result}"
        return result

    def _is_training_active(self, gpu_sample, output_tail: str) -> bool:
        """Detect if GPU training has started based on GPU state and stdout."""
        # Check GPU memory or utilization
        if gpu_sample:
            max_mem = max(g["mem_used"] for g in gpu_sample)
            max_util = max(g["util"] for g in gpu_sample)
            # Require utilization > 10%, OR both high memory AND some compute (>5% util).
            # Memory-only check (original: max_mem > threshold) causes false positives for
            # inference-only scripts that load a large model but do no backprop.
            if max_util > 10:
                return True
            if max_mem > PREFLIGHT_TRAIN_MEM_THR and max_util > 5:
                return True
        # Check output for training keywords
        if re.search(
            r'Epoch\s+\d|loss[=:]\s*[\d.]|training.*start|\.fit\('
            r'|backward\(\)|optimizer\.step|step\s+\d+.*loss'
            r'|Train Loss|Val Loss|train_loss|Batch\s+\d+/\d+',
            output_tail, re.I,
        ):
            return True
        return False

    def _sample_gpu(self, env: dict):
        """Sample current GPU utilization and memory via nvidia-smi."""
        cvd = env.get("CUDA_VISIBLE_DEVICES", "0")
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                return None
            gpu_indices = set(int(x.strip()) for x in cvd.split(",") if x.strip())
            results = []
            for line in r.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4 and int(parts[0]) in gpu_indices:
                    results.append({
                        "util": int(parts[1]),
                        "mem_used": int(parts[2]),
                        "mem_total": int(parts[3]),
                    })
            return results if results else None
        except Exception:
            return None

    def _analyze_preflight(self, gpu_samples, output: str, env: dict) -> dict:
        """Analyze preflight data and decide: kill / continue / advise."""
        # Flatten GPU metrics across all samples
        all_utils = [g["util"] for s in gpu_samples if s for g in s]
        all_mems = [g["mem_used"] for s in gpu_samples if s for g in s]
        all_totals = [g["mem_total"] for s in gpu_samples if s for g in s]

        avg_util = mean(all_utils) if all_utils else 0
        avg_mem = mean(all_mems) if all_mems else 0
        max_total = max(all_totals) if all_totals else 81000
        mem_pct = avg_mem / max_total * 100 if max_total > 0 else 0

        # Parse training progress from stdout
        progress = self._parse_training_progress(output)

        # Decision logic
        action = "continue"
        reasons = []
        suggestions = []

        if avg_util < 5 and avg_mem < 2000:
            action = "kill"
            reasons.append(f"GPU completely idle (util={avg_util:.0f}%, mem={avg_mem:.0f}MB)")
            suggestions.append(
                "Code is not using GPU at all. Ensure model.to('cuda') and data.to('cuda')."
            )
        elif avg_util < PREFLIGHT_MIN_GPU_UTIL:
            action = "kill"
            reasons.append(f"GPU severely underutilized (util={avg_util:.0f}%)")
            suggestions.extend([
                "Data loading is the bottleneck. PRE-CACHE all data as numpy arrays BEFORE training.",
                f"Increase batch_size — you have {max_total}MB per GPU, use it.",
                "Use num_workers=8+ in DataLoader for parallel data loading.",
                "Use torch.cuda.amp (mixed precision) for ~2x speedup.",
            ])

        if progress.get("est_total_secs") and progress["est_total_secs"] > self.exec_timeout * 0.9:
            action = "kill"
            reasons.append(
                f"Estimated {progress['est_total_secs']/60:.0f}min > budget {self.exec_timeout/60:.0f}min"
            )
            suggestions.append(
                "Reduce folds, epochs, or image size to fit within time budget."
            )

        advice = ""
        if mem_pct < 20 and avg_mem > 500 and action == "continue":
            advice = (
                f"Only {mem_pct:.0f}% GPU memory used ({avg_mem:.0f}/{max_total}MB). "
                f"Consider larger batch_size or larger model for better results."
            )

        return {
            "action": action,
            "reasons": reasons,
            "suggestions": suggestions,
            "avg_util": avg_util,
            "avg_mem": avg_mem,
            "mem_pct": mem_pct,
            "max_total": max_total,
            "progress": progress,
            "advice": advice,
        }

    def _parse_training_progress(self, output: str) -> dict:
        """Parse training output to estimate total time."""
        # Match "Epoch 1/10", "epoch 2/5", etc.
        epoch_match = re.findall(r'[Ee]poch\s+(\d+)/(\d+)', output)
        if epoch_match:
            current, total = int(epoch_match[-1][0]), int(epoch_match[-1][1])
            # Try to find per-epoch timing
            epoch_times = re.findall(r'[Ee]poch\s+\d+.*?(\d+\.?\d*)s', output)
            if epoch_times:
                avg_epoch_sec = float(epoch_times[-1])
                est_total = avg_epoch_sec * total
                return {
                    "current_epoch": current,
                    "total_epochs": total,
                    "avg_epoch_sec": avg_epoch_sec,
                    "est_total_secs": est_total,
                }
            return {"current_epoch": current, "total_epochs": total}
        return {}

    def _kill_proc_group(self, proc):
        """Kill entire process group for clean GPU process cleanup."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait()

    def _build_output(self, stdout_lines: list, stderr_lines: list, returncode: int) -> str:
        """Build standard output string from collected lines."""
        output = "".join(stdout_lines)
        stderr_text = "".join(stderr_lines)
        if stderr_text:
            output += "\n[STDERR]\n" + stderr_text
        if returncode != 0:
            output += f"\n[EXIT CODE: {returncode}]"
        return output or "(no output)"

    def _build_preflight_abort_msg(self, partial_output: str, diagnostic: dict) -> str:
        """Build rich diagnostic message for preflight abort."""
        reasons_str = "\n  - ".join(diagnostic["reasons"])
        suggestions_str = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(diagnostic["suggestions"])
        )
        progress = diagnostic["progress"]
        progress_str = "  - No training progress detected in output"
        if progress.get("current_epoch"):
            progress_str = f"  - Detected: Epoch {progress['current_epoch']}/{progress.get('total_epochs', '?')}"
            if progress.get("est_total_secs"):
                progress_str += f", estimated total: {progress['est_total_secs']/60:.0f}min"

        msg = (
            f"[PREFLIGHT ABORT — GPU underutilized DURING TRAINING phase]\n\n"
            f"Your script started training (GPU memory was allocated), but GPU compute\n"
            f"utilization is very low. This means data loading is NOT overlapping with\n"
            f"GPU compute — the GPU is idle waiting for the next batch.\n\n"
            f"GPU Diagnostics (measured during training, not preprocessing):\n"
            f"  - Average GPU utilization: {diagnostic['avg_util']:.0f}% (target: >50%)\n"
            f"  - Average GPU memory: {diagnostic['avg_mem']:.0f}MB / {diagnostic['max_total']}MB "
            f"({diagnostic['mem_pct']:.1f}%)\n"
            f"  - Issue: {reasons_str}\n\n"
            f"Training Progress (from output):\n"
            f"{progress_str}\n\n"
            f"REQUIRED OPTIMIZATIONS — data loading is the bottleneck:\n"
            f"{suggestions_str}\n\n"
        )

        if partial_output.strip():
            # Include last portion of output for context
            tail = partial_output[-2000:] if len(partial_output) > 2000 else partial_output
            msg += f"Partial output (last chars):\n{tail}"

        return msg

    def _build_timeout_msg(self, partial_output: str, diagnostic: dict) -> str:
        """Build timeout message with GPU context from preflight."""
        advice_parts = []
        if diagnostic.get("advice"):
            advice_parts.append(diagnostic["advice"])
        if diagnostic["avg_util"] < 50:
            advice_parts.append(
                f"GPU utilization was only {diagnostic['avg_util']:.0f}% during preflight. "
                f"Pre-cache data and increase batch_size."
            )

        advice_str = " ".join(advice_parts)
        msg = (
            f"[TIMEOUT: execution exceeded {self.exec_timeout}s limit]\n"
            f"GPU context from preflight: util={diagnostic['avg_util']:.0f}%, "
            f"mem={diagnostic['avg_mem']:.0f}/{diagnostic['max_total']}MB\n"
        )
        if advice_str:
            msg += f"Optimization hint: {advice_str}\n"
        msg += (
            f"IMPORTANT: Do NOT retry the same approach. Instead: "
            f"(1) Pre-cache all data before training, "
            f"(2) Use a smaller/faster model, "
            f"(3) Reduce epochs/folds, or "
            f"(4) Switch to a completely different approach."
        )

        if partial_output.strip():
            tail = partial_output[-2000:] if len(partial_output) > 2000 else partial_output
            msg += f"\n\nPartial output (last chars):\n{tail}"

        return msg

    def _call_eval_score(self, submission_path: str):
        """Call eval_score_fn and return (score, is_lower_better), or (None, None) on error."""
        try:
            result = self.eval_score_fn(submission_path)
            if result is None:
                return None, None
            score, is_lower = result
            return score, is_lower
        except Exception as e:
            logger.warning(f"eval_score_fn failed: {e}")
            return None, None

    def _run_format_check(self, submission_path: str) -> Optional[str]:
        """Run format_check_fn if available. Returns issue string or None."""
        if not self.format_check_fn:
            return None
        try:
            return self.format_check_fn(submission_path)
        except Exception as e:
            logger.warning(f"format_check_fn failed: {e}")
            return None

    def _handle_submit_with_score(self, submission_path: str,
                                   prev_best: Optional[float],
                                   is_lower: Optional[bool]) -> str:
        """Handle submit when eval scoring is available — return score as observation."""
        if not submission_path or not Path(submission_path).exists():
            return f"Submission file NOT found at {submission_path}. Please create it first."

        # Run format check first — give specific feedback before score
        fmt_issues = self._run_format_check(submission_path)
        if fmt_issues:
            return (
                f"{fmt_issues}\n\n"
                f"Fix these format issues and resubmit. "
                f"The submission was not scored due to format errors."
            )

        score, new_is_lower = self._call_eval_score(submission_path)

        if score is None:
            return (
                f"Submission saved at {submission_path}. "
                f"Eval scoring unavailable (service down or eval format mismatch). "
                f"You can keep improving and submit again."
            )

        direction = "lower is better" if new_is_lower else "higher is better"
        msg = f"Submission scored. Eval score: {score:.6g} ({direction})."

        if prev_best is not None:
            improved = (new_is_lower and score < prev_best) or (not new_is_lower and score > prev_best)
            if improved:
                delta = abs(score - prev_best)
                msg += f" IMPROVED from {prev_best:.6g} (+{delta:.6g}). This is now your best model — keep going!"
            else:
                delta = abs(score - prev_best)
                msg += f" No improvement vs best {prev_best:.6g} (diff {delta:.6g}). Try a different approach."
        else:
            msg += " First submission recorded. Keep improving!"

        msg += f"\nYou can continue training and submit again to improve your score."
        return msg

    def _handle_submit(self, path: str) -> str:
        """Handle submit action — verify the file exists."""
        resolved = self._resolve_path(path)
        if not resolved or not Path(resolved).exists():
            return f"Submission file NOT found at {path}. Please create it first."
        fmt_issues = self._run_format_check(resolved)
        if fmt_issues:
            return (
                f"Submission file found at {resolved}.\n{fmt_issues}\n"
                f"Fix these format issues before final submission."
            )
        return f"Submission file found at {resolved}. Run complete."

    def _resolve_path(self, path: str) -> Optional[str]:
        """Resolve a path relative to work_dir."""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(self.work_dir / p)

    def _find_submission(self) -> Optional[str]:
        """Try to find submission.csv in the working directory and parent dirs."""
        candidates = [
            self.work_dir / "submission.csv",
            self.work_dir / "output" / "submission.csv",
            self.work_dir / "submissions" / "submission.csv",
        ]
        # Also check parent directory (agent may write to competition root)
        parent = self.work_dir.parent
        candidates.extend([
            parent / "submission.csv",
            parent / "output" / "submission.csv",
        ])
        for c in candidates:
            try:
                if c.exists():
                    return str(c)
            except (PermissionError, OSError):
                continue
        # Search recursively in work_dir
        for f in self.work_dir.rglob("submission.csv"):
            return str(f)
        # Search in parent dir too
        for f in parent.rglob("submission.csv"):
            # Exclude HCE sample submissions
            if "hce" not in str(f) and "sample" not in str(f):
                return str(f)
        return None

    def _collect_solution_code(self) -> str:
        """Collect the main solution code from the working directory.

        Prefers the most recently modified .py file (which is likely the one that
        produced the submission), over fixed name patterns.
        """
        # Find the most recently modified .py file (most likely the successful one)
        py_files = sorted(
            self.work_dir.glob("*.py"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for f in py_files:
            if not f.name.startswith("_react_tmp"):
                return f.read_text()

        return ""
