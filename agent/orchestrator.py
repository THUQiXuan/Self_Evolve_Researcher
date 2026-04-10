"""SER — Main Orchestrator (parallel multi-instance version)."""

import json
import time
import shutil
import socket
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
from filelock import FileLock, Timeout as FileLockTimeout

from config import (
    DEFAULT_TIME_LIMIT, PYTHON_BIN, CODE_EXEC_TIMEOUT,
)
from llm import LLMClient
from competition import CompetitionManager
from react_agent import ReActAgent
from evolution import EvolutionManager
from database import Program, ProgramDatabase
from prompts.system import build_system_prompt
from prompts.mutation import build_initial_prompt, build_mutation_prompt
from prompts.crossover import build_crossover_prompt
from prompts.planning import build_planning_prompt
from eda_agent import run_eda
from critic import run_critic
from tracer import init_langfuse, RunTrace

logger = logging.getLogger(__name__)


class Orchestrator:
    """Parallel multi-instance orchestrator for SER."""

    def __init__(
        self,
        competition_id: str,
        work_dir: Path,
        time_limit: int = DEFAULT_TIME_LIMIT,
        llm: Optional[LLMClient] = None,
        gpu_id: int = None,
        instance_id: int = None,
    ):
        self.competition_id = competition_id
        self.gpu_id = gpu_id
        # instance_id: explicit > gpu_id > 0
        self.instance_id = instance_id if instance_id is not None else (gpu_id if gpu_id is not None else 0)

        # Shared work_dir: work_dir/competition_id/
        self.work_dir = Path(work_dir) / competition_id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.time_limit = time_limit

        # Components
        self.llm = llm or LLMClient()
        self.comp = CompetitionManager(competition_id, self.work_dir)

        # No grading service — use direct CompetitionManager grading
        self.grader = None

        # Shared programs directory (all instances read/write here)
        self.db = ProgramDatabase(self.work_dir / "programs")

        # Seed per instance for diversity
        evo_seed = 42 + self.instance_id
        self.evo = EvolutionManager(self.db, seed=evo_seed)

        # Instance-isolated iteration directory
        self.inst_dir = self.work_dir / f"inst_{self.instance_id}"
        self.inst_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.start_time = None
        self.iteration = 0
        self.data_profile = ""
        self.stop_requested = False

        # Async critic state
        self._critic_executor = None
        self._pending_critic = None
        self._last_critic_summary = ""

        # Langfuse tracing
        init_langfuse()
        self._run_trace: Optional[RunTrace] = None

    def run(self) -> dict:
        """Run the full evolutionary search loop."""
        self.start_time = time.time()
        logger.info(f"=== Starting SER for {self.competition_id} "
                    f"(instance={self.instance_id}, gpu={self.gpu_id}, limit={self.time_limit}s) ===")

        # Create Langfuse trace for this run; propagate trace_id to LLM client
        self._run_trace = RunTrace(self.competition_id, self.instance_id, self.time_limit)
        if self._run_trace.trace_id:
            self.llm.trace_id = self._run_trace.trace_id

        # Step 1: Load pre-split HCE data (must be prepared by presplit_all.py)
        hce_dir = self.work_dir / "hce"
        hce_done = hce_dir / ".split.done"
        # Private dir for eval answers — stored in workspace (no separate service)
        self._private_eval_dir = (
            Path.home() / ".ser_grading" / self.competition_id
        )
        if hce_done.exists():
            logger.info("HCE split already prepared, loading...")
            # Run leakage check retroactively for already-split competitions
            # (guarded by .leakage.done marker, so it only runs once)
            self.comp._check_and_fix_leakage()
        else:
            # Fallback: prepare inline if not pre-split (backward compat)
            logger.warning("HCE split not pre-prepared, running inline (use presplit_all.py first)")
            hce_dir.mkdir(parents=True, exist_ok=True)
            hce_lock = hce_dir / ".split.lock"
            with FileLock(str(hce_lock), timeout=600):
                if not hce_done.exists():
                    self.comp.prepare_hce_split()
                    # Store eval answers in private dir, NOT in hce/eval/
                    import os as _os
                    self._private_eval_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        _os.chmod(str(self._private_eval_dir.parent), 0o700)
                        _os.chmod(str(self._private_eval_dir), 0o700)
                    except OSError:
                        pass
                    self.comp.prepare_eval_data(answers_dir=self._private_eval_dir)
                    hce_done.touch()

        # Load eval info — also propagate to comp object so get_context_for_agent() shows eval section
        eval_info_path = hce_dir / "eval" / "eval_info.json"
        if eval_info_path.exists():
            self._eval_info = json.loads(eval_info_path.read_text())
            self.comp._eval_info = self._eval_info   # propagate so context includes eval section
            logger.info(f"Eval data loaded: {self._eval_info.get('n_total', '?')} samples")
        else:
            self._eval_info = None
            logger.warning("Eval data not available — will use test-only grading")

        # Step 2: Pipeline smoke test — verify grading works end-to-end (once per competition)
        self._run_pipeline_smoke_test()

        # Step 3: EDA — data analysis (only once, with lock)
        logger.info("Running EDA data analysis...")
        eda_lock = self.work_dir / ".eda.lock"
        eda_done = self.work_dir / ".eda.done"
        with FileLock(str(eda_lock), timeout=1800):  # 30 min — EDA on large datasets can take 15+ min
            if not eda_done.exists():
                try:
                    sample_sub = pd.read_csv(self.comp.sample_submission_path, nrows=3)
                    self.data_profile = run_eda(
                        llm=self.llm,
                        work_dir=self.work_dir,
                        data_dir=self.comp.dtrain_dir,
                        description=self.comp.description,
                        sample_submission_preview=sample_sub.to_string(index=False),
                        gpu_id=self.gpu_id,
                    )
                except Exception as e:
                    logger.error(f"EDA failed, continuing without data profile: {e}")
                    self.data_profile = ""
                eda_done.touch()
                # Write profile to disk (even if empty) so other instances can read it
                profile_path = self.work_dir / "data_profile.txt"
                profile_path.write_text(self.data_profile)
            else:
                # Load cached profile
                profile_path = self.work_dir / "data_profile.txt"
                self.data_profile = profile_path.read_text() if profile_path.exists() else ""
        logger.info(f"Data profile loaded ({len(self.data_profile)} chars)")

        # Build device-aware system prompt
        system_prompt = build_system_prompt(gpu_id=self.gpu_id)

        # Step 3: Evolutionary loop
        while self._time_remaining() > 0:
            self.iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {self.iteration} | Elapsed: {self._elapsed():.0f}s | "
                       f"Remaining: {self._time_remaining():.0f}s | Population: {self.db.size()}")
            logger.info(f"{'='*60}")

            # Check if stop requested (SIGTERM) or not enough time
            if self.stop_requested:
                logger.info("Stop requested (SIGTERM), exiting loop.")
                break
            if self._time_remaining() < 600:
                logger.info("Less than 10 minutes remaining, stopping.")
                break

            # Reload shared population from disk (see other instances' programs)
            self.db.reload_from_disk()

            task = self.evo.next_task()
            operation = task["operation"]
            parents = task["parents"]

            # Log decision details
            if parents:
                logger.info(f"Decision: op={operation}, parents=[{', '.join(p.id + '(' + str(p.search_score) + ')' for p in parents)}]")
                if operation == "crossover":
                    logger.info(f"  Parent A [{parents[0].id}]: {self._summarize_approach(parents[0].code)} | score={parents[0].search_score} | PR={self.comp.compute_percentile_rank(parents[0].search_score):.1f}%")
                    logger.info(f"  Parent B [{parents[1].id}]: {self._summarize_approach(parents[1].code)} | score={parents[1].search_score} | PR={self.comp.compute_percentile_rank(parents[1].search_score):.1f}%")
                elif operation == "mutation":
                    logger.info(f"  Parent [{parents[0].id}]: {self._summarize_approach(parents[0].code)} | score={parents[0].search_score} | PR={self.comp.compute_percentile_rank(parents[0].search_score):.1f}%")
            else:
                logger.info(f"Decision: op={operation}, parents=[]")

            # Create isolated working directory for this iteration (per-instance)
            iter_dir = self.inst_dir / f"iter_{self.iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # Build prompt
            competition_context = self.comp.get_context_for_agent()

            if operation == "init":
                # Planning call (single LLM call, not ReAct)
                strategy = self._plan_strategy()
                task_prompt = build_initial_prompt(
                    competition_context,
                    gpu_id=self.gpu_id,
                    data_profile=self.data_profile,
                    strategy=strategy,
                )
                # If population exists, share history so init doesn't repeat failed approaches
                if self.db.size() > 0:
                    population_history = self._build_population_history()
                    if population_history:
                        task_prompt += f"\n\n**Previous approaches tried:**\n{population_history}\n\nTry a DIFFERENT approach from those listed above.\n"
            elif operation == "mutation":
                parent_code = parents[0].code
                parent_score = parents[0].search_score
                # If parent code is empty/tiny, fall back to init prompt
                if not parent_code or len(parent_code.strip()) < 50:
                    logger.warning(f"Parent {parents[0].id} has empty/tiny code, falling back to init")
                    task_prompt = build_initial_prompt(
                        competition_context, gpu_id=self.gpu_id,
                        data_profile=self.data_profile,
                    )
                # If all scores are 0 or near-zero AND higher-is-better, the format is likely wrong
                # Do NOT reset for lower-is-better metrics (e.g. Levenshtein, KL divergence)
                # where 0.0 is a perfect score.
                elif (parent_score is not None and parent_score == 0.0
                      and not self.comp.is_lower_better):
                    logger.warning(f"Parent score is 0.0 (higher-is-better) — likely wrong submission format, resetting to init")
                    task_prompt = build_initial_prompt(
                        competition_context, gpu_id=self.gpu_id,
                        data_profile=self.data_profile,
                    )
                    task_prompt += ("\n\n**CRITICAL WARNING**: All previous submissions scored 0.0, "
                                   "which means the submission FORMAT is wrong. Common issues:\n"
                                   "- IDs formatted as floats (e.g., '123.0') instead of strings/integers\n"
                                   "- Wrong column names or missing columns\n"
                                   "- Wrong number of rows in submission\n"
                                   "- Predictions in wrong format (probabilities vs classes)\n"
                                   "FIRST: carefully compare your submission format with sample_submission.csv. "
                                   "THEN: fix the format issue before trying to improve the model.\n")
                else:
                    # Build population history for context
                    population_history = self._build_population_history()
                    best = self.db.get_best_by_val()
                    best_pr = self.comp.compute_percentile_rank(best.search_score) if best else 0
                    critic_fb = self._collect_critic_feedback()
                    task_prompt = build_mutation_prompt(
                        competition_context,
                        parent_code=parent_code,
                        parent_score=parent_score,
                        is_lower_better=self.comp.is_lower_better,
                        population_history=population_history,
                        best_percentile_rank=best_pr,
                        gpu_id=self.gpu_id,
                        data_profile=self.data_profile,
                        critic_feedback=critic_fb,
                    )
            elif operation == "crossover":
                population_history = self._build_population_history()
                best = self.db.get_best_by_val()
                best_pr = self.comp.compute_percentile_rank(best.search_score) if best else 0
                critic_fb = self._collect_critic_feedback()
                task_prompt = build_crossover_prompt(
                    competition_context,
                    parent_a_code=parents[0].code,
                    parent_a_score=parents[0].search_score,
                    parent_b_code=parents[1].code,
                    parent_b_score=parents[1].search_score,
                    is_lower_better=self.comp.is_lower_better,
                    gpu_id=self.gpu_id,
                    population_history=population_history,
                    best_percentile_rank=best_pr,
                    data_profile=self.data_profile,
                    critic_feedback=critic_fb,
                )
            else:
                raise ValueError(f"Unknown operation: {operation}")

            # Run ReAct agent with time-aware timeout
            remaining = self._time_remaining()
            exec_timeout = min(int(remaining * 0.8), CODE_EXEC_TIMEOUT)  # Leave 20% margin

            # Build eval scoring callback so the agent can get live feedback on submit()
            def _make_eval_score_fn(comp, private_eval_dir):
                def eval_score_fn(submission_path):
                    eval_path = Path(submission_path).parent / "eval_submission.csv"
                    if not eval_path.exists():
                        return None
                    score = comp.grade_eval_submission(
                        eval_path, 'search', answers_dir=private_eval_dir)
                    if score is None:
                        return None
                    return score, comp.is_lower_better
                return eval_score_fn

            eval_score_fn = _make_eval_score_fn(self.comp, self._private_eval_dir)

            # Build format-check callback: runs on every submit(), returns issue string or None
            def _make_format_check_fn(comp):
                def format_check_fn(submission_path):
                    return comp.check_submission_format(Path(submission_path))
                return format_check_fn

            format_check_fn = _make_format_check_fn(self.comp)

            # Per-iteration TMPDIR: isolated per machine+instance+iteration, auto-cleaned after
            _host = socket.gethostname().split(".")[0]  # short hostname to distinguish machines
            _tmp_base = Path("/prodcpfs/user/yuchen/mle-bench/tmp") / _host
            _tmp_base.mkdir(parents=True, exist_ok=True)
            iter_tmp = _tmp_base / f"ser_i{self.instance_id}_it{self.iteration}"
            iter_tmp.mkdir(exist_ok=True)
            # Make writable by aira_agent
            try:
                import os as _os
                _os.chmod(str(iter_tmp), 0o777)
            except OSError:
                pass

            agent = ReActAgent(self.llm, iter_dir, exec_timeout=max(exec_timeout, 120),
                               gpu_id=self.gpu_id, eval_score_fn=eval_score_fn,
                               format_check_fn=format_check_fn,
                               tmp_dir=iter_tmp)
            logger.info(f"Running ReAct agent (op={operation}, exec_timeout={exec_timeout}s)...")

            # Add time budget to task prompt
            time_note = f"\n\n**Time budget: {int(remaining/60)} minutes remaining.** Keep training fast.\n"
            task_prompt += time_note

            # Langfuse iteration span
            parent_scores = [p.search_score for p in parents if p.search_score is not None]
            iter_span = self._run_trace.start_iteration(self.iteration, operation, parent_scores)

            try:
                result = agent.run(system_prompt, task_prompt, iter_span=iter_span)
            except Exception as e:
                logger.error(f"Iteration {self.iteration} crashed: {e}")
                iter_span.end(score=None, percentile=None, steps=0, elapsed=0)
                continue
            finally:
                # Clean up this iteration's tmp dir to free disk space
                try:
                    shutil.rmtree(str(iter_tmp), ignore_errors=True)
                    logger.debug(f"Cleaned up {iter_tmp}")
                except Exception:
                    pass

            logger.info(f"Agent completed: success={result['success']}, "
                       f"steps={result['steps']}, elapsed={result['elapsed']:.0f}s")

            if not result["success"]:
                logger.warning(f"Iteration {self.iteration} failed: no submission produced")
                # Still add to DB with None scores
                program = Program(
                    code=result["code"],
                    parent_ids=[p.id for p in parents],
                    operation=operation,
                    search_score=None,
                    val_score=None,
                    is_lower_better=self.comp.is_lower_better,
                    metadata={
                        "iter": self.iteration,
                        "steps": result["steps"],
                        "agent_elapsed": result["elapsed"],
                        "instance_id": self.instance_id,
                        "operation": operation,
                        "parent_scores": {p.id: p.search_score for p in parents},
                        "parent_approaches": {p.id: self._summarize_approach(p.code) for p in parents},
                            "population_size_at_start": self.db.size(),
                        "time_remaining_at_start": self._time_remaining(),
                    },
                )
                self.db.add(program)
                continue

            # Grade submission
            submission_path = Path(result["submission_path"])

            # Copy submission to a permanent location (instance-aware)
            perm_submission = self.work_dir / "programs" / f"submission_inst{self.instance_id}_iter{self.iteration}.csv"
            shutil.copy2(submission_path, perm_submission)

            # Full score from test set (for percentile rank) — direct grading
            full_score = self.comp.grade_submission(submission_path)

            # Eval grading for search/val scores (from held-out training data)
            search_score = None
            val_score = None
            eval_submission_path = self._find_eval_submission(iter_dir)
            if eval_submission_path:
                search_score = self.comp.grade_eval_submission(
                    eval_submission_path, 'search', answers_dir=self._private_eval_dir)
                val_score = self.comp.grade_eval_submission(
                    eval_submission_path, 'val', answers_dir=self._private_eval_dir)
                if search_score is not None or val_score is not None:
                    logger.info(f"Eval scores: search={search_score}, val={val_score}")

            # Fallback: use full_score if eval grading not available
            if search_score is None:
                search_score = full_score
                logger.info("Eval search score unavailable, using full test score as fallback")
            if val_score is None:
                val_score = full_score
                logger.info("Eval val score unavailable, using full test score as fallback")

            # Sanity check: detect suspiciously perfect eval score that doesn't match
            # test score — a symptom of eval label leakage (agent read validation archives
            # containing ground-truth labels for the HCE eval split).
            # A genuine perfect eval score (0.0 for lower-is-better) would only occur if
            # the agent predicts every eval sequence exactly right, which would also
            # produce a very good test score — not a high test score like 0.92.
            if (search_score is not None and full_score is not None
                    and full_score != search_score):
                is_lb = self.comp.is_lower_better
                # For lower-is-better: suspiciously if eval=0.0 but test is far from 0
                # For higher-is-better: suspiciously if eval=perfect(1.0) but test is far
                perfect = 0.0 if is_lb else 1.0
                eval_is_perfect = abs(search_score - perfect) < 1e-6
                test_far_from_perfect = abs(full_score - perfect) > 0.1
                if eval_is_perfect and test_far_from_perfect:
                    logger.warning(
                        f"Eval score suspiciously perfect ({search_score}) while "
                        f"test score is {full_score:.4f} — possible eval label leakage. "
                        f"Falling back to full test score for selection."
                    )
                    search_score = full_score
                    val_score = full_score

            logger.info(f"Scores: search={search_score}, val={val_score}, full={full_score}")

            percentile = self.comp.compute_percentile_rank(full_score) if full_score is not None else 0

            # Add to population
            program = Program(
                code=result["code"],
                code_path=str(perm_submission),
                parent_ids=[p.id for p in parents],
                operation=operation,
                search_score=search_score,
                val_score=val_score,
                full_score=full_score,
                is_lower_better=self.comp.is_lower_better,
                metadata={
                    "iter": self.iteration,
                    "steps": result["steps"],
                    "agent_elapsed": result["elapsed"],
                    "instance_id": self.instance_id,
                    "operation": operation,
                    "parent_scores": {p.id: p.search_score for p in parents},
                    "parent_approaches": {p.id: self._summarize_approach(p.code) for p in parents},
                    "eval_graded": eval_submission_path is not None,
                    "population_size_at_start": self.db.size(),
                    "time_remaining_at_start": self._time_remaining(),
                },
            )
            self.db.add(program)

            logger.info(f"Result: iter={self.iteration} op={operation} "
                        f"search={search_score} val={val_score} full={full_score} PR={percentile:.1f}% "
                        f"parents=[{','.join(p.id for p in parents)}] steps={result['steps']} "
                        f"elapsed={result['elapsed']:.0f}s pop={self.db.size()}")

            # End Langfuse iteration span
            iter_span.end(
                score=full_score, percentile=percentile,
                steps=result["steps"], elapsed=result["elapsed"],
            )

            # Save intermediate result.json (so kill doesn't lose progress)
            self._save_intermediate_result()

            # Cleanup old iteration directories to free disk space
            self._cleanup_old_iterations()

            # Launch async critic for the next iteration's feedback
            if result["code"] and self.data_profile:
                self._launch_async_critic(result["code"])

        # Step 3: Final selection
        best = self.db.get_best_by_val()
        self.db.save_summary()

        result = {
            "competition_id": self.competition_id,
            "total_iterations": self.iteration,
            "population_size": self.db.size(),
            "total_elapsed": self._elapsed(),
            "llm_usage": self.llm.get_usage(),
        }

        if best:
            result["best_program_id"] = best.id
            result["best_search_score"] = best.search_score
            result["best_val_score"] = best.val_score
            result["best_full_score"] = best.full_score
            # Use full_score for percentile rank (evaluated on complete test set)
            score_for_pr = best.full_score if best.full_score is not None else best.val_score
            percentile = self.comp.compute_percentile_rank(score_for_pr) if score_for_pr is not None else -1.0
            result["percentile_rank"] = percentile
            logger.info(f"\n=== FINAL RESULT for {self.competition_id} ===")
            logger.info(f"Best program: {best.id}")
            logger.info(f"Search score: {best.search_score}")
            logger.info(f"Val score: {best.val_score}")
            logger.info(f"Full score: {best.full_score}")
            logger.info(f"Percentile rank: {percentile:.1f}%")
        else:
            result["best_program_id"] = None
            result["best_search_score"] = None
            result["best_val_score"] = None
            result["percentile_rank"] = -1.0
            logger.warning(f"No successful solutions for {self.competition_id}")

        # End Langfuse run trace
        if self._run_trace:
            self._run_trace.end(result)

        return result

    def _save_intermediate_result(self):
        """Save current best result to result.json (incremental, survives kill)."""
        try:
            best = self.db.get_best_by_val()
            if not best:
                return

            result = {
                "competition_id": self.competition_id,
                "total_iterations": self.iteration,
                "population_size": self.db.size(),
                "total_elapsed": self._elapsed(),
                "best_program_id": best.id,
                "best_search_score": best.search_score,
                "best_val_score": best.val_score,
                "best_full_score": best.full_score,
                "intermediate": True,
            }
            score_for_pr = best.full_score if best.full_score is not None else best.val_score
            result["percentile_rank"] = self.comp.compute_percentile_rank(score_for_pr) if score_for_pr else 0

            result_path = self.work_dir / "result.json"
            lock_path = self.work_dir / ".result.lock"
            with FileLock(str(lock_path), timeout=10):
                should_save = True
                if result_path.exists():
                    try:
                        with open(result_path) as f:
                            existing = json.load(f)
                        existing_pct = existing.get("percentile_rank", -1.0)
                        new_pct = result.get("percentile_rank", -1.0)

                        # If both have valid PR (>=0), compare by PR
                        if existing_pct is not None and existing_pct >= 0 and new_pct is not None and new_pct >= 0:
                            should_save = new_pct >= existing_pct
                        else:
                            # Fallback: compare raw scores (best_full_score)
                            existing_score = existing.get("best_full_score")
                            new_score = result.get("best_full_score")
                            if existing_score is not None and new_score is not None:
                                is_lower = self.comp.is_lower_better
                                should_save = (new_score <= existing_score) if is_lower else (new_score >= existing_score)
                            else:
                                should_save = new_score is not None
                    except (json.JSONDecodeError, IOError):
                        should_save = True

                if should_save:
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2, default=str)
                    logger.info(f"Intermediate result saved (score={result.get('best_full_score')}, PR={result.get('percentile_rank')})")
        except Exception as e:
            logger.warning(f"Failed to save intermediate result: {e}")

    # -------------------------------------------------------------------------
    # Pipeline smoke test
    # -------------------------------------------------------------------------

    def _make_random_submission(self, sample: pd.DataFrame,
                                id_override=None,
                                eval_input_df: "pd.DataFrame | None" = None) -> pd.DataFrame:
        """Generate a structurally valid random submission from sample_submission.

        When eval_input_df is provided its columns are used as the base for all
        shared columns (e.g. id + class for uw-madison) so that multi-column key
        competitions get correctly structured eval submissions instead of having
        their secondary key columns randomly scrambled.
        """
        rand = sample.copy()
        pred_cols_only: "list[str] | None" = None  # columns to randomize (None = all value cols)
        if eval_input_df is not None:
            # Use eval_input as base: preserve its key columns (id, class, …) and only
            # add/randomize the prediction columns that are NOT already in eval_input.
            rand = eval_input_df.copy()
            pred_cols_only = [col for col in sample.columns if col not in eval_input_df.columns]
            for col in pred_cols_only:
                rand[col] = None  # will be filled below
            rand = rand[sample.columns]  # reorder to match sample column order
        elif id_override is not None:
            # Reindex to match eval IDs, filling with random rows if needed
            n = len(id_override)
            rand = pd.concat([rand] * (n // len(rand) + 2), ignore_index=True).iloc[:n].copy()
            rand.iloc[:, 0] = id_override.values

        n_rows = len(rand)
        # Only randomize prediction columns — key columns from eval_input are preserved
        value_cols = pd.Index(pred_cols_only) if pred_cols_only is not None else rand.columns[1:]

        all_numeric = all(
            pd.api.types.is_float_dtype(sample[c].dtype) or
            pd.api.types.is_integer_dtype(sample[c].dtype)
            for c in value_cols
        )
        if len(value_cols) > 1 and all_numeric:
            # Likely multi-class probabilities — generate Dirichlet-distributed rows
            # so values are in (0, 1) and sum to 1 per row
            alpha = np.ones(len(value_cols))
            probs = np.random.dirichlet(alpha, size=n_rows)
            for i, col in enumerate(value_cols):
                rand[col] = probs[:, i]
        else:
            for col in value_cols:
                dtype = sample[col].dtype
                if pd.api.types.is_float_dtype(dtype):
                    rand[col] = np.random.random(n_rows)
                elif pd.api.types.is_integer_dtype(dtype):
                    unique = sample[col].dropna().unique()
                    rand[col] = np.random.choice(unique, n_rows) if len(unique) else 0
                else:
                    unique = sample[col].dropna().unique()
                    if len(unique):
                        rand[col] = np.random.choice(unique, n_rows)
        return rand

    def _run_pipeline_smoke_test(self):
        """Verify the full grading pipeline before spending time on training.

        Generates a random submission from sample_submission.csv and grades it
        via both the full grader and the eval grader (if configured).
        Runs exactly once per competition (guarded by a .smoke_test.done file).
        Diagnoses and logs specific failures so they can be fixed without
        wasting a full iteration.
        """
        smoke_done = self.work_dir / ".smoke_test.done"
        smoke_lock = self.work_dir / ".smoke_test.lock"

        try:
            _smoke_lock = FileLock(str(smoke_lock), timeout=120)
            _smoke_lock.acquire()
        except FileLockTimeout:
            # Another instance is running the smoke test; this instance skips it and continues.
            logger.info("Smoke test lock busy (another instance running it) — skipping for this instance")
            return

        try:
            if smoke_done.exists():
                logger.info("Pipeline smoke test already passed, skipping")
                return

            logger.info("=== Pipeline smoke test: verifying grading end-to-end ===")
            all_ok = True

            try:
                sample = pd.read_csv(self.comp.sample_submission_path)
            except Exception as e:
                logger.error(f"Smoke test: cannot read sample_submission: {e}")
                return

            smoke_path = self.work_dir / "_smoke_submission.csv"
            try:
                rand_sub = self._make_random_submission(sample)
                rand_sub.to_csv(smoke_path, index=False)

                # --- Format check ---
                fmt = self.comp.check_submission_format(smoke_path)
                if fmt:
                    logger.error(f"Smoke test: random submission has format issues:\n{fmt}")
                    all_ok = False
                else:
                    logger.info("Smoke test: format check OK")

                # --- Full grading ---
                full_score = self.comp.grade_submission(smoke_path)

                if full_score is None:
                    logger.warning(
                        "Smoke test: full grading returned None  "
                        "— grading pipeline may be broken (check grader or data paths)"
                    )
                    all_ok = False
                else:
                    pr = self.comp.compute_percentile_rank(full_score)
                    logger.info(
                        f"Smoke test: full grading OK "
                        f"(random score={full_score:.4g}, PR={pr:.1f}%)"
                    )

                # --- Eval grading ---
                if self._eval_info:
                    eval_input_path = Path(self._eval_info.get("eval_input_path", ""))
                    if eval_input_path.exists():
                        eval_smoke_path = self.work_dir / "_smoke_eval_submission.csv"
                        try:
                            eval_input = pd.read_csv(eval_input_path)
                            id_col = eval_input.columns[0]
                            rand_eval = self._make_random_submission(
                                sample, id_override=eval_input[id_col],
                                eval_input_df=eval_input
                            )
                            rand_eval.to_csv(eval_smoke_path, index=False)

                            eval_score = self.comp.grade_eval_submission(
                                eval_smoke_path, 'search',
                                answers_dir=self._private_eval_dir
                            )

                            if eval_score is None:
                                logger.warning(
                                    "Smoke test: eval grading returned None  "
                                    "— eval pipeline broken (check eval_info and answer files)"
                                )
                                all_ok = False
                            else:
                                logger.info(
                                    f"Smoke test: eval grading OK "
                                    f"(random eval score={eval_score:.4g})"
                                )
                        finally:
                            eval_smoke_path.unlink(missing_ok=True)
                    else:
                        logger.warning(
                            f"Smoke test: eval_input.csv not found at {eval_input_path}"
                        )
                        all_ok = False
                else:
                    logger.info("Smoke test: eval not configured (no eval_info), skipping eval check")

            except Exception as e:
                logger.error(f"Smoke test: unexpected error: {e}", exc_info=True)
                all_ok = False
            finally:
                smoke_path.unlink(missing_ok=True)

            if all_ok:
                smoke_done.touch()
                logger.info("=== Pipeline smoke test PASSED ===")
            else:
                logger.warning(
                    "=== Pipeline smoke test FAILED — see warnings above. "
                    "Continuing anyway but grading feedback may be unreliable. ==="
                )
        finally:
            _smoke_lock.release()

    def _cleanup_old_iterations(self):
        """Keep last 3 iter dirs, delete large files from older ones."""
        try:
            iter_dirs = sorted(self.inst_dir.glob("iter_*"))
            for old_dir in iter_dirs[:-3]:
                for f in old_dir.rglob("*"):
                    try:
                        if f.is_file() and f.suffix in ('.pth', '.pt', '.npy', '.npz', '.pkl', '.h5', '.bin') \
                           and f.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                            f.unlink()
                    except OSError:
                        continue
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def _plan_strategy(self) -> str:
        """Single LLM call for strategy planning (no ReAct needed)."""
        try:
            remaining_min = int(self._time_remaining() / 60)
            population_history = self._build_population_history()

            prompt = build_planning_prompt(
                data_profile=self.data_profile,
                competition_description=self.comp.description,
                gpu_available=(self.gpu_id is not None),
                time_budget_minutes=remaining_min,
                population_history=population_history,
            )

            # Collect critic feedback from previous iteration if available
            critic_feedback = self._collect_critic_feedback()
            if critic_feedback:
                prompt += f"\n\n### Critic Feedback (from previous solution)\n{critic_feedback}\n"

            strategy = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                system="You are an expert ML strategist. Output a clear, actionable plan.",
                temperature=0.7,
            )

            # Save strategy to disk for debugging
            strategy_path = self.inst_dir / f"strategy_iter{self.iteration:03d}.txt"
            strategy_path.write_text(strategy)
            logger.info(f"Strategy planned ({len(strategy)} chars) -> {strategy_path}")

            return strategy
        except Exception as e:
            logger.error(f"Strategy planning failed: {e}")
            return ""

    def _launch_async_critic(self, code: str):
        """Launch critic as background task (non-blocking)."""
        try:
            if self._critic_executor is None:
                self._critic_executor = ThreadPoolExecutor(max_workers=1)
            # Cancel previous if still pending
            if self._pending_critic and not self._pending_critic.done():
                self._pending_critic.cancel()
            self._pending_critic = self._critic_executor.submit(
                run_critic,
                llm=LLMClient(),  # Independent LLM client to avoid token counting conflicts
                code=code,
                data_profile=self.data_profile,
                competition_description=self.comp.description,
            )
        except Exception as e:
            logger.warning(f"Failed to launch async critic: {e}")

    def _collect_critic_feedback(self) -> str:
        """Collect critic results if available (non-blocking). Caches last result."""
        if self._pending_critic is None:
            return self._last_critic_summary
        if not self._pending_critic.done():
            return self._last_critic_summary
        try:
            critic_result = self._pending_critic.result(timeout=1)
            self._pending_critic = None
            self._last_critic_summary = critic_result["summary"]
            return self._last_critic_summary
        except Exception as e:
            logger.warning(f"Critic result collection failed: {e}")
            self._pending_critic = None
            return self._last_critic_summary

    def _build_population_history(self) -> str:
        """Build a summary of all previous attempts for the mutation prompt."""
        scored = self.db.get_scored_programs()
        if not scored:
            return ""
        lines = []
        better_dir = "lower" if self.comp.is_lower_better else "higher"
        lines.append(f"| # | Score | PR% | Operation | Key Approach |")
        lines.append(f"|---|-------|-----|-----------|-------------|")
        for i, p in enumerate(scored[:10]):  # Show top 10
            # Extract a brief description from code
            approach = self._summarize_approach(p.code)
            pr = self.comp.compute_percentile_rank(p.search_score)
            lines.append(f"| {i+1} | {p.search_score} | {pr:.0f}% | {p.operation} | {approach} |")
        lines.append(f"\n({better_dir} is better. Population size: {self.db.size()})")
        return "\n".join(lines)

    @staticmethod
    def _summarize_approach(code: str) -> str:
        """Extract a brief description of the approach from code."""
        if not code:
            return "unknown"
        code_lower = code.lower()
        approaches = []
        if "xgboost" in code_lower or "xgb." in code_lower:
            approaches.append("XGBoost")
        if "lightgbm" in code_lower or "lgb." in code_lower:
            approaches.append("LightGBM")
        if "randomforest" in code_lower:
            approaches.append("RandomForest")
        if "bert" in code_lower:
            approaches.append("BERT")
        if "roberta" in code_lower:
            approaches.append("RoBERTa")
        if "deberta" in code_lower:
            approaches.append("DeBERTa")
        if "efficientnet" in code_lower:
            approaches.append("EfficientNet")
        if "resnet" in code_lower:
            approaches.append("ResNet")
        if "ensemble" in code_lower or "blend" in code_lower:
            approaches.append("Ensemble")
        if "kfold" in code_lower or "cross_val" in code_lower or "stratifiedkfold" in code_lower:
            approaches.append("CV")
        if "transformers" in code_lower and not any(m in approaches for m in ["BERT", "RoBERTa", "DeBERTa"]):
            approaches.append("Transformer")
        if "torch" in code_lower and not approaches:
            approaches.append("PyTorch-NN")
        if "sklearn" in code_lower and not approaches:
            approaches.append("sklearn")
        return ", ".join(approaches[:3]) if approaches else "custom"

    def _find_eval_submission(self, iter_dir: Path) -> Optional[Path]:
        """Find eval_submission.csv in the agent's working directory."""
        candidates = [
            iter_dir / "eval_submission.csv",
            iter_dir / "output" / "eval_submission.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Search recursively in iter_dir
        for f in iter_dir.rglob("eval_submission.csv"):
            return f
        return None

    def _elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0

    def _time_remaining(self) -> float:
        return self.time_limit - self._elapsed() if self.start_time else self.time_limit
