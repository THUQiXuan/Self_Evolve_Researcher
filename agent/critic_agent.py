#!/usr/bin/env python3
"""
SER Critic Agent - Monitors experiments across machines.
Writes alerts to /tmp/critic_agent/alerts.jsonl
Tracks per-task GPU utilization with rolling averages.

Usage:
  python3 critic_agent.py              # single check
  python3 critic_agent.py --loop 60    # loop every 60s with dashboard
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ALERT_FILE = "/tmp/critic_agent/alerts.jsonl"
STATE_FILE = "/tmp/critic_agent/state.json"
GPU_HISTORY_FILE = "/tmp/critic_agent/gpu_history.json"

MACHINES = {
    "22.4.243.44": ["new-york-city-taxi-fare-prediction"],                                          # CPU only (GPU occupied)
    "22.1.6.211":  ["champs-scalar-coupling", "cassava-leaf-disease-classification"],                # CPU + GPU
    "22.1.36.19":  ["ventilator-pressure-prediction", "jigsaw-unintended-bias-in-toxicity-classification"],  # CPU + GPU
    "22.1.1.106":  ["spooky-author-identification", "us-patent-phrase-to-phrase-matching"],           # CPU + GPU
}

# GPU assignment: each GPU competition runs 8 instances (1 GPU each)
NUM_GPU_INSTANCES = 8

PAPER_SCORES = {
    "us-patent-phrase-to-phrase-matching":              96.52,
    "tweet-sentiment-extraction":                      73.26,
    "cassava-leaf-disease-classification":              58.93,
    "ventilator-pressure-prediction":                  48.84,
    "jigsaw-unintended-bias-in-toxicity-classification": 8.14,
    "new-york-city-taxi-fare-prediction":               37.50,
    "aptos2019-blindness-detection":                   70.30,
    "champs-scalar-coupling":                          30.70,
    "spooky-author-identification":                    93.10,
}

LOWER_BETTER = {
    "ventilator-pressure-prediction",
    "champs-scalar-coupling",
}

SSH_OPTS = "-o ConnectTimeout=8 -o StrictHostKeyChecking=no -o BatchMode=yes"

os.makedirs("/tmp/critic_agent", exist_ok=True)


def now_utc():
    return datetime.now(timezone.utc)


def ssh_run(host, cmd):
    full_cmd = f"ssh {SSH_OPTS} root@{host} {repr(cmd)}"
    try:
        r = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=20)
        return r.stdout.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", -1
    except Exception:
        return "", -2


def write_alert(level, comp, host, msg, data=None):
    alert = {
        "ts": now_utc().isoformat(),
        "level": level,
        "comp": comp,
        "host": host,
        "msg": msg,
    }
    if data:
        alert["data"] = data
    with open(ALERT_FILE, "a") as f:
        f.write(json.dumps(alert) + "\n")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_gpu_output(gpu_out):
    """Parse nvidia-smi CSV output into list of dicts."""
    gpus = []
    if not gpu_out:
        return gpus
    for line in gpu_out.splitlines():
        parts = [p.strip().rstrip(" %MiB") for p in line.split(",")]
        if len(parts) >= 4:
            try:
                gpus.append({
                    "idx": int(parts[0]),
                    "util": int(parts[1]),
                    "mem_used": int(parts[2]),
                    "mem_total": int(parts[3]),
                })
            except ValueError:
                pass
    return gpus


def update_gpu_history(gpu_hist, host, comp_idx, comp, gpus, cpu_util=0):
    """Track GPU + CPU utilization samples for a task. Return current and avg stats."""
    key = f"{host}/{comp}"
    if key not in gpu_hist:
        gpu_hist[key] = {"samples": [], "comp": comp, "host": host}

    # All comps on a machine share the same 8 GPUs (CPU tasks don't use them)
    slot_gpus = [g for g in gpus if 0 <= g["idx"] < 8]

    sample = {
        "ts": now_utc().isoformat(),
        "avg_util": 0,
        "avg_mem_mb": 0,
        "max_mem_mb": 0,
        "cpu_util": cpu_util,
    }
    if slot_gpus:
        avg_util = sum(g["util"] for g in slot_gpus) / len(slot_gpus)
        avg_mem = sum(g["mem_used"] for g in slot_gpus) / len(slot_gpus)
        max_mem = max(g["mem_used"] for g in slot_gpus)
        sample["avg_util"] = round(avg_util, 1)
        sample["avg_mem_mb"] = round(avg_mem)
        sample["max_mem_mb"] = max_mem

    gpu_hist[key]["samples"].append(sample)
    # Keep last 500 samples (~8h at 60s interval)
    gpu_hist[key]["samples"] = gpu_hist[key]["samples"][-500:]

    return gpu_hist[key]


def get_gpu_stats(gpu_hist_entry):
    """Compute current and lifetime average GPU stats."""
    samples = gpu_hist_entry.get("samples", [])
    if not samples:
        return {"cur_util": 0, "cur_mem": 0, "avg_util": 0, "avg_mem": 0, "n_samples": 0}

    cur = samples[-1]
    all_utils = [s["avg_util"] for s in samples]
    all_mems = [s["avg_mem_mb"] for s in samples]

    return {
        "cur_util": cur["avg_util"],
        "cur_mem": cur["avg_mem_mb"],
        "avg_util": round(sum(all_utils) / len(all_utils), 1),
        "avg_mem": round(sum(all_mems) / len(all_mems)),
        "n_samples": len(samples),
    }


def get_training_time_stats(host, comp):
    """Parse log to compute training time ratio. Uses parse_training_time.py on remote.
    Tries instance 0 log first, falls back to single log."""
    # Try instance 0 log first (representative of parallel instances)
    log_inst = f"$WORKSPACE_DIR/logs/{comp}_inst0.log"
    log_single = f"$WORKSPACE_DIR/logs/{comp}.log"
    out, rc = ssh_run(host,
        f"python3 /root/ser/parse_training_time.py {log_inst} 2>/dev/null || "
        f"python3 /root/ser/parse_training_time.py {log_single} 2>/dev/null")
    try:
        return json.loads(out.strip()) if out.strip() else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def get_training_resource_stats(gpu_hist_entry):
    """Compute CPU and GPU stats only during active training phases.

    A sample is considered 'during training' if GPU util > 5% OR CPU util > 10%.
    This captures both GPU-heavy (DL) and CPU-heavy (LightGBM) tasks.
    """
    samples = gpu_hist_entry.get("samples", [])
    active = [s for s in samples
              if s.get("avg_util", 0) > 5 or s.get("cpu_util", 0) > 10]
    if not active:
        return {"train_gpu_util": 0, "train_gpu_mem": 0,
                "train_cpu_util": 0, "train_samples": 0,
                "total_samples": len(samples)}
    avg_gpu = sum(s.get("avg_util", 0) for s in active) / len(active)
    avg_mem = sum(s.get("avg_mem_mb", 0) for s in active) / len(active)
    avg_cpu = sum(s.get("cpu_util", 0) for s in active) / len(active)
    return {
        "train_gpu_util": round(avg_gpu, 1),
        "train_gpu_mem": round(avg_mem),
        "train_cpu_util": round(avg_cpu, 1),
        "train_samples": len(active),
        "total_samples": len(samples),
    }


def check_machine(host, comps, state, gpu_hist):
    """Check one machine. Returns per-comp info dict for dashboard."""
    info = {}

    # 1) Check processes alive — count instances per comp
    out, rc = ssh_run(host, "ps aux | grep run.py | grep -v grep")
    running_comps = set()
    comp_instance_count = {c: 0 for c in comps}
    if rc == 0 and out:
        for line in out.splitlines():
            for comp in comps:
                if comp in line:
                    running_comps.add(comp)
                    comp_instance_count[comp] += 1
    elif rc != 0:
        write_alert("CRITICAL", "ALL", host, "SSH connection failed")
        for comp in comps:
            info[comp] = {"status": "SSH_FAIL"}
        return info

    # 2) GPU utilization
    gpu_out, _ = ssh_run(host,
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null")
    gpus = parse_gpu_output(gpu_out)

    # 3) CPU utilization (overall machine)
    cpu_out, _ = ssh_run(host,
        "top -bn1 | head -3 | grep 'Cpu' | grep -oP '[\\d.]+(?= id)' 2>/dev/null || echo 100")
    try:
        cpu_idle = float(cpu_out.strip().split()[0]) if cpu_out.strip() else 100
        cpu_util = round(100 - cpu_idle, 1)
    except (ValueError, IndexError):
        cpu_util = 0

    # Per-competition checks
    for comp_idx, comp in enumerate(comps):
        prev = state.get(host, {}).get(comp, {})
        ci = {"comp": comp, "host": host, "cpu_util": cpu_util,
              "instances": comp_instance_count.get(comp, 0)}

        # GPU + CPU tracking
        hist_entry = update_gpu_history(gpu_hist, host, comp_idx, comp, gpus, cpu_util)
        gpu_stats = get_gpu_stats(hist_entry)
        ci["gpu"] = gpu_stats

        # Process status
        is_running = comp in running_comps
        ci["running"] = is_running
        if not is_running and prev.get("was_running", True):
            write_alert("INFO", comp, host,
                f"Process finished. Last score: {prev.get('best_score', 'N/A')} "
                f"(PR: {prev.get('best_pct', 'N/A')}%)")
        state.setdefault(host, {})[comp] = dict(prev, was_running=is_running)

        # Iteration count from logs (aggregate across instances)
        # Try multi-instance logs first, fall back to single log
        iter_out, _ = ssh_run(host,
            f"grep -ohP 'Iteration \\d+' $WORKSPACE_DIR/logs/{comp}_inst*.log $WORKSPACE_DIR/logs/{comp}.log 2>/dev/null | "
            f"grep -oP '\\d+' | sort -rn | head -1")
        total_iter_out, _ = ssh_run(host,
            f"grep -c 'Iteration' $WORKSPACE_DIR/logs/{comp}_inst*.log $WORKSPACE_DIR/logs/{comp}.log 2>/dev/null | "
            f"tail -1 | grep -oP '\\d+'")
        max_iter = iter_out.strip() if iter_out.strip() else "—"
        total_iters = total_iter_out.strip() if total_iter_out.strip() else ""
        ci["iter"] = f"{max_iter}" + (f"({total_iters})" if total_iters else "")

        # Error count (across all instance logs)
        err_out, _ = ssh_run(host,
            f"grep -ciE 'ERROR|Traceback|exception' $WORKSPACE_DIR/logs/{comp}_inst*.log $WORKSPACE_DIR/logs/{comp}.log 2>/dev/null | "
            f"awk -F: '{{s+=$NF}} END {{print s+0}}'")
        if not err_out.strip():
            err_out = "0"
        try:
            err_count = int(err_out.split()[0]) if err_out else 0
        except (ValueError, IndexError):
            err_count = 0
        prev_err = prev.get("error_count", 0)
        if err_count > prev_err + 5:
            write_alert("WARNING", comp, host, f"Error count jumped: {prev_err} -> {err_count}")
        state[host][comp]["error_count"] = err_count
        ci["errors"] = err_count

        # Timeout count (across all instance logs)
        to_out, _ = ssh_run(host,
            f"grep -ic 'timeout\\|timed out\\|time.out\\|exec_timeout' "
            f"$WORKSPACE_DIR/logs/{comp}_inst*.log $WORKSPACE_DIR/logs/{comp}.log 2>/dev/null | "
            f"awk -F: '{{s+=$NF}} END {{print s+0}}'")
        if not to_out.strip():
            to_out = "0"
        try:
            to_count = int(to_out.split()[0]) if to_out else 0
        except (ValueError, IndexError):
            to_count = 0
        prev_to = prev.get("timeout_count", 0)
        if to_count > prev_to + 3:
            write_alert("WARNING", comp, host, f"Repeated timeouts: {prev_to} -> {to_count}")
        state[host][comp]["timeout_count"] = to_count
        ci["timeouts"] = to_count

        # Score check
        best_pct = None
        best_score = None

        log_pct_out, _ = ssh_run(host,
            f"grep 'Percentile rank:' $WORKSPACE_DIR/logs/{comp}_inst*.log $WORKSPACE_DIR/logs/{comp}.log 2>/dev/null | "
            f"sed 's/.*Percentile rank: //' | sed 's/%.*//' | sort -rn | head -1")
        if log_pct_out.strip():
            try:
                best_pct = float(log_pct_out.strip())
            except ValueError:
                pass

        log_score_out, _ = ssh_run(host,
            f"grep 'search=' $WORKSPACE_DIR/logs/{comp}_inst*.log $WORKSPACE_DIR/logs/{comp}.log 2>/dev/null | "
            f"grep -oP 'search=\\K[0-9.]+' | sort -rn | head -1")
        if log_score_out.strip():
            try:
                best_score = float(log_score_out.strip())
            except ValueError:
                pass

        ci["best_pct"] = best_pct
        ci["best_score"] = best_score
        ci["paper_pct"] = PAPER_SCORES.get(comp, 50.0)
        ci["gap"] = (best_pct - ci["paper_pct"]) if best_pct is not None else None

        if best_pct is not None:
            state[host][comp]["best_score"] = best_score
            state[host][comp]["best_pct"] = best_pct

            prev_pct = prev.get("best_pct")
            if prev_pct is None or best_pct > prev_pct:
                paper_pct = ci["paper_pct"]
                gap = ci["gap"]
                level = "INFO"
                if best_pct < paper_pct - 20:
                    level = "CRITICAL"
                elif best_pct < paper_pct - 10:
                    level = "WARNING"
                write_alert(level, comp, host,
                    f"Score update: {best_pct:.1f}% (paper={paper_pct:.1f}%, gap={gap:+.1f}pp)",
                    {"score": best_score, "pct": best_pct})

            if prev_pct is not None and best_pct == prev_pct and is_running:
                stuck = prev.get("stuck_rounds", 0) + 1
                state[host][comp]["stuck_rounds"] = stuck
                if stuck >= 3:
                    write_alert("WARNING", comp, host,
                        f"Score STUCK at {best_pct:.1f}% for {stuck} cycles")
                ci["stuck"] = stuck
            else:
                state[host][comp]["stuck_rounds"] = 0
                ci["stuck"] = 0

            if best_pct < ci["paper_pct"] - 30 and not prev.get("far_below_alerted"):
                write_alert("CRITICAL", comp, host,
                    f"Score {best_pct:.1f}% FAR BELOW paper {ci['paper_pct']:.1f}%")
                state[host][comp]["far_below_alerted"] = True
        else:
            ci["stuck"] = 0

        # Invalid submissions (across all instance logs)
        inv_out, _ = ssh_run(host,
            f"grep -c 'Score is None\\|invalid submission' "
            f"$WORKSPACE_DIR/logs/{comp}_inst*.log $WORKSPACE_DIR/logs/{comp}.log 2>/dev/null | "
            f"awk -F: '{{s+=$NF}} END {{print s+0}}'")
        if not inv_out.strip():
            inv_out = "0"
        try:
            inv_count = int(inv_out.split()[0]) if inv_out else 0
        except (ValueError, IndexError):
            inv_count = 0
        prev_inv = prev.get("invalid_count", 0)
        if inv_count > prev_inv:
            write_alert("WARNING", comp, host, f"Invalid submission ({inv_count} total)")
        state[host][comp]["invalid_count"] = inv_count

        # Log stall (sum all instance log sizes)
        log_size_out, _ = ssh_run(host,
            f"wc -c $WORKSPACE_DIR/logs/{comp}_inst*.log $WORKSPACE_DIR/logs/{comp}.log 2>/dev/null | "
            f"tail -1 | awk '{{print $1}}'")
        try:
            log_size = int(log_size_out.strip()) if log_size_out.strip() else 0
        except ValueError:
            log_size = 0
        prev_log_size = prev.get("log_size", 0)
        if prev_log_size > 0 and log_size == prev_log_size and is_running:
            stall = prev.get("stall_rounds", 0) + 1
            state[host][comp]["stall_rounds"] = stall
            if stall >= 2:
                write_alert("WARNING", comp, host, f"Log stalled for {stall} cycles")
        else:
            state[host][comp]["stall_rounds"] = 0
        state[host][comp]["log_size"] = log_size

        # Training time stats
        train_stats = get_training_time_stats(host, comp)
        ci["train"] = train_stats

        # Resource stats during active training only
        train_res = get_training_resource_stats(hist_entry)
        ci["train_res"] = train_res

        # GPU idle alert (per task)
        if is_running and gpu_stats["cur_util"] < 3 and gpu_stats["cur_mem"] < 1000:
            ci["gpu_status"] = "IDLE"
        elif is_running and gpu_stats["cur_util"] < 20:
            ci["gpu_status"] = "LOW"
        elif is_running:
            ci["gpu_status"] = "OK"
        else:
            ci["gpu_status"] = "OFF"

        info[comp] = ci

    return info


def render_dashboard(all_info):
    """Render a colorful terminal dashboard."""
    ts = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")

    # ANSI colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    WHITE = "\033[97m"

    def color_pct(pct, paper):
        if pct is None:
            return f"{DIM}  —  {RESET}"
        gap = pct - paper
        if gap >= -2:
            return f"{GREEN}{pct:5.1f}%{RESET}"
        elif gap >= -15:
            return f"{YELLOW}{pct:5.1f}%{RESET}"
        else:
            return f"{RED}{pct:5.1f}%{RESET}"

    def color_gpu(util, status):
        if status == "OFF":
            return f"{DIM}  OFF {RESET}"
        elif status == "IDLE":
            return f"{RED}{util:4.0f}%{RESET}"
        elif status == "LOW":
            return f"{YELLOW}{util:4.0f}%{RESET}"
        else:
            return f"{GREEN}{util:4.0f}%{RESET}"

    def color_status(running, stuck):
        if not running:
            return f"{CYAN}DONE{RESET}"
        elif stuck >= 3:
            return f"{RED}STUCK{RESET}"
        else:
            return f"{GREEN} RUN{RESET}"

    # Clear screen
    print("\033[2J\033[H", end="")

    print(f"{BOLD}{WHITE}{'='*100}{RESET}")
    print(f"{BOLD}{WHITE}  SER Critic Dashboard — {ts}{RESET}")
    print(f"{BOLD}{WHITE}{'='*100}{RESET}")
    print()

    # Table header
    hdr = (f"  {'Competition':<40} {'St':>5} {'Inst':>4} {'Iter':>8} "
           f"{'PR%':>7} {'Paper':>7} {'Gap':>7} "
           f"{'TrCPU':>6} {'TrGPU':>6} {'TrMem':>7} "
           f"{'Train%':>7} {'Exec':>6} {'LLM':>6}")
    print(f"{BOLD}{hdr}{RESET}")
    print(f"  {'─'*120}")

    total_running = 0
    total_done = 0

    for host, comps in MACHINES.items():
        host_short = host.split(".")[-1]
        print(f"{DIM}  [{host}]{RESET}")

        for comp in comps:
            ci = all_info.get(comp, {})
            if ci.get("status") == "SSH_FAIL":
                print(f"    {comp:<46} {RED}SSH FAIL{RESET}")
                continue

            running = ci.get("running", False)
            stuck = ci.get("stuck", 0)
            iter_n = ci.get("iter", "—")
            best_pct = ci.get("best_pct")
            paper_pct = ci.get("paper_pct", 0)
            gap = ci.get("gap")
            gpu = ci.get("gpu", {})
            gpu_status = ci.get("gpu_status", "OFF")

            if running:
                total_running += 1
            elif best_pct is not None:
                total_done += 1

            # Shorten comp name
            short = comp[:38]

            status_str = color_status(running, stuck)
            pct_str = color_pct(best_pct, paper_pct)
            paper_str = f"{paper_pct:5.1f}%"
            gap_str = f"{gap:+5.1f}pp" if gap is not None else f"{DIM}  —  {RESET}"

            # Training-phase resource utilization
            tr = ci.get("train_res", {})
            has_train_data = tr.get("train_samples", 0) > 0

            # TrCPU (training-phase avg CPU)
            if not running:
                tr_cpu_str = f"{DIM}  OFF {RESET}"
            elif has_train_data:
                tc = tr["train_cpu_util"]
                tc_color = GREEN if tc >= 50 else (YELLOW if tc >= 10 else RED)
                tr_cpu_str = f"{tc_color}{tc:4.0f}%{RESET}"
            else:
                tr_cpu_str = f"{DIM}   — {RESET}"

            # TrGPU (training-phase avg GPU)
            if not running:
                tr_gpu_str = f"{DIM}  OFF {RESET}"
            elif has_train_data:
                tg = tr["train_gpu_util"]
                tg_color = GREEN if tg >= 50 else (YELLOW if tg >= 20 else RED)
                tr_gpu_str = f"{tg_color}{tg:4.0f}%{RESET}"
            else:
                tr_gpu_str = f"{DIM}   — {RESET}"

            # TrMem (training-phase avg GPU mem)
            if has_train_data and tr.get("train_gpu_mem", 0) > 0:
                gpu_mem = f"{tr['train_gpu_mem']:5.0f}M"
            else:
                gpu_mem = f"{DIM}    — {RESET}"

            # Training time breakdown
            ts_info = ci.get("train", {})
            if ts_info.get("total", 0) > 0:
                exec_pct = ts_info["pct"]
                exec_min = ts_info["exec"] / 60
                llm_min = ts_info["llm"] / 60
                pct_color = GREEN if exec_pct >= 60 else (YELLOW if exec_pct >= 30 else RED)
                train_pct_str = f"{pct_color}{exec_pct:5.1f}%{RESET}"
                exec_str = f"{exec_min:4.1f}m"
                llm_str = f"{llm_min:4.1f}m"
            else:
                train_pct_str = f"{DIM}   — {RESET}"
                exec_str = f"{DIM}  — {RESET}"
                llm_str = f"{DIM}  — {RESET}"

            inst_count = ci.get("instances", 0)
            inst_str = f"{inst_count}" if inst_count > 0 else f"{DIM}—{RESET}"

            print(f"    {short:<38} {status_str} {inst_str:>4} {iter_n:>8} "
                  f"{pct_str} {paper_str:>7} {gap_str:>7} "
                  f"{tr_cpu_str:>6} {tr_gpu_str:>6} {gpu_mem:>7} "
                  f"{train_pct_str:>7} {exec_str:>6} {llm_str:>6}")

    print(f"\n  {DIM}{'─'*120}{RESET}")
    print(f"  {BOLD}Running: {total_running}  Done: {total_done}  "
          f"TrCPU: {GREEN}>50%{RESET} {YELLOW}10-50%{RESET} {RED}<10%{RESET}  "
          f"TrGPU: {GREEN}>50%{RESET} {YELLOW}20-50%{RESET} {RED}<20%{RESET}  "
          f"Train%: {GREEN}>60%{RESET} {YELLOW}30-60%{RESET} {RED}<30%{RESET}")

    # Recent alerts (last 5)
    print(f"\n  {BOLD}Recent Alerts:{RESET}")
    try:
        with open(ALERT_FILE) as f:
            lines = f.readlines()
        recent = lines[-5:] if len(lines) >= 5 else lines
        for line in recent:
            a = json.loads(line)
            ts_short = a["ts"][11:19]
            lvl = a["level"]
            if lvl == "CRITICAL":
                lvl_c = f"{RED}{lvl}{RESET}"
            elif lvl == "WARNING":
                lvl_c = f"{YELLOW}{lvl}{RESET}"
            else:
                lvl_c = f"{DIM}{lvl}{RESET}"
            comp_short = a["comp"][:30]
            msg_short = a["msg"][:60]
            print(f"    {ts_short} [{lvl_c:>18}] {comp_short}: {msg_short}")
    except Exception:
        print(f"    {DIM}(no alerts){RESET}")

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=0,
                        help="Loop interval in seconds (0 = single run)")
    args = parser.parse_args()

    while True:
        state = load_json(STATE_FILE)
        gpu_hist = load_json(GPU_HISTORY_FILE)
        all_info = {}

        for host, comps in MACHINES.items():
            try:
                info = check_machine(host, comps, state, gpu_hist)
                all_info.update(info)
            except Exception as e:
                write_alert("CRITICAL", "ALL", host, f"Critic exception: {e}")

        save_json(STATE_FILE, state)
        save_json(GPU_HISTORY_FILE, gpu_hist)

        if args.loop > 0:
            render_dashboard(all_info)
        else:
            # Single run: print simple summary
            render_dashboard(all_info)

        if args.loop <= 0:
            break

        time.sleep(args.loop)


if __name__ == "__main__":
    main()
