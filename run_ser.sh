#!/usr/bin/env bash
# SER — Run script for MLE-Bench competitions (no Docker)
# Usage:
#   bash run_ser.sh prepare <competition-id>
#   bash run_ser.sh run <competition-id> [--background] [--instances N] [--time-limit SECS]
#   bash run_ser.sh stop <competition-id>
#   bash run_ser.sh status <competition-id>
#   bash run_ser.sh grade <competition-id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$SCRIPT_DIR/agent"
WORKSPACE_DIR="$SCRIPT_DIR/workspace"
LOONGFLOW_DIR="/newcpfs/user/qixuan1/310/LoongFlow"
MLEBENCH_DATA_DIR="$LOONGFLOW_DIR/output/mlebench"
PYTHON="/root/.conda/envs/loongflow_ml/bin/python"
MLEBENCH_PYTHON="$PYTHON"

# LLM proxy settings
export SER_LLM_URL="http://127.0.0.1:8010"
export SER_LLM_KEY="YOUR_API_KEY"
export SER_LLM_MODEL="gemini3_pro"
export SER_WORK_DIR="$WORKSPACE_DIR"
export SER_PYTHON_BIN="$PYTHON"
export PYTHONPATH="$LOONGFLOW_DIR/mle-bench:$AGENT_DIR:${PYTHONPATH:-}"

# ------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

ensure_proxy_running() {
    if ! curl -s "$SER_LLM_URL/healthz" 2>/dev/null | grep -q "ok\|healthy\|status"; then
        log "Starting Gemini proxy..."
        source "$LOONGFLOW_DIR/../../310/miniforge3/use_shared_loongflow_ml.sh" 2>/dev/null || true
        cd "$LOONGFLOW_DIR"
        export UPSTREAM_API_KEY="YOUR_API_KEY"
        export UPSTREAM_URL="https://runway.devops.rednote.life/openai/google/v1:generateContent"
        export ALLOWED_MODELS="gemini/gemini-3-flash-preview,gemini-3-flash-preview,gemini3_pro,gemini/gemini3_pro"
        export REQUEST_TIMEOUT=300
        nohup "$PYTHON" -m uvicorn local_proxy.gemini_gateway_proxy:app \
            --host 127.0.0.1 --port 8010 \
            > "$LOONGFLOW_DIR/local_proxy/gemini_proxy_8010.log" 2>&1 &
        sleep 5
        curl -s "$SER_LLM_URL/healthz" || log "WARNING: proxy may not be ready"
    else
        log "Gemini proxy already running at $SER_LLM_URL"
    fi
}

do_prepare() {
    local competition="$1"
    local data_dir="$MLEBENCH_DATA_DIR/$competition/prepared/public"

    if [[ -d "$data_dir" ]]; then
        log "Competition '$competition' already prepared at $data_dir"
        return 0
    fi

    log "Preparing competition: $competition"
    log "Data will be downloaded to: $MLEBENCH_DATA_DIR/$competition"

    # Use the mlebench binary from the loongflow_ml conda env
    MLEBENCH_BIN="$(dirname "$PYTHON")/mlebench"
    PYTHONPATH="$LOONGFLOW_DIR/mle-bench:${PYTHONPATH:-}" \
        "$MLEBENCH_BIN" prepare \
            --competition-id "$competition" \
            --data-dir "$MLEBENCH_DATA_DIR"

    if [[ -d "$data_dir" ]]; then
        log "Preparation complete for '$competition'."
    else
        log "ERROR: prepare finished but data not found at $data_dir"
        exit 1
    fi
}

do_run() {
    local competition="$1"
    shift
    local background=false
    local num_instances=4
    local time_limit=10800  # 3 hours

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --background) background=true; shift ;;
            --instances) num_instances="$2"; shift 2 ;;
            --time-limit) time_limit="$2"; shift 2 ;;
            *) log "Unknown option: $1"; shift ;;
        esac
    done

    ensure_proxy_running

    local data_dir="$MLEBENCH_DATA_DIR/$competition/prepared/public"
    if [[ ! -d "$data_dir" ]]; then
        log "Competition data not found at $data_dir — running prepare first..."
        do_prepare "$competition"
    fi

    mkdir -p "$WORKSPACE_DIR/$competition"

    # Free GPUs: GPU 0,1 are mostly free (0,1 have 3306MB used each)
    # Assign one GPU per instance
    local available_gpus=(0 1 4 5)  # GPUs with most free memory
    local pid_file="$WORKSPACE_DIR/$competition/.pids"
    > "$pid_file"

    log "Starting $num_instances parallel instances for $competition (time_limit=${time_limit}s)..."

    for ((i=0; i<num_instances; i++)); do
        local gpu_idx=${available_gpus[$((i % ${#available_gpus[@]}))]}
        local log_file="$WORKSPACE_DIR/$competition/inst_${i}.log"

        log "  Instance $i on GPU $gpu_idx -> $log_file"

        CUDA_VISIBLE_DEVICES="$gpu_idx" \
        SER_WORK_DIR="$WORKSPACE_DIR" \
        nohup "$PYTHON" "$AGENT_DIR/run.py" \
            "$competition" \
            --time-limit "$time_limit" \
            --work-dir "$WORKSPACE_DIR" \
            --gpu-id "$gpu_idx" \
            --instance-id "$i" \
            > "$log_file" 2>&1 &

        local pid=$!
        echo "$pid" >> "$pid_file"
        log "  Instance $i PID=$pid"
    done

    log "All $num_instances instances launched. PIDs saved to $pid_file"
    log "Monitor: tail -f $WORKSPACE_DIR/$competition/inst_0.log"

    if [[ "$background" == false ]]; then
        log "Waiting for all instances to complete..."
        while IFS= read -r pid; do
            wait "$pid" 2>/dev/null || true
        done < "$pid_file"
        log "All instances complete. Run: bash run_ser.sh grade $competition"
    fi
}

do_stop() {
    local competition="$1"
    local pid_file="$WORKSPACE_DIR/$competition/.pids"

    if [[ ! -f "$pid_file" ]]; then
        log "No PID file found for $competition"
        return
    fi

    log "Stopping all instances of $competition..."
    while IFS= read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -SIGTERM "$pid" 2>/dev/null || true
            log "  Sent SIGTERM to PID $pid"
        fi
    done < "$pid_file"

    sleep 10

    # Force kill any survivors
    while IFS= read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -SIGKILL "$pid" 2>/dev/null || true
            log "  Force killed PID $pid"
        fi
    done < "$pid_file"

    rm -f "$pid_file"
    log "All instances stopped."
}

do_status() {
    local competition="$1"
    local pid_file="$WORKSPACE_DIR/$competition/.pids"
    local result_file="$WORKSPACE_DIR/$competition/result.json"

    log "=== Status: $competition ==="

    if [[ -f "$pid_file" ]]; then
        log "Running PIDs:"
        while IFS= read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  PID $pid: RUNNING"
            else
                echo "  PID $pid: STOPPED"
            fi
        done < "$pid_file"
    else
        log "No PID file (not running or not started)"
    fi

    if [[ -f "$result_file" ]]; then
        log "Current best result:"
        "$PYTHON" -c "
import json
with open('$result_file') as f:
    r = json.load(f)
print(f'  Best score: {r.get(\"best_full_score\")}')
print(f'  Percentile rank: {r.get(\"percentile_rank\", \"N/A\")}')
print(f'  Iterations: {r.get(\"total_iterations\", 0)}')
print(f'  Elapsed: {r.get(\"total_elapsed\", 0):.0f}s')
"
    fi

    # Show last few lines from first instance log
    local log0="$WORKSPACE_DIR/$competition/inst_0.log"
    if [[ -f "$log0" ]]; then
        log "Last 10 lines from inst_0.log:"
        tail -10 "$log0"
    fi
}

do_grade() {
    local competition="$1"
    local result_file="$WORKSPACE_DIR/$competition/result.json"

    if [[ ! -f "$result_file" ]]; then
        log "No result.json found for $competition"
        log "Has the agent run yet? Try: bash run_ser.sh status $competition"
        exit 1
    fi

    log "=== Grading $competition ==="
    "$PYTHON" -c "
import json, sys
sys.path.insert(0, '$AGENT_DIR')
sys.path.insert(0, '$LOONGFLOW_DIR/mle-bench')
from pathlib import Path
from competition import CompetitionManager

with open('$result_file') as f:
    result = json.load(f)

comp_id = result.get('competition_id', '$competition')
comp = CompetitionManager(comp_id, Path('$WORKSPACE_DIR/$competition'))

score = result.get('best_full_score')
if score is None:
    print('No score available yet')
else:
    pr = comp.compute_percentile_rank(score)
    print(f'Competition: {comp_id}')
    print(f'Raw score: {score}')
    print(f'Percentile rank: {pr:.1f}%')
    print(f'Is lower better: {comp.is_lower_better}')

# Also check via mle-bench grader if submission file exists
programs_dir = Path('$WORKSPACE_DIR/$competition/programs')
submissions = list(programs_dir.glob('submission_*.csv')) if programs_dir.exists() else []
if submissions:
    best_sub = submissions[-1]  # last one
    print(f'Latest submission: {best_sub.name}')
    # Grade it
    try:
        s = comp.grade_submission(best_sub)
        print(f'Grade result: {s}')
        print(f'Percentile: {comp.compute_percentile_rank(s):.1f}%')
    except Exception as e:
        print(f'Grading error: {e}')
" 2>&1
}

# ------------------------------------------------------------------
# Main dispatch
cmd="${1:-help}"
shift || true

case "$cmd" in
    prepare) do_prepare "${1:?Usage: $0 prepare <competition-id>}" ;;
    run)     do_run "${1:?Usage: $0 run <competition-id> [--background] [--instances N] [--time-limit SECS]}" "${@:2}" ;;
    stop)    do_stop "${1:?Usage: $0 stop <competition-id>}" ;;
    status)  do_status "${1:?Usage: $0 status <competition-id>}" ;;
    grade)   do_grade "${1:?Usage: $0 grade <competition-id>}" ;;
    *)
        echo "SER — Self-Evolve Researcher"
        echo "Usage: $0 {prepare|run|stop|status|grade} <competition-id> [options]"
        echo ""
        echo "Commands:"
        echo "  prepare <id>                    Download and prepare competition data"
        echo "  run <id> [--background]         Run SER agent (4 parallel instances)"
        echo "      [--instances N]             Number of parallel instances (default: 4)"
        echo "      [--time-limit SECS]         Time limit per instance (default: 10800 = 3h)"
        echo "  stop <id>                       Stop running instances"
        echo "  status <id>                     Show current status and best score"
        echo "  grade <id>                      Show final grade and percentile rank"
        echo ""
        echo "Example:"
        echo "  bash run_ser.sh run stanford-covid-vaccine --background"
        ;;
esac
