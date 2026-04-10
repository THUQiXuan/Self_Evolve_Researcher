#!/usr/bin/env bash
# SER Benchmark Runner — runs all 29 MLE-Bench competitions sequentially.
#
# Usage:
#   bash run_benchmark.sh [options]
#
# Options:
#   --gpus   GPU_LIST    Comma-separated GPU IDs to use (default: 2,3,6,7)
#   --instances N        Parallel instances per competition (default: 4)
#   --time-limit SECS    Time per competition in seconds (default: 10800 = 3h)
#   --start-from COMP    Skip competitions before this one (resume support)
#   --dry-run            Print competition order and exit
#   --clean-workspace    Delete each competition's workspace before starting it
#
# Environment:
#   SER_LLM_KEY               Proxy API key (required)
#   SER_LANGFUSE_PUBLIC_KEY   Langfuse public key (optional)
#   SER_LANGFUSE_SECRET_KEY   Langfuse secret key (optional)
#   SER_LANGFUSE_HOST         Langfuse host (optional)
#
# Example:
#   export SER_LLM_KEY="your_key"
#   export SER_LANGFUSE_PUBLIC_KEY="pk-lf-..."
#   export SER_LANGFUSE_SECRET_KEY="sk-lf-..."
#   export SER_LANGFUSE_HOST="https://us.cloud.langfuse.com"
#   bash run_benchmark.sh --gpus 2,3,6,7 --instances 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$SCRIPT_DIR/workspace"
LOG_DIR="$SCRIPT_DIR/benchmark_logs"
SUMMARY_FILE="$LOG_DIR/benchmark_summary.tsv"

# ── Defaults ────────────────────────────────────────────────────────────────
GPUS="2,3,6,7"
INSTANCES=4
TIME_LIMIT=10800
START_FROM=""
DRY_RUN=false
CLEAN_WORKSPACE=false

# ── 29 competitions (MLE-Bench 30 minus stanford-covid-vaccine) ─────────────
# Ordered roughly easy→hard to build early confidence;
# adjust order freely — each is independent.
COMPETITIONS=(
    # ── tabular / regression ──────────────────────────────────────────────
    "ventilator-pressure-prediction"
    "nomad2018-predict-transparent-conductors"
    "osic-pulmonary-fibrosis-progression"
    "new-york-city-taxi-fare-prediction"
    "petfinder-pawpularity-score"
    "champs-scalar-coupling"
    # ── NLP ───────────────────────────────────────────────────────────────
    "spooky-author-identification"
    "tweet-sentiment-extraction"
    "jigsaw-unintended-bias-in-toxicity-classification"
    "us-patent-phrase-to-phrase-matching"
    "billion-word-imputation"
    "tensorflow2-question-answering"
    # ── vision / classification ───────────────────────────────────────────
    "aptos2019-blindness-detection"
    "cassava-leaf-disease-classification"
    "histopathologic-cancer-detection"
    "plant-pathology-2021-fgvc8"
    "hubmap-kidney-segmentation"
    "hms-harmful-brain-activity-classification"
    "kuzushiji-recognition"
    # ── multi-modal / audio ───────────────────────────────────────────────
    "freesound-audio-tagging-2019"
    "mlsp-2013-birds"
    "multi-modal-gesture-recognition"
    # ── retrieval / ranking ───────────────────────────────────────────────
    "h-and-m-personalized-fashion-recommendations"
    "imet-2020-fgvc7"
    "hotel-id-2021-fgvc8"
    # ── time series / contact ─────────────────────────────────────────────
    "nfl-player-contact-detection"
    "smartphone-decimeter-2022"
    # ── molecular / chemistry ─────────────────────────────────────────────
    "bms-molecular-translation"
    "whale-categorization-playground"
)

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)            GPUS="$2"; shift 2 ;;
        --instances)       INSTANCES="$2"; shift 2 ;;
        --time-limit)      TIME_LIMIT="$2"; shift 2 ;;
        --start-from)      START_FROM="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --clean-workspace) CLEAN_WORKSPACE=true; shift ;;
        -h|--help)
            head -20 "$0"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Dry-run: just print order ─────────────────────────────────────────────
if [[ "$DRY_RUN" == true ]]; then
    echo "Competition order (${#COMPETITIONS[@]} total, ~$(( ${#COMPETITIONS[@]} * TIME_LIMIT / 3600 ))h total):"
    for i in "${!COMPETITIONS[@]}"; do
        printf "  %2d. %s\n" $((i+1)) "${COMPETITIONS[$i]}"
    done
    exit 0
fi

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/benchmark.log"; }
warn() { log "WARNING: $*"; }

# Summary TSV header (create only if file doesn't exist)
if [[ ! -f "$SUMMARY_FILE" ]]; then
    echo -e "competition\tbest_score\tpercentile_rank\ttotal_iterations\ttotal_elapsed_s\tstatus" \
        > "$SUMMARY_FILE"
fi

log "=== SER Benchmark Start ==="
log "Competitions: ${#COMPETITIONS[@]}"
log "GPUs: $GPUS | Instances: $INSTANCES | Time/comp: ${TIME_LIMIT}s"
log "Summary: $SUMMARY_FILE"
log "Langfuse: ${SER_LANGFUSE_HOST:-disabled}"

# ── Main loop ────────────────────────────────────────────────────────────────
skip=false
if [[ -n "$START_FROM" ]]; then
    skip=true
    log "Skipping competitions before '$START_FROM'"
fi

total=${#COMPETITIONS[@]}
for idx in "${!COMPETITIONS[@]}"; do
    comp="${COMPETITIONS[$idx]}"
    num=$((idx + 1))

    # Handle --start-from
    if [[ "$skip" == true ]]; then
        if [[ "$comp" == "$START_FROM" ]]; then
            skip=false
        else
            log "Skipping [$num/$total] $comp"
            continue
        fi
    fi

    log ""
    log "════════════════════════════════════════════════════════════"
    log "[$num/$total] Starting: $comp"
    log "════════════════════════════════════════════════════════════"

    # Clean workspace if requested (fresh start, no inherited solutions)
    if [[ "$CLEAN_WORKSPACE" == true ]]; then
        if [[ -d "$WORKSPACE_DIR/$comp" ]]; then
            log "Cleaning workspace for $comp..."
            rm -rf "$WORKSPACE_DIR/$comp"
        fi
    fi

    comp_log="$LOG_DIR/${comp}.log"
    comp_start=$(date +%s)

    # Run — foreground, waiting for completion
    set +e
    bash "$SCRIPT_DIR/run_ser.sh" run "$comp" \
        --instances "$INSTANCES" \
        --time-limit "$TIME_LIMIT" \
        --gpus "$GPUS" \
        2>&1 | tee "$comp_log"
    run_exit=$?
    set -e

    comp_end=$(date +%s)
    comp_elapsed=$(( comp_end - comp_start ))

    # Parse result
    result_file="$WORKSPACE_DIR/$comp/result.json"
    if [[ -f "$result_file" ]]; then
        read -r best_score pr iterations elapsed status <<< "$(
            python3 -c "
import json, sys
try:
    with open('$result_file') as f:
        r = json.load(f)
    print(r.get('best_full_score','N/A'),
          r.get('percentile_rank','N/A'),
          r.get('total_iterations',0),
          round(r.get('total_elapsed',0)),
          'ok')
except Exception as e:
    print('N/A','N/A','N/A','N/A','error:'+str(e)[:40))
" 2>/dev/null
        )"
    else
        best_score="N/A"; pr="N/A"; iterations="N/A"; elapsed="$comp_elapsed"; status="no_result"
    fi

    # Append to summary
    echo -e "$comp\t$best_score\t$pr\t$iterations\t$elapsed\t$status" >> "$SUMMARY_FILE"

    log "[$num/$total] DONE: $comp | score=$best_score PR=$pr% iters=$iterations elapsed=${elapsed}s"
    log "Running summary:"
    column -t -s $'\t' "$SUMMARY_FILE" 2>/dev/null || cat "$SUMMARY_FILE"
    log ""
done

log ""
log "=== Benchmark Complete ==="
log "Results saved to: $SUMMARY_FILE"
log ""
column -t -s $'\t' "$SUMMARY_FILE" 2>/dev/null || cat "$SUMMARY_FILE"
