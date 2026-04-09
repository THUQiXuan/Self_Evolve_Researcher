# SER — Self-Evolve Researcher

## How it works

SER runs an evolutionary search over candidate ML solutions for Kaggle competitions:

1. **EDA Agent** — explores the dataset and produces a structured data profile
2. **ReAct Worker** — a Reason-Act-Observe loop that writes and executes code (bash / Python), up to 15 steps per iteration
3. **Orchestrator** — steady-state evolutionary loop: temperature-scaled rank selection (T=0.2), mutation + crossover (p=15%), parallel instances sharing a program database via file locks
4. **HCE Split** — training data is split 80/10/10 into D\_train / D\_search / D\_val so the fitness signal stays honest throughout the search

All scoring is done directly via [mle-bench](https://github.com/openai/mle-bench) — no separate grading service needed.

## Prerequisites

- Conda environment `loongflow_ml` with `mlebench` and standard ML libraries installed
- Local Gemini proxy running at `http://127.0.0.1:8010` (see below)
- Kaggle API credentials configured (`~/.kaggle/kaggle.json`) for data download

## Quickstart

### 1. Start the Gemini proxy

```bash
source /path/to/miniforge3/use_shared_loongflow_ml.sh
cd /path/to/LoongFlow

export UPSTREAM_API_KEY="YOUR_API_KEY"
export UPSTREAM_URL="https://runway.devops.rednote.life/openai/google/v1:generateContent"
export ALLOWED_MODELS="gemini/gemini-3-flash-preview,gemini-3-flash-preview,gemini3_pro,gemini/gemini3_pro"
export REQUEST_TIMEOUT=300

nohup python -m uvicorn local_proxy.gemini_gateway_proxy:app \
  --host 127.0.0.1 --port 8010 \
  > gemini_proxy.log 2>&1 &

curl http://127.0.0.1:8010/healthz   # should return {"status":"ok"}
```

### 2. Set your API key

```bash
export SER_LLM_KEY="YOUR_API_KEY"
```

### 3. Run a competition

```bash
# Prepare data (downloads from Kaggle — skipped automatically if already done)
bash run_ser.sh prepare stanford-covid-vaccine

# Run the agent (3-hour limit, 4 parallel instances)
bash run_ser.sh run stanford-covid-vaccine --background

# Monitor progress
bash run_ser.sh status stanford-covid-vaccine

# View final score
bash run_ser.sh grade stanford-covid-vaccine
```

The `run` command calls `prepare` automatically if the data is not yet downloaded.

## CLI reference

```
bash run_ser.sh <command> <competition-id> [options]

Commands:
  prepare <id>                    Download and prepare competition data
  run     <id> [options]          Run SER agent
    --background                  Run in background (default: foreground)
    --instances N                 Number of parallel instances (default: 4)
    --time-limit SECS             Wall-clock budget per run (default: 10800 = 3h)
  stop    <id>                    Send SIGTERM to all running instances
  status  <id>                    Show live status, PID list, and best score so far
  grade   <id>                    Print final score and percentile rank
```

## Configuration

Key settings in `agent/config.py` (all overridable via environment variables):

| Variable | Default | Description |
|---|---|---|
| `SER_LLM_URL` | `http://127.0.0.1:8010` | Gemini proxy address |
| `SER_LLM_KEY` | *(required)* | Proxy API key |
| `SER_LLM_MODEL` | `gemini3_pro` | Model name |
| `SER_WORK_DIR` | `./workspace` | Output directory |
| `SER_PYTHON_BIN` | loongflow_ml Python | Python interpreter |

## Project structure

```
run_ser.sh          # Main entry point
agent/
  run.py            # CLI entry point for a single instance
  orchestrator.py   # Evolutionary loop + HCE split + grading
  react_agent.py    # ReAct worker (bash/python/submit)
  evolution.py      # Rank-based selection, crossover
  database.py       # Population store (FileLock-safe)
  competition.py    # mle-bench integration, HCE data split
  llm.py            # Gemini 3 Pro client
  config.py         # Global configuration
  eda_agent.py      # EDA sub-agent
  critic.py         # Async critic for code feedback
  prompts/          # System, mutation, crossover, EDA prompts
```
