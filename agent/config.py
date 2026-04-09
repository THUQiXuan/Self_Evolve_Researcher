"""SER — Global Configuration (Gemini 3 Pro + this server)."""

import os
from pathlib import Path

# === Paths ===
LOONGFLOW_DIR = Path("/newcpfs/user/qixuan1/310/LoongFlow")
MLEBENCH_DATA_DIR = LOONGFLOW_DIR / "output" / "mlebench"

WORK_DIR = Path(os.environ.get(
    "SER_WORK_DIR", "/newcpfs/user/qixuan1/SER/workspace"
))

# Python binary — use the loongflow_ml conda env which has mle-bench installed
PYTHON_BIN = Path(os.environ.get(
    "SER_PYTHON_BIN",
    "/root/.conda/envs/loongflow_ml/bin/python"
))

# eval_program.py from LoongFlow for OOF scoring
EVAL_PROGRAM_PATH = LOONGFLOW_DIR / "agents" / "ml_agent" / "examples" / "mlebench" / "eval_program.py"

# === LLM (Gemini 3 Pro via local proxy) ===
LLM_PROXY_URL = os.environ.get("SER_LLM_URL", "http://127.0.0.1:8010")
LLM_API_KEY = os.environ.get("SER_LLM_KEY", "YOUR_API_KEY")
LLM_MODEL = os.environ.get("SER_LLM_MODEL", "gemini3_pro")
LLM_MAX_TOKENS = 65535
LLM_TEMPERATURE = 1.0   # Gemini 3 Pro works best at temperature=1

# === Evolution Hyperparameters (from paper) ===
SELECTION_TEMPERATURE = 0.2   # T=0.2 near-greedy rank selection
CROSSOVER_PROB = 0.15         # pc=15%

# === ReAct Agent ===
MAX_REACT_STEPS = 15
CODE_EXEC_TIMEOUT = 1200      # 20 min per code execution
OBSERVATION_MAX_CHARS = 30000

# === HCE Data Split ===
TRAIN_RATIO = 0.8
SEARCH_RATIO = 0.1
VAL_RATIO = 0.1

# === Time Limits ===
DEFAULT_TIME_LIMIT = 3 * 3600   # 3h (paper baseline)
