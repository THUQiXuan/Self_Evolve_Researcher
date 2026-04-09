"""SER — EDA (Exploratory Data Analysis) Agent.

Runs a short ReAct loop focused on data understanding, not model building.
Results are cached to disk so parallel instances can reuse them.
"""

import logging
import re
from pathlib import Path

from llm import LLMClient
from react_agent import ReActAgent
from prompts.eda import build_eda_prompt
from prompts.system import build_eda_system_prompt

logger = logging.getLogger(__name__)

EDA_MAX_STEPS = 6          # EDA doesn't need many steps
EDA_EXEC_TIMEOUT = 300     # 5 min per code execution (no training)
DATA_PROFILE_FILENAME = "data_profile.txt"


def run_eda(
    llm: LLMClient,
    work_dir: Path,
    data_dir: Path,
    description: str,
    sample_submission_preview: str,
    gpu_id: int = None,
) -> str:
    """Run EDA agent and return data profile text.

    Results cached to {work_dir}/data_profile.txt -- skips if already exists.
    """
    profile_path = work_dir / DATA_PROFILE_FILENAME

    # Cache: if profile already exists, reuse it
    if profile_path.exists():
        profile = profile_path.read_text()
        if len(profile) > 100:  # sanity check
            logger.info(f"EDA profile already cached at {profile_path}")
            return profile

    logger.info("Running EDA agent...")
    eda_dir = work_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Symlink data files into EDA working directory
    for item in data_dir.iterdir():
        link = eda_dir / item.name
        if not link.exists():
            try:
                link.symlink_to(item)
            except OSError:
                pass  # skip if symlink fails (e.g., already exists as dir)

    system_prompt = build_eda_system_prompt()
    task_prompt = build_eda_prompt(
        data_dir=str(eda_dir),
        description=description,
        sample_submission_preview=sample_submission_preview,
        output_dir=str(eda_dir),
    )

    agent = ReActAgent(
        llm=llm,
        work_dir=eda_dir,
        max_steps=EDA_MAX_STEPS,
        exec_timeout=EDA_EXEC_TIMEOUT,
        gpu_id=None,  # EDA doesn't need GPU
    )

    result = agent.run(system_prompt, task_prompt)

    # Extract data profile from trajectory observations
    profile = _extract_profile(result["trajectory"])

    # Save to disk
    profile_path.write_text(profile)
    logger.info(f"EDA complete, profile saved to {profile_path} ({len(profile)} chars)")

    return profile


def _extract_profile(trajectory: list) -> str:
    """Extract structured data profile from EDA trajectory.

    Concatenates all observations (agent's analysis output).
    Looks for the === DATA PROFILE === section if present.
    """
    all_observations = []
    for reason, action, obs in trajectory:
        if obs and obs != "No action parsed":
            all_observations.append(obs)

    full_text = "\n".join(all_observations)

    # Try to extract structured profile section
    match = re.search(
        r'=== DATA PROFILE ===(.*?)=== END PROFILE ===',
        full_text, re.DOTALL,
    )
    if match:
        structured = match.group(0)
        # Also include the full analysis for context
        return f"{structured}\n\n## Full Analysis Output\n{full_text[-5000:]}"

    # Fallback: return last 5000 chars of all observations
    return f"## EDA Analysis Output\n{full_text[-5000:]}"
