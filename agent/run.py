#!/usr/bin/env python3
"""SER — Entry point for running a single competition."""

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path

from filelock import FileLock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import WORK_DIR, DEFAULT_TIME_LIMIT, LLM_MODEL
from orchestrator import Orchestrator
from llm import LLMClient


def setup_logging(log_dir: Path, competition_id: str, instance_id: int = None):
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    if instance_id is not None:
        log_file = log_dir / f"{competition_id}_inst{instance_id}.log"
    else:
        log_file = log_dir / f"{competition_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="SER — Run evolutionary search on a Kaggle competition")
    parser.add_argument("competition", type=str, help="Competition ID (e.g., plant-pathology-2021-fgvc8)")
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT,
                        help=f"Time limit in seconds (default: {DEFAULT_TIME_LIMIT})")
    parser.add_argument("--work-dir", type=str, default=str(WORK_DIR),
                        help=f"Working directory (default: {WORK_DIR})")
    parser.add_argument("--model", type=str, default=LLM_MODEL,
                        help=f"LLM model to use (default: {LLM_MODEL})")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="GPU index for this instance (0-7). If not set, runs as CPU task.")
    parser.add_argument("--instance-id", type=int, default=None,
                        help="Instance ID for CPU parallel tasks (0-7). If not set, defaults to gpu-id or 0.")
    args = parser.parse_args()

    # Resolve instance_id: --instance-id > --gpu-id > None
    instance_id = args.instance_id if args.instance_id is not None else args.gpu_id

    # Set CUDA_VISIBLE_DEVICES early for this process
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    else:
        # CPU task — ensure no GPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    work_dir = Path(args.work_dir)
    setup_logging(work_dir / "logs", args.competition, instance_id)

    logger = logging.getLogger(__name__)
    logger.info(f"SER — Competition: {args.competition}")
    logger.info(f"Time limit: {args.time_limit}s, Model: {args.model}, "
                f"GPU: {args.gpu_id}, Instance: {instance_id}")

    # Create LLM client
    llm = LLMClient(model=args.model)

    # Run orchestrator
    orch = Orchestrator(
        competition_id=args.competition,
        work_dir=work_dir,
        time_limit=args.time_limit,
        llm=llm,
        gpu_id=args.gpu_id,
        instance_id=instance_id,
    )

    # Register SIGTERM handler for graceful shutdown
    def sigterm_handler(signum, frame):
        logger.info("Received SIGTERM, stopping gracefully...")
        orch.stop_requested = True
    signal.signal(signal.SIGTERM, sigterm_handler)

    result = orch.run()

    # Save result — atomic best-wins with file lock
    result_path = work_dir / args.competition / "result.json"
    lock_path = work_dir / args.competition / ".result.lock"

    with FileLock(str(lock_path), timeout=30):
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
                        is_lower = orch.comp.is_lower_better
                        should_save = (new_score <= existing_score) if is_lower else (new_score >= existing_score)
                    else:
                        # New has a score, existing doesn't → save
                        should_save = new_score is not None
            except (json.JSONDecodeError, IOError):
                should_save = True

        if should_save:
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Result saved to {result_path} (score={result.get('best_full_score')}, PR={result.get('percentile_rank')})")
        else:
            logger.info(f"Result NOT saved — existing is better")

    return result


if __name__ == "__main__":
    main()
