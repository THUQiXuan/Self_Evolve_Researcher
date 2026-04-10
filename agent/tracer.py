"""SER — Langfuse tracing singleton.

All other modules import `get_langfuse()` and `get_trace_context()` from here.
Set environment variables before running:
    SER_LANGFUSE_PUBLIC_KEY
    SER_LANGFUSE_SECRET_KEY
    SER_LANGFUSE_HOST   (default: https://cloud.langfuse.com)
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_langfuse = None
_enabled = False


def init_langfuse() -> bool:
    """Initialize Langfuse client. Returns True if successfully enabled."""
    global _langfuse, _enabled

    public_key = os.environ.get("SER_LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("SER_LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("SER_LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.info("Langfuse disabled (SER_LANGFUSE_PUBLIC_KEY / SER_LANGFUSE_SECRET_KEY not set)")
        _enabled = False
        return False

    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        _enabled = True
        logger.info(f"Langfuse enabled (host={host})")
        return True
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e} — tracing disabled")
        _enabled = False
        return False


def get_langfuse():
    """Return Langfuse client, or None if disabled."""
    return _langfuse if _enabled else None


def is_enabled() -> bool:
    return _enabled


class RunTrace:
    """One trace per competition run (shared across all instances via trace_id)."""

    def __init__(self, competition_id: str, instance_id: int, time_limit: int):
        self.competition_id = competition_id
        self.instance_id = instance_id
        self._trace = None
        self._current_span = None

        lf = get_langfuse()
        if lf is None:
            return

        try:
            self._trace = lf.trace(
                name=f"ser-run",
                input={"competition": competition_id, "instance": instance_id,
                       "time_limit": time_limit},
                tags=[competition_id, f"inst-{instance_id}"],
                metadata={"competition": competition_id, "instance_id": instance_id},
            )
            logger.info(f"Langfuse trace created: {self._trace.id}")
        except Exception as e:
            logger.warning(f"Langfuse trace creation failed: {e}")

    @property
    def trace_id(self) -> Optional[str]:
        return self._trace.id if self._trace else None

    def start_iteration(self, iteration: int, operation: str,
                        parent_scores: list[float]) -> "IterationSpan":
        return IterationSpan(self._trace, iteration, operation, parent_scores)

    def end(self, result: dict):
        if self._trace is None:
            return
        try:
            self._trace.update(
                output=result,
                metadata={
                    "total_iterations": result.get("total_iterations"),
                    "best_score": result.get("best_full_score"),
                    "percentile_rank": result.get("percentile_rank"),
                },
            )
            lf = get_langfuse()
            if lf:
                lf.flush()
        except Exception as e:
            logger.warning(f"Langfuse trace end failed: {e}")


class IterationSpan:
    """One span per evolutionary iteration."""

    def __init__(self, trace, iteration: int, operation: str, parent_scores: list[float]):
        self._span = None
        self._trace = trace
        self.operation = operation
        self.iteration = iteration

        if trace is None:
            return
        try:
            self._span = trace.span(
                name=f"iter-{iteration:03d}-{operation}",
                input={"operation": operation, "parent_scores": parent_scores},
                metadata={"iteration": iteration, "operation": operation},
            )
        except Exception as e:
            logger.warning(f"Langfuse iteration span failed: {e}")

    def start_react_step(self, step: int, reason: str) -> "StepSpan":
        return StepSpan(self._span, step, reason)

    def end(self, score: Optional[float], percentile: Optional[float],
            steps: int, elapsed: float):
        if self._span is None:
            return
        try:
            self._span.end(
                output={"score": score, "percentile_rank": percentile,
                        "steps": steps, "elapsed_s": round(elapsed, 1)},
            )
        except Exception as e:
            logger.warning(f"Langfuse iteration span end failed: {e}")


class StepSpan:
    """One span per ReAct step."""

    def __init__(self, parent_span, step: int, reason: str):
        self._span = None

        if parent_span is None:
            return
        try:
            self._span = parent_span.span(
                name=f"step-{step:02d}",
                input={"reason": reason[:500] if reason else ""},
                metadata={"step": step},
            )
        except Exception as e:
            logger.warning(f"Langfuse step span failed: {e}")

    def end(self, action_type: str, action_content: str, observation: str):
        if self._span is None:
            return
        try:
            self._span.end(
                output={
                    "action_type": action_type,
                    "action": action_content[:300] if action_content else "",
                    "observation": observation[:500] if observation else "",
                },
            )
        except Exception as e:
            logger.warning(f"Langfuse step span end failed: {e}")
