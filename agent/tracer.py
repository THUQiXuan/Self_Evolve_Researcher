"""SER — Langfuse tracing singleton (Langfuse SDK v3/v4 compatible).

All other modules import `get_langfuse()` from here.
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
    """One trace per competition run."""

    def __init__(self, competition_id: str, instance_id: int, time_limit: int):
        self.competition_id = competition_id
        self.instance_id = instance_id
        self._trace_id: Optional[str] = None
        self._root_span = None

        lf = get_langfuse()
        if lf is None:
            return

        try:
            from langfuse.types import TraceContext
            import time as _time

            # Create trace id
            self._trace_id = lf.create_trace_id()

            # Immediately register the trace with correct name/tags via ingestion API
            # (the root span won't flush until 3h later, so we pre-register here)
            self._ingest_trace_create(
                trace_id=self._trace_id,
                name="ser-run",
                input={"competition": competition_id, "instance": instance_id,
                       "time_limit": time_limit},
                tags=[competition_id, f"inst-{instance_id}"],
                metadata={"competition": competition_id, "instance_id": instance_id},
            )

            # Open a root span — used as parent for iteration spans
            self._root_span = lf.start_observation(
                trace_context=TraceContext(trace_id=self._trace_id),
                name="ser-run",
                as_type="span",
                input={"competition": competition_id, "instance": instance_id,
                       "time_limit": time_limit},
                metadata={"competition": competition_id, "instance_id": instance_id},
            )
            logger.info(f"Langfuse trace created: {self._trace_id}")
        except Exception as e:
            logger.warning(f"Langfuse trace creation failed: {e}")
            self._trace_id = None
            self._root_span = None

    def _ingest_trace_create(self, trace_id: str, name: str, input: dict,
                              tags: list, metadata: dict):
        """Send a trace-create event directly via ingestion API so the trace
        name/tags are visible immediately (before root span flushes)."""
        try:
            import urllib.request as _ur
            import json as _json
            import base64 as _b64
            import os as _os
            from datetime import datetime, timezone

            pk = _os.environ.get("SER_LANGFUSE_PUBLIC_KEY", "")
            sk = _os.environ.get("SER_LANGFUSE_SECRET_KEY", "")
            host = _os.environ.get("SER_LANGFUSE_HOST", "https://cloud.langfuse.com")
            creds = _b64.b64encode(f"{pk}:{sk}".encode()).decode()
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            payload = {"batch": [{
                "id": f"tc-{trace_id}",
                "type": "trace-create",
                "body": {"id": trace_id, "name": name, "input": input,
                         "tags": tags, "metadata": metadata},
                "timestamp": ts,
            }]}
            req = _ur.Request(
                f"{host}/api/public/ingestion",
                _json.dumps(payload).encode(),
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Basic {creds}")
            with _ur.urlopen(req, timeout=10) as r:
                r.read()
        except Exception as e:
            logger.debug(f"trace pre-register failed (non-fatal): {e}")

    @property
    def trace_id(self) -> Optional[str]:
        return self._trace_id

    def start_iteration(self, iteration: int, operation: str,
                        parent_scores: list) -> "IterationSpan":
        return IterationSpan(self._root_span, iteration, operation, parent_scores)

    def end(self, result: dict):
        try:
            if self._root_span is not None:
                self._root_span.update(output=result)
                self._root_span.end()
            # Also push final output to trace header via ingestion API
            if self._trace_id:
                self._ingest_trace_create(
                    trace_id=self._trace_id,
                    name="ser-run",
                    input={},
                    tags=[self.competition_id, f"inst-{self.instance_id}"],
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

    def __init__(self, parent_span, iteration: int, operation: str, parent_scores: list):
        self._span = None
        self.operation = operation
        self.iteration = iteration

        if parent_span is None:
            return
        try:
            self._span = parent_span.start_observation(
                name=f"iter-{iteration:03d}-{operation}",
                as_type="span",
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
            self._span.update(
                output={"score": score, "percentile_rank": percentile,
                        "steps": steps, "elapsed_s": round(elapsed, 1)},
            )
            self._span.end()
        except Exception as e:
            logger.warning(f"Langfuse iteration span end failed: {e}")


class StepSpan:
    """One span per ReAct step."""

    def __init__(self, parent_span, step: int, reason: str):
        self._span = None

        if parent_span is None:
            return
        try:
            self._span = parent_span.start_observation(
                name=f"step-{step:02d}",
                as_type="span",
                input={"reason": reason[:500] if reason else ""},
                metadata={"step": step},
            )
        except Exception as e:
            logger.warning(f"Langfuse step span failed: {e}")

    def end(self, action_type: str, action_content: str, observation: str):
        if self._span is None:
            return
        try:
            self._span.update(
                output={
                    "action_type": action_type,
                    "action": action_content[:300] if action_content else "",
                    "observation": observation[:500] if observation else "",
                },
            )
            self._span.end()
        except Exception as e:
            logger.warning(f"Langfuse step span end failed: {e}")
