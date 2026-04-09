"""SER — Program Database (population storage)."""

import json
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from filelock import FileLock

logger = logging.getLogger(__name__)


@dataclass
class Program:
    """A candidate solution in the evolutionary population."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    code: str = ""                          # Full solution code (main script)
    code_path: Optional[str] = None         # Path to saved code file
    parent_ids: list[str] = field(default_factory=list)
    operation: str = "init"                 # "init", "mutation", "crossover"
    search_score: Optional[float] = None    # Score on D_search (eval data)
    val_score: Optional[float] = None       # Score on D_val (eval data, for final selection)
    full_score: Optional[float] = None      # Score on full test set (for percentile rank)
    is_lower_better: bool = False           # Whether lower score = better
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def comparable_search_score(self) -> float:
        """Return score normalized so higher is always better."""
        if self.search_score is None:
            return float("-inf")
        return -self.search_score if self.is_lower_better else self.search_score

    @property
    def comparable_val_score(self) -> float:
        if self.val_score is None:
            return float("-inf")
        return -self.val_score if self.is_lower_better else self.val_score

    @property
    def comparable_full_score(self) -> float:
        if self.full_score is None:
            return float("-inf")
        return -self.full_score if self.is_lower_better else self.full_score


class ProgramDatabase:
    """In-memory population with disk persistence."""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.programs: list[Program] = []

    def add(self, program: Program) -> None:
        """Add program with file lock for multi-process safety."""
        self.programs.append(program)
        lock_path = self.save_dir / ".write.lock"
        with FileLock(str(lock_path), timeout=30):
            self._save_program(program)
        logger.info(
            f"Added program {program.id} (op={program.operation}, "
            f"search={program.search_score}, val={program.val_score})"
        )

    def reload_from_disk(self):
        """Reload programs from disk that were added by other instances."""
        known_ids = {p.id for p in self.programs}
        for json_file in sorted(self.save_dir.glob("*.json")):
            if json_file.name == "summary.json":
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)
                pid = data.get("id")
                if pid and pid not in known_ids:
                    valid_fields = {k: v for k, v in data.items()
                                    if k in Program.__dataclass_fields__}
                    prog = Program(**valid_fields)
                    self.programs.append(prog)
                    known_ids.add(prog.id)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    def get_scored_programs(self) -> list[Program]:
        """Return programs with valid search scores, sorted best-first."""
        scored = [p for p in self.programs if p.search_score is not None]
        scored.sort(key=lambda p: p.comparable_search_score, reverse=True)
        return scored

    def get_best_by_val(self) -> Optional[Program]:
        """Return the program with the best validation score."""
        scored = [p for p in self.programs if p.val_score is not None]
        if not scored:
            return None
        return max(scored, key=lambda p: p.comparable_val_score)

    def get_best_by_search(self) -> Optional[Program]:
        """Return the program with the best search score."""
        scored = self.get_scored_programs()
        return scored[0] if scored else None

    def size(self) -> int:
        return len(self.programs)

    def _save_program(self, program: Program) -> None:
        """Persist program to disk as JSON."""
        path = self.save_dir / f"{program.id}.json"
        data = asdict(program)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_summary(self) -> None:
        """Save a summary of all programs."""
        best_search = self.get_best_by_search()
        best_val = self.get_best_by_val()

        programs_summary = []
        for p in self.programs:
            programs_summary.append({
                "id": p.id,
                "operation": p.operation,
                "parent_ids": p.parent_ids,
                "search_score": p.search_score,
                "val_score": p.val_score,
                "full_score": p.full_score,
                "metadata": p.metadata,
                "created_at": p.created_at,
            })

        summary = {
            "total_programs": len(self.programs),
            "scored_programs": len(self.get_scored_programs()),
            "best_search_score": best_search.search_score if best_search else None,
            "best_val_score": best_val.val_score if best_val else None,
            "best_program_id": best_val.id if best_val else None,
            "programs": programs_summary,
        }
        path = self.save_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
