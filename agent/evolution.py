"""SER — Evolutionary Search (population management, selection, mutation/crossover)."""

import math
import random
import logging
from typing import Optional

from database import Program, ProgramDatabase

from config import SELECTION_TEMPERATURE, CROSSOVER_PROB

logger = logging.getLogger(__name__)


def temperature_scaled_rank_selection(
    programs: list[Program],
    n: int = 1,
    temperature: float = SELECTION_TEMPERATURE,
    rng: Optional[random.Random] = None,
) -> list[Program]:
    """
    Temperature-scaled rank-based selection from the paper.

    p(i) = (N - r_i + 1)^(1/T) / Σ(N - r_j + 1)^(1/T)

    where r_i is the rank (1=best), T is temperature, N is population size.
    Lower temperature → more greedy (exploits top solutions).
    """
    if not programs:
        raise ValueError("Cannot select from empty population")

    if rng is None:
        rng = random.Random()

    N = len(programs)
    # Programs should already be sorted best-first by comparable_search_score
    sorted_progs = sorted(programs, key=lambda p: p.comparable_search_score, reverse=True)

    # Compute selection weights
    weights = []
    for rank_idx in range(N):
        r = rank_idx + 1  # 1-indexed rank
        w = (N - r + 1) ** (1.0 / temperature)
        weights.append(w)

    total = sum(weights)
    probs = [w / total for w in weights]

    # Sample n parents
    selected = []
    for _ in range(n):
        r = rng.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                selected.append(sorted_progs[i])
                rank = i + 1
                prob = probs[i]
                logger.info(f"  Selected [{sorted_progs[i].id}] rank={rank}/{N} prob={prob:.2%} score={sorted_progs[i].search_score}")
                break
        else:
            selected.append(sorted_progs[-1])
            logger.info(f"  Selected [{sorted_progs[-1].id}] rank={N}/{N} prob={probs[-1]:.2%} score={sorted_progs[-1].search_score}")

    return selected


def should_crossover(rng: Optional[random.Random] = None) -> bool:
    """Decide whether to do crossover (with probability CROSSOVER_PROB)."""
    if rng is None:
        rng = random.Random()
    return rng.random() < CROSSOVER_PROB


class EvolutionManager:
    """Manages the evolutionary search process."""

    def __init__(
        self,
        database: ProgramDatabase,
        temperature: float = SELECTION_TEMPERATURE,
        crossover_prob: float = CROSSOVER_PROB,
        seed: Optional[int] = None,
    ):
        self.db = database
        self.temperature = temperature
        self.crossover_prob = crossover_prob
        self.rng = random.Random(seed)
        self.generation = 0

    def next_task(self) -> dict:
        """
        Decide the next evolutionary operation.

        Returns dict with:
            - operation: "init", "mutation", or "crossover"
            - parents: list of parent Programs (empty for init)
        """
        scored = self.db.get_scored_programs()

        if len(scored) == 0:
            # No solutions yet — initialize from scratch
            return {"operation": "init", "parents": []}

        if len(scored) >= 2 and should_crossover(self.rng):
            # Crossover: select two distinct parents
            parents = temperature_scaled_rank_selection(
                scored, n=2, temperature=self.temperature, rng=self.rng
            )
            # Ensure distinct parents
            if len(scored) >= 2 and parents[0].id == parents[1].id:
                # Re-sample second parent
                others = [p for p in scored if p.id != parents[0].id]
                if others:
                    parents[1] = temperature_scaled_rank_selection(
                        others, n=1, temperature=self.temperature, rng=self.rng
                    )[0]

            self.generation += 1
            n = len(scored)
            sorted_progs = sorted(scored, key=lambda p: p.comparable_search_score, reverse=True)
            rank1 = next((i+1 for i, p in enumerate(sorted_progs) if p.id == parents[0].id), "?")
            rank2 = next((i+1 for i, p in enumerate(sorted_progs) if p.id == parents[1].id), "?")
            logger.info(f"Gen {self.generation}: CROSSOVER")
            logger.info(f"  Parent A: [{parents[0].id}] score={parents[0].search_score} rank={rank1}/{n}")
            logger.info(f"  Parent B: [{parents[1].id}] score={parents[1].search_score} rank={rank2}/{n}")
            return {"operation": "crossover", "parents": parents}

        else:
            # Mutation: select one parent
            parents = temperature_scaled_rank_selection(
                scored, n=1, temperature=self.temperature, rng=self.rng
            )
            self.generation += 1
            logger.info(
                f"Gen {self.generation}: MUTATION "
                f"parent={parents[0].id}(s={parents[0].search_score})"
            )
            return {"operation": "mutation", "parents": parents}
