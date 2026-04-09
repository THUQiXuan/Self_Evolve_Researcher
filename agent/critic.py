"""SER — Multi-Perspective Critic for code review."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import LLMClient

logger = logging.getLogger(__name__)

# Each perspective is a meta-prompt template
PERSPECTIVES = {
    "correctness": """Review this ML code for CORRECTNESS issues:
- Are labels/targets computed correctly?
- Is the data pipeline correct (no accidental data loss, wrong joins, wrong splits)?
- Does the submission format match requirements?
- Are there off-by-one errors, wrong column names, or dtype mismatches?
- Is the loss function appropriate for the metric?
Output: [OK / ISSUE: description] with confidence [HIGH/MEDIUM/LOW]""",

    "efficiency": """Review this ML code for EFFICIENCY:
- Will it finish within {timeout} minutes?
- Is GPU/CPU being used efficiently?
- Are there unnecessary loops, redundant computations, or memory waste?
- Could batch size be larger? Could data loading be faster?
- Is the model appropriately sized for the time budget?
Output: [OK / SLOW: specific bottleneck and fix] with estimated runtime""",

    "leakage": """Review this ML code for DATA LEAKAGE:
- Is test data used during training?
- Is validation data used for feature engineering?
- Are there features that encode the target?
- Is preprocessing (scaling, encoding) fit on test data?
- Are there temporal leakage issues?
Output: [OK / LEAK: description] with severity [CRITICAL/MINOR]""",

    "robustness": """Review this ML code for ROBUSTNESS issues:
- Will it crash on edge cases (NaN, empty, very large values)?
- Is error handling adequate?
- Could it produce NaN predictions?
- Are there hardcoded assumptions about data that might not hold?
Output: [OK / FRAGILE: description and fix]""",

    "strategy": """Review this ML approach for STRATEGIC quality:
- Is this the right model family for the data type and size?
- Is the training strategy (epochs, folds, LR) reasonable?
- Are there obvious improvements being missed?
- Should this be ensembled with something else?
Output: [OK / IMPROVE: specific suggestion] with expected impact [HIGH/MEDIUM/LOW]""",
}


def run_critic(
    llm: LLMClient,
    code: str,
    data_profile: str,
    competition_description: str,
    timeout_minutes: int = 20,
    perspectives: list[str] = None,
    max_parallel: int = 3,
) -> dict:
    """Run multi-perspective critic on code. Returns aggregated feedback.

    Can run in parallel with training (caller manages this).
    """
    if perspectives is None:
        perspectives = list(PERSPECTIVES.keys())

    results = {}

    def _evaluate(perspective: str) -> tuple[str, str]:
        prompt_template = PERSPECTIVES[perspective]
        prompt = f"""{prompt_template.format(timeout=timeout_minutes)}

### Code to Review
```python
{code[:16000]}
```

### Data Profile
{data_profile[:4000]}

### Competition
{competition_description[:1000]}
"""
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system="You are a code reviewer. Be specific and concise.",
            max_tokens=1500,
            temperature=0.3,
        )
        return perspective, response

    # Parallel evaluation
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(_evaluate, p): p for p in perspectives}
        for future in as_completed(futures):
            try:
                perspective, response = future.result(timeout=60)
                results[perspective] = response
            except Exception as e:
                perspective = futures[future]
                results[perspective] = f"[ERROR: {e}]"
                logger.warning(f"Critic perspective {perspective} failed: {e}")

    # Aggregate: count issues
    issues = []
    for p, r in results.items():
        r_upper = r.upper()
        if any(kw in r_upper for kw in ["ISSUE:", "SLOW:", "LEAK:", "FRAGILE:", "IMPROVE:"]):
            issues.append(f"[{p}] {r}")

    summary = f"Critic: {len(issues)}/{len(perspectives)} perspectives flagged issues."
    if issues:
        summary += "\n" + "\n".join(issues[:5])  # Top 5 issues

    return {
        "perspectives": results,
        "issues": issues,
        "summary": summary,
        "has_critical": any(
            "CRITICAL" in r.upper() or "LEAK:" in r.upper()
            for r in results.values()
        ),
    }
