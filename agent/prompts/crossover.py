"""SER — Crossover prompt templates."""


def build_crossover_prompt(
    competition_context: str,
    parent_a_code: str,
    parent_a_score: float,
    parent_b_code: str,
    parent_b_score: float,
    is_lower_better: bool = False,
    gpu_id: int = None,
    population_history: str = "",
    best_percentile_rank: float = 0,
    data_profile: str = "",
    critic_feedback: str = "",
) -> str:
    """Build the prompt for crossover (combine two parent solutions)."""
    better_dir = "lower" if is_lower_better else "higher"

    population_section = ""
    if population_history:
        population_section = f"""
## Population History
Our best solution is at percentile rank {best_percentile_rank:.1f}% on the leaderboard.

{population_history}
"""

    critic_section = ""
    if critic_feedback:
        critic_section = f"""
## Critic Feedback (from previous solution review)
{critic_feedback}
Address the flagged issues in your combined solution.
"""

    data_section = ""
    if data_profile:
        data_section = f"""
## Data Analysis (from EDA phase)
{data_profile}
"""

    return f"""## Task: Combine two solutions to create a better one

{competition_context}
{data_section}
{population_section}
{critic_section}
## Solution A (score: {parent_a_score}, {better_dir} is better)
```python
{parent_a_code}
```

## Solution B (score: {parent_b_score}, {better_dir} is better)
```python
{parent_b_code}
```

## Strategic Crossover Analysis
Before writing code, analyze BOTH parents from these perspectives:
1. **Architecture**: What model/approach does each parent use? Which is stronger?
2. **Feature Engineering**: What preprocessing/features does each parent have? What's unique?
3. **Training Strategy**: Epochs, folds, learning rate, augmentation — which settings are better?
4. **Ensemble Potential**: Can outputs from both parents be combined (averaging, stacking)?

Choose the crossover strategy that maximizes the strengths of BOTH parents.

## Crossover Strategies (pick the best one for this competition)

1. **Weighted Ensemble** (most reliable): Run BOTH models, average their predictions with weights proportional to scores
2. **Best-of-Both**: Take the better preprocessing pipeline + the better model architecture
3. **Stacking**: Use predictions from both models as features for a meta-learner (e.g., logistic regression or simple NN)
4. **Diverse Ensemble**: If both use similar models, modify one to use a different architecture, then ensemble

## Instructions

1. Analyze what makes each solution good/bad
2. Write a COMPLETE combined solution script that produces better predictions
3. Run it (use up to 15 minutes for training)
4. Submit: Action: submit(submission.csv)

IMPORTANT:
- Save submission as `./submission.csv`
"""
