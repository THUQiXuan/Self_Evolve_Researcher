"""SER — Mutation prompt templates."""


def build_initial_prompt(
    competition_context: str,
    gpu_id: int = None,
    data_profile: str = "",
    strategy: str = "",
) -> str:
    """Build the prompt for initial solution generation (no parent)."""
    is_gpu = gpu_id is not None

    data_section = ""
    if data_profile:
        data_section = f"""
## Data Analysis (from EDA phase)
{data_profile}
"""

    strategy_section = ""
    if strategy:
        strategy_section = f"""
## Recommended Strategy (from planning phase)
{strategy}

Follow this strategy. If you discover issues during execution, adapt but explain why.
"""

    return f"""## Task: Create an initial solution for this Kaggle competition

{competition_context}
{data_section}
{strategy_section}
## Instructions

1. Spend 1 step to explore the data directory (ls, head -3 on key files).
2. Write a COMPLETE solution script that:
   - Loads and preprocesses data
   - Trains a strong model appropriate for the task type
   - Generates predictions on the test set
   - Saves as `./submission.csv` matching sample_submission format exactly
3. Run the script.
4. Submit: Action: submit(submission.csv)

## General Approach
1. Based on the data, decide the task type and modality.
2. Write a COMPLETE solution script. Choose a strong model appropriate for the modality:
   - Tabular: gradient boosting (LightGBM, XGBoost) with feature engineering
   - Text: {"fine-tune a pretrained transformer model" if is_gpu else "TF-IDF + gradient boosting (LightGBM/XGBoost). Do NOT fine-tune transformers on CPU."}
   - Images: {"fine-tune a pretrained CNN or ViT from timm" if is_gpu else "use pre-extracted features + gradient boosting (CPU only)"}
   - Audio: convert to spectrograms, then {"treat as images" if is_gpu else "extract features + gradient boosting"}
   - Multi-modal: combine modality-specific approaches
3. Run the script, verify submission format, and submit.

## Key Rules
- Write the ENTIRE solution in ONE script, then run it
- Save submission as `./submission.csv` (current directory)
"""


def _build_strategic_reflection(best_percentile_rank: float) -> str:
    """Build the Strategic Reflection section based on percentile rank."""
    pr = best_percentile_rank

    if pr > 80:
        strategy = """You are already in the top 20%. Focus on SMALL, SAFE improvements:
- Ensemble: average predictions from multiple folds or models
- Post-processing: threshold optimization, rank averaging, prediction clipping
- Hyperparameter tuning: small adjustments to learning rate, regularization
- Do NOT change the core model architecture — it's working well
- Any change must be validated locally. If local metric is worse, DO NOT submit."""
    elif pr >= 40:
        strategy = """There is significant room for improvement. Make TARGETED, meaningful changes:
- Consider a stronger model or better preprocessing pipeline
- Add cross-validation if not already using it
- Review the data pipeline for bugs or missed features
- Each change should address a specific identified weakness
- Change ONE thing at a time so you can measure its impact"""
    else:
        strategy = """Your approach is fundamentally underperforming. INCREMENTAL changes will NOT close this gap.
You MUST try a RADICALLY DIFFERENT approach:
- Use a completely different model family or problem formulation
- Re-examine your understanding of the task — are you solving the right problem?
- Look at the data again from scratch — is there a key signal you're missing?
- Consider whether your data preprocessing is destroying information
- Do NOT just tune hyperparameters of the existing approach — that cannot fix a fundamental issue
- It is better to try a bold new idea that might fail than to submit another minor variant"""

    return f"""## Strategic Reflection (MANDATORY — do this BEFORE writing any code)

Your current best solution ranks at the {pr:.0f}th percentile on the leaderboard.

{strategy}

Before writing ANY code, you MUST explicitly answer these questions in your reasoning:
1. **What patterns do you see in the population history?** Which approaches worked, which failed, what's the trend?
2. **What is the SINGLE biggest weakness** of the current best solution?
3. **Is an incremental change enough, or do you need a fundamentally different approach?**
4. **What specific change will you make, and WHY do you expect it to improve the score?**
5. **How will you verify the improvement** before submitting? (What local validation metric will you check?)

Only after answering these questions should you proceed to write code.
"""


def _build_conservative_rule(best_percentile_rank: float) -> str:
    """Build the Conservative Improvement Rule, conditional on percentile rank."""
    if best_percentile_rank > 80:
        return """**CONSERVATIVE IMPROVEMENT RULE (you are in the top 20%):**
- Do NOT rewrite the entire solution from scratch. Start from the parent code and change ONE thing at a time.
- Keep the parts that work (data loading, submission format, model architecture) and only change the weakest component.
- Make small improvements: hyperparameters, ensembling, post-processing.
- If the parent solution has a clear bug, fix ONLY that bug and keep everything else the same.
- Test your change locally before submitting — if local validation is worse than parent, revert and try something else."""
    elif best_percentile_rank >= 40:
        return """**TARGETED IMPROVEMENT RULE:**
- Start from the parent code but make ONE meaningful, targeted change.
- Focus on the weakest component — don't change things that are already working well.
- If the same approach has been tried 3+ times without improvement, consider switching to a fundamentally different strategy.
- Test your change locally before submitting — if local validation is worse than parent, revert and try something else."""
    else:
        return """**BOLD CHANGE RULE (your approach is fundamentally underperforming):**
- You ARE allowed and ENCOURAGED to rewrite the entire solution from scratch if the current approach is clearly not working.
- Incremental tweaks to a failing approach waste iterations — be bold.
- Try a completely different model family, problem formulation, or feature set.
- Still validate locally before submitting, but don't be afraid to try something radically new."""


def build_mutation_prompt(
    competition_context: str,
    parent_code: str,
    parent_score: float,
    is_lower_better: bool = False,
    population_history: str = "",
    best_percentile_rank: float = 0,
    gpu_id: int = None,
    data_profile: str = "",
    critic_feedback: str = "",
) -> str:
    """Build the prompt for mutation (improve existing solution)."""
    better_dir = "lower" if is_lower_better else "higher"

    history_section = ""
    if population_history:
        history_section = f"""
## Population History (previous attempts and scores)
{population_history}

Analyze what has been tried. Do NOT repeat the same approach. Try something SIGNIFICANTLY different.
"""

    reflection_section = _build_strategic_reflection(best_percentile_rank)
    conservative_rule = _build_conservative_rule(best_percentile_rank)

    critic_section = ""
    if critic_feedback:
        critic_section = f"""
## Critic Feedback (from previous solution review)
{critic_feedback}
Address the flagged issues in your improved solution.
"""

    data_section = ""
    if data_profile:
        data_section = f"""
## Data Analysis (from EDA phase)
{data_profile}
"""

    return f"""## Task: Improve an existing solution for this Kaggle competition

{competition_context}
{data_section}
## Parent Solution (score: {parent_score}, {better_dir} is better)

```python
{parent_code}
```
{history_section}
{reflection_section}
{critic_section}
## Before Improving: Diagnose the Problem

FIRST, analyze why the parent solution scores {parent_score}. Check these in order:
1. **Are the labels/targets correct?** Print 3-5 training examples to verify label construction is right.
2. **Is the submission format correct?** Compare submission.csv columns, dtypes, and row count to sample_submission.csv.
3. **Is the model actually learning?** Check if training loss decreases and validation metric improves.
4. **What is the bottleneck?** Is it data preprocessing, model capacity, feature engineering, or post-processing?

## Improvement Strategies (pick the most impactful one)

### High-Impact Changes (try these first):
1. **Fix bugs in label/target computation** — this is often the #1 issue. Wrong labels = bad model no matter how good the architecture.
2. **Switch to a stronger or different model architecture** (e.g., DeBERTa-v3, RoBERTa-large, larger ViT)
4. **Train longer / more data**: increase epochs to 5+, use full dataset, use cosine annealing schedule
5. **Ensemble**: average predictions from K-fold CV (5 folds), or blend 2-3 different model types
6. **Better preprocessing**: handle missing values, add text cleaning, stronger image augmentation (mixup, cutmix)

### Medium-Impact Changes:
6. **Hyperparameter tuning**: learning rate, batch size, regularization, number of estimators
7. **Post-processing**: threshold optimization, rank averaging, clip predictions to valid range
8. **Cross-validation**: use K-fold CV to avoid overfitting, average fold predictions for test

### CRITICAL: Code Efficiency (affects iteration count)
- Faster code = more iterations = better final score

### Metric-Specific Optimizations:
- **AUC**: optimize threshold, use class weights for imbalanced data
- **RMSE/MAE**: try log-transform of target, add outlier handling
- **Jaccard/F1**: optimize per-class thresholds
- **MAP@K**: focus on ranking quality, not just classification accuracy

## Instructions

1. **Diagnose first**: identify the biggest weakness in the parent solution
2. Write a COMPLETE improved solution script — start from the parent code and make TARGETED changes
3. **Add local validation**: compute the competition metric on a 10% holdout to verify improvement before submitting
4. Run it
5. Submit: Action: submit(submission.csv)

{conservative_rule}

IMPORTANT:
- Save submission as `./submission.csv`
"""
