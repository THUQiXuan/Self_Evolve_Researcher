"""SER — Planning Meta-Prompt.

Competition-agnostic prompt for single-LLM-call strategy design,
driven by the data profile from the EDA phase.
"""


def build_planning_prompt(
    data_profile: str,
    competition_description: str,
    gpu_available: bool,
    time_budget_minutes: int,
    population_history: str = "",
) -> str:
    """Build competition-agnostic planning meta-prompt."""
    device = "1x GPU with 81GB VRAM" if gpu_available else "CPU only (100 cores, no GPU)"

    history_section = ""
    if population_history:
        history_section = f"""
### Previous Attempts
{population_history}
Analyze what worked and what failed. Your strategy must be DIFFERENT from failed approaches.
"""

    return f"""## Task: Design an ML Solution Strategy

Based on the data analysis below, design the best approach for this competition.

### Data Profile
{data_profile}

### Competition Description
{competition_description}

### Constraints
- Compute: {device}
- Time budget: {time_budget_minutes} minutes for training
- Code execution timeout: 20 minutes per script
- Must produce a submission.csv matching the required format
{history_section}
### Strategy Design Protocol

Answer each question concisely but specifically:

1. **Problem Formulation**
   - What exactly are we predicting? What's the target format?
   - What metric are we optimizing? Any metric-specific tricks?
   - Is this a standard task or does it need special formulation?

2. **Data Pipeline**
   - What preprocessing is needed? (missing values, encoding, normalization)
   - What feature engineering opportunities did the data profile reveal?
   - For images/audio: what resolution? what augmentation? (justify from data stats)
   - Any data quality issues to handle first?

3. **Model Selection**
   - What model architecture? (justify based on data modality, size, and compute constraints)
   - What pretrained weights if applicable?
   - Ensemble strategy?

4. **Training Strategy**
   - Number of folds, epochs, batch size (justify based on data size and time budget)
   - Learning rate and schedule
   - Regularization approach
   - Estimated training time

5. **Risk Assessment**
   - What could go wrong? (timeout, OOM, wrong format, overfitting)
   - Fallback plan if primary approach fails?
   - What's the minimum viable submission?

Output your strategy as a structured plan the execution agent can follow.
"""
