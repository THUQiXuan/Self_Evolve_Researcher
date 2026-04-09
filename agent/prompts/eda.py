"""SER — EDA (Exploratory Data Analysis) Meta-Prompt.

Competition-agnostic prompt that teaches the agent HOW to analyze data,
without containing any competition-specific advice or model recommendations.
"""


def build_eda_prompt(
    data_dir: str,
    description: str,
    sample_submission_preview: str,
    output_dir: str = "",
) -> str:
    """Build competition-agnostic EDA meta-prompt."""
    save_dir = output_dir or data_dir
    return f"""## Task: Comprehensive Data Analysis

You have access to a machine learning competition dataset. Your job is to thoroughly
analyze the data and produce a structured report. Do NOT build any models yet.

### Data Location
- Data directory: {data_dir}
- Sample submission preview:
{sample_submission_preview}

### Competition Description
{description}

### Analysis Protocol

Execute Python/bash code to analyze the data systematically.

**Step 1: File Inventory & Schema**
- List all files/directories with sizes
- For each tabular file: shape, columns, dtypes, first 5 rows
- For image/audio directories: count files, sample file sizes, naming patterns
- Identify: ID column(s), target column(s), feature columns

**Step 2: Statistical Profile**
Run a script that outputs:
- For numeric columns: mean, std, min, max, quartiles, % zeros, skewness
- For categorical columns: unique count, top-5 values with frequencies
- Target variable: full distribution (class counts for classification, histogram stats for regression)
- Missing values: count and % per column
- Duplicates check
- Constant or near-constant columns

**Step 3: Data Quality & Relationships**
- Feature-target correlations (top 10 by absolute correlation)
- Train/test feature distribution comparison (at least top-5 features)
- Check for potential data leakage (features identical to or strongly correlated with target)
- For text: sample lengths, vocabulary size, language detection
- For images: sample dimensions, channels, aspect ratios, file format

**Step 4: Task Characterization**
Based on ALL findings, produce a structured summary:
```
=== DATA PROFILE ===
Task type: [classification/regression/ranking/segmentation/generation/etc.]
Data modality: [tabular/text/image/audio/multimodal/time-series]
Evaluation metric: [from description]
Train samples: [N]
Test samples: [N]
Features: [N numeric, N categorical, N text, N image]
Target distribution: [balanced/imbalanced (ratio X:1) / continuous (range)]
Key challenges: [list top 3-5]
Missing data: [none / minor (<5%) / significant (>20%)]
Data size: [small <10K / medium 10K-100K / large >100K / very large >1M]
=== END PROFILE ===
```

If the dataset contains images, save 3 representative sample images as:
  {save_dir}/sample_images/sample_001.png
  {save_dir}/sample_images/sample_002.png
  {save_dir}/sample_images/sample_003.png
Resize to max 512px on longest side to keep file size small.

IMPORTANT:
- Run ALL analysis code in ONE comprehensive script per step
- Print results clearly with section headers
- Do NOT build models or make predictions
- Do NOT spend more than 4 steps total
- The goal is UNDERSTANDING, not prediction
"""
