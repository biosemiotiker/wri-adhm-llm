# WRI-ADHD Pipeline Guide

## Overview

The complete pipeline from raw transcript to statistical results:

```
1. TRANSCRIPT (.txt)
     ↓
2. wri_rating_gptoss.py
   - LLM rates 28 items + 7 criteria + diagnosis
   - Output: wri_results_table.csv (raw LLM ratings)
     ↓
3. add_criteria_columns.py
   - Computes thresholds for human reference data
   - Derives criterion-met flags and diagnosis rule
     ↓
4. check_criteria_and_diagnosis.py
   - Validates criterion and diagnosis rules
   - Reports discrepancies
     ↓
5. (Merge human + LLM data)
     ↓
6. llm_criteria_long.R
   - Agreement: Cohen's κ, ICC(A,1)
   - Diagnostic accuracy: sensitivity, specificity, ROC/AUC
   - GLMM: model LLM ~ (1|participant) + model
   - Figures & tables
```

## Step 1: Rating Transcripts

### Single transcript (test)
```bash
python 03_llm_scoring/wri_rating_gptoss.py --prompt-file <lokaler_prompt.txt> \
    --test \
    --test-file examples/adhs_syn001/syn001-corrected.txt \
    --model "gpt-oss-20b" \
    --temperature 0.75
```

**Output:**
- `./wri_output/syn001.txt`: Raw LLM response with [KEY]=value pairs
- LM Studio must be running at `http://localhost:1234`

### Batch rating
```bash
python 03_llm_scoring/wri_rating_gptoss.py --prompt-file <lokaler_prompt.txt> \
    --input-folder /path/to/transcripts \
    --output-dir ./wri_results \
    --model "gpt-oss-120b" \
    --temperature 1.5
```

**Input directory structure:**
```
/path/to/transcripts/
├── adhs_001/
│   └── 001-corrected.txt
├── adhs_002/
│   └── 002-corrected.txt
└── adhs_003/
    └── 003-corrected.txt
```

**Output:**
- `wri_output/syn001.txt`, `syn002.txt`, …
- `wri_output/wri_results_table.csv` (merged results)

### CLI options
```
--model           Model name (default: gpt-oss-20b)
--temperature     0.0–1.5 (default: 0.75)
--top-p           Top-p nucleus sampling (default: 1.0)
--context-length  Context window size (default: 40000)
--test            Single-file test mode (requires --test-file)
--test-file       Path to single *-corrected.txt
```

## Step 2: Parse LLM Output → CSV

Automatically called at end of `wri_rating_gptoss.py`, but can be re-run:
```bash
# (Happens automatically in main(), no separate call needed)
```

**Parsing logic:**
- Regex: `\[KEY\]\s*=\s*(-?\d+)`
- Unparsed values → NaN
- Output: `wri_results_table.csv` with columns:
  - `item_1_1` … `item_7_5` (28 item scores)
  - `k1_sum` … `k7_sum` (criterion sums)
  - `k1_gb` … `k7_gb` (global ratings 0–4)
  - `k1_sum_erf` … `k7_sum_erf` (sum threshold met: 0/1)
  - `k1_gb_erf` … `k7_gb_erf` (GB ≥2: 0/1)
  - `k1_erf` … `k7_erf` (criterion met: 0/1)
  - `diag` (ADHD diagnosis: 0/1)
  - `k1_usum` … `k7_usum` (uncertainty 0–100)
  - `k1_ugb` … `k7_ugb` (uncertainty 0–100)
  - `diag_u` (diagnosis uncertainty 0–100)

## Step 3: Add Criterion Thresholds (Human Data)

If you have a CSV with human WRI ratings, derive the criterion-met flags:

```bash
python 04_qa/add_criteria_columns.py examples/demo_data_synthetic.csv
```

**Input CSV columns (human):**
- `1_SC`, `2_SC`, … `7_SC`: criterion sums
- `1_GB`, `2_GB`, … `7_GB`: global ratings (0–4)
- `k1_erfuellt` … `k7_erfuellt`: criterion-met flags (to update)
- `diagnose`: diagnosis column (0/1)

**Output:** `*_added_columns.csv` with:
- New: `k1_sum_met_human`, `k1_gb_met_human`, etc.
- Updated: `k1_erfuellt`, `diagnose` (recalculated)

**Thresholds (hardcoded):**
```python
K1 ≥ 5, K2 ≥ 3, K3 ≥ 3, K4 ≥ 4, K5 ≥ 3, K6 ≥ 5, K7 ≥ 5
GB ≥ 2 for all
```

## Step 4: Validate Criterion Rules

```bash
python 04_qa/check_criteria_and_diagnosis.py examples/demo_data_synthetic_added_columns.csv
```

**Checks:**
1. Sum threshold met (SUM ≥ item count)? → `k1_sum_met_human`
2. GB threshold met (GB ≥ 2)? → `k1_gb_met_human`
3. Criterion met (both)? → `k1_erfuellt`
4. Diagnosis rule (K1 ∧ K2 ∧ ≥2 of K3–K7)? → `diagnosis_new`
5. Old vs. new diagnosis differ? → flag for review

**Output:** `*_validation_issues.csv` (rows with errors)

## Step 5: Merge LLM + Human → Master Table

Typically in R (R scripts are not part of this repository):
```r
# Pseudocode
llm <- read.csv("wri_results_table.csv")
human <- read.csv("demo_data_added_columns.csv")
merged <- full_join(llm, human, by="patient_id")
write.csv(merged, "analysis_data_combined.csv")
```

## Step 6: Statistical Analysis (R)

```bash
cd analysis
Rscript llm_criteria_long.R
```

### Agreement
```r
irr::kappa2(data.frame(human=x, llm=y))  # Cohen's κ
irrCAC::gwet.ac.1(ratings)                 # Gwet's AC1
nlme::gls(llm ~ human, correlation=...)    # ICC(A,1)
```

### Diagnostic Accuracy
```r
pROC::roc(gold_standard ~ llm_diagnosis)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
```

### GLMM
```r
lme4::glmer(
  llm_score ~ 1 + (1 | participant) + model_condition,
  family="gaussian"
)
```

### Figures
- Bar plots (agreement per criterion)
- Bland-Altman plots (LLM vs. human)
- ROC curves (diagnostic accuracy)
- Uncertainty distributions

### Tables
- gt package: agreement summary, accuracy, GLMM results
- webshot2: PNG exports for manuscripts

## Troubleshooting

### LM Studio connection fails
- Ensure LM Studio is running: `http://localhost:1234/health`
- Check firewall and port 1234
- Fallback: adapt `lmstudio` endpoint in code

### Parsing fails (NaN in CSV)
- Check LLM output format: must have `[KEY]=value` lines
- Run `check_rules.py` to diagnose format errors
- Increase `--temperature` if model is too conservative

### Criterion thresholds don't match
- Edit `THRESHOLDS` dict in `add_criteria_columns.py`
- Re-run the script
- Re-run validation step

## Models & Settings (Paper)

| Model | Temperatures | Backend |
|-------|-------------|---------|
| gpt-oss-20b | 0, 0.75, 1.5 | LM Studio |
| gpt-oss-120b | 0, 0.75, 1.5 | LM Studio |
| Qwen-32B | 0.75 | vLLM or HF Transformers |
| Qwen-235B | 0.75 | vLLM or HF Transformers |

Default (most balanced): **gpt-oss-20b at 0.75**

## Notes

- **Prompt language:** German (validated in study). English translation provided for reference only.
- **Uncertainty scale:** 0–100 (%)
  - 0 = clinician very likely agrees
  - 50 = complete uncertainty
  - 100 = clinician very likely disagrees
- **Diagnostic decision:** K1 ∧ K2 ∧ (≥2 of K3–K7) = ADHD
- **Not clinical:** Use only for research. Moderate agreement (κ ≈ 0.5) and imperfect accuracy (0.83).
