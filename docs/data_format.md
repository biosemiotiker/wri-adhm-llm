# Input & Output Format Reference

## Input: Patient Transcript

**Format:** Plain text (UTF-8), no timestamps required.

**File naming:**
- Single file: `<PATIENT_ID>-corrected.txt`
- Directory: `adhs_<PATIENT_ID>/<PATIENT_ID>-corrected.txt`

**Speaker format:**
```
[SPEAKER_00]: Interviewer question or statement
[SPEAKER_01]: Patient response
```

**Example (`examples/adhs_syn001/syn001-corrected.txt`):**
```
[SPEAKER_00]: Guten Tag. Können wir beginnen?
[SPEAKER_01]: Ja, gerne.
[SPEAKER_00]: Erste Frage: Hast du Schwierigkeiten, dich zu konzentrieren?
[SPEAKER_01]: Ja, mir fällt es sehr schwer.
```

### VTT Alternative
WebVTT format with timestamps (for reference):
```
WEBVTT

00:00.000 --> 00:05.000
[SPEAKER_00]: Guten Tag. Können wir beginnen?

00:05.000 --> 00:07.000
[SPEAKER_01]: Ja, gerne.
```

To convert VTT → TXT:
```bash
python 02_preprocessing/remove_timestamps.py input.vtt output-corrected.txt
```

---

## Output: LLM Rating (`wri_results_table.csv`)

**Dimensions:** N participants × 61 columns

**Example row (abbreviated):**

| patient_id | item_1_1 | item_1_2 | … | k1_sum | k1_gb | … | diag | k1_usum | diag_u |
|------------|----------|----------|---|--------|-------|---|------|---------|--------|
| syn001 | 2 | 1 | … | 5 | 2 | … | 1 | 25 | 15 |

### Item Scores (28 columns)

| Criterion | Items | Columns |
|-----------|-------|---------|
| **K1** Attention disturbance | 5 | `item_1_1` to `item_1_5` |
| **K2** Hyperactivity/restlessness | 3 | `item_2_1` to `item_2_3` |
| **K3** Temperament | 3 | `item_3_1` to `item_3_3` |
| **K4** Affective lability | 4 | `item_4_1` to `item_4_4` |
| **K5** Emotional overreactivity | 3 | `item_5_1` to `item_5_3` |
| **K6** Disorganization | 5 | `item_6_1` to `item_6_5` |
| **K7** Impulsivity | 5 | `item_7_1` to `item_7_5` |

**Value scale:**
- `0` = absent
- `1` = mild
- `2` = moderate to severe
- `-1` = not assessable (missing/insufficient info)
- `NaN` = parse error

### Criterion Sums (7 columns)
`k1_sum` to `k7_sum`

**Range:** 0–10 (per criterion)
- Sum of item scores (0, 1, 2 only; -1 excluded)
- Used to check threshold: SUM ≥ {5, 3, 3, 4, 3, 5, 5}

### Global Ratings (7 columns)
`k1_gb` to `k7_gb`

**Range:** 0–4 (or -1 if not assessable)
- 0 = none
- 1 = mild
- 2 = moderate
- 3 = severe
- 4 = very severe

Used to check threshold: GB ≥ 2

### Criterion Met Flags (21 columns)
| Flag | Columns | Meaning |
|------|---------|---------|
| Sum ≥ threshold | `k1_sum_erf` to `k7_sum_erf` | 1 if SUM ≥ item count, else 0 |
| GB ≥ 2 | `k1_gb_erf` to `k7_gb_erf` | 1 if GB ≥ 2, else 0 |
| Criterion met | `k1_erf` to `k7_erf` | 1 if BOTH sum_erf AND gb_erf, else 0 |

### Diagnosis (1 column)
`diag`

**Value:** 0 (no ADHD) or 1 (ADHD)

**Rule:**
```
DIAG = 1 IFF (K1_erf=1 AND K2_erf=1 AND (≥2 of K3_erf to K7_erf = 1))
```

### Uncertainty Estimates (15 columns)

| Column | Range | Meaning |
|--------|-------|---------|
| `k1_usum` to `k7_usum` | 0–100 (%) | Uncertainty: clinician would give different sum? |
| `k1_ugb` to `k7_ugb` | 0–100 (%) | Uncertainty: clinician would give different GB? |
| `diag_u` | 0–100 (%) | Uncertainty: clinician would give different diagnosis? |

**Interpretation:**
- 0 = very confident (clinician likely agrees)
- 50 = uncertain
- 100 = very uncertain (clinician likely disagrees)

---

## Human Reference Data (`demo_data_synthetic.csv`)

Optional reference table with human WRI ratings for agreement/accuracy checks.

**Columns:**

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `patient_id` | str | `syn001` | Identifier (matches transcript) |
| `1_SC` to `7_SC` | int | 5 | Criterion sum (human rater) |
| `1_GB` to `7_GB` | int | 2 | Criterion global rating (0–4) |
| `k1_erfuellt` to `k7_erfuellt` | int | 1 | Criterion met (0/1) |
| `diagnose` | int | 1 | ADHD diagnosis (0/1) |
| `age_years` | int | 35 | Demographics (optional) |
| `sex` | str | M/F | Demographics (optional) |

### Synthetic Example

```csv
patient_id,1_SC,1_GB,2_SC,2_GB,...,7_SC,7_GB,k1_erfuellt,...,k7_erfuellt,diagnose,age_years,sex
syn001,5,2,3,1,...,5,3,1,...,1,1,35,M
syn002,3,1,2,1,...,4,2,0,...,0,0,28,F
...
```

Run `add_criteria_columns.py` to automatically compute thresholds:
```bash
python 04_qa/add_criteria_columns.py demo_data_synthetic.csv
```

This adds:
- `k1_sum_met_human`, `k1_gb_met_human`, etc.
- `diagnosis_new` (recalculated from rule)

---

## Raw LLM Output (`<patient_id>.txt`)

Intermediate file (before CSV parsing). Contains the LLM's full response.

**Format:**

```
[1.1]=2
[1.2]=1
[1.3]=0
[1.4]=1
[1.5]=2
[2.1]=2
...
[K1_SUM]=6
[K1_GB]=2
[K2_SUM]=4
[K2_GB]=2
...
[K1_SUM_ERF]=1
[K1_GB_ERF]=1
[K1_ERF]=1
[K2_SUM_ERF]=1
[K2_GB_ERF]=1
[K2_ERF]=1
...
[DIAG]=1
[K1_USUM]=25
[K1_UGB]=30
...
[DIAG_U]=15
```

All 28 items, 7 criteria (SUM + GB), phase 3 checks, diagnosis, and phase 4 uncertainties.

**Parsing:** Regex `\[KEY\]\s*=\s*(-?\d+)` extracts values. Lines without matches → NaN in CSV.

---

## QC Output (`*_validation_issues.csv`)

Rows where criterion rules were violated (from `check_criteria_and_diagnosis.py`).

**Columns (in addition to data columns):**
- `issue_type`: One of:
  - `k1_sum_flag_mismatch` (and k2–k7)
  - `k1_gb_flag_mismatch` (and k2–k7)
  - `k1_overall_flag_mismatch` (and k2–k7)
  - `diagnosis_new_rule_mismatch`
  - `diagnose_old_vs_new_differs`

**Use:** Identify rows needing manual review or data cleaning.

---

## Notes

- **Patient IDs:** Can be pseudonymized (e.g., `syn001`, `adhs_abc123`). Avoid real names or identifiers.
- **Missing values:** Use `NA`, `NaN`, or empty cell. Parsed as NaN in output.
- **Encoding:** UTF-8 required (German umlauts: ä, ö, ü, ß).
- **Case sensitivity:** `[SPEAKER_00]` vs. `[speaker_00]` — code is case-sensitive; use exact format.
- **Newlines:** Unix (LF) or Windows (CRLF) both work.
