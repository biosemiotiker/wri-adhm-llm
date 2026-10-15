# WRI-ADHD LLM Rating Tool

A Python toolkit for automated rating of Wender-Reimherr Interview (WRI) transcripts using large language models. The pipeline evaluates ADHD diagnostic criteria across 28 items, 7 criteria, and produces diagnostic classification and uncertainty estimates.

**Paper:** "LLM-Based Evaluation of Wender-Reimherr Interview Transcripts for ADHD Assessment" (in preparation)

**Key features:**
- Batch or single-transcript rating via local LLM (LM Studio)
- Structured 4-phase prompt in German (original study language)
- 28 WRI items → 7 criterion sums/global ratings → criterion-level diagnosis → ADHD yes/no
- Uncertainty quantification per criterion and diagnosis (0–100 scale)
- Post-processing, validation, and R-based statistical analysis
- Support for gpt-oss-20b/120b and Qwen LLM backends

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
huggingface-cli login  # Required for Pyannote (WhisperX diarization)
```

### 2. (Optional) Transcribe audio with WhisperX

**Stage 1: Transcription**
```bash
./01_transcription/whisperx_transcribe.sh \
    --input-dir /path/to/audio/ \
    --output-dir /path/to/vtt/
```
See [`docs/whisperx.md`](docs/whisperx.md) for details.

### 3. Prepare transcripts (remove timestamps)

**Stage 2: Preprocessing**
```bash
python 02_preprocessing/remove_timestamps.py \
    --input transcripts/audio_id.vtt \
    --output data/adhs_audio_id/audio_id-corrected.txt
```

### 4. Score transcripts with LLM

**Stage 3: LLM Scoring**

**Option A: gpt-oss-20b (via LM Studio)**
```bash
python 03_llm_scoring/wri_rating_gptoss.py \
    --input-folder data/ \
    --output-base results/ \
    --temperatures 0.0 0.75 1.5 \
    --backend lmstudio
```

**Option B: Qwen3 (via HuggingFace Transformers)**
```bash
python 03_llm_scoring/wri_rating_qwen.py \
    --input-folder data/ \
    --output-base results/ \
    --model Qwen/Qwen3-32B-Instruct \
    --temperatures 0.0
```

### 5. Quality assurance

**Stage 4: QA**
```bash
python 04_qa/check_criteria_and_diagnosis.py wri_results_table.csv
python 04_qa/add_criteria_columns.py wri_results_table.csv
python 04_qa/check_rules.py
```

## Complete Pipeline Overview

```
AUDIO FILE (.wav, .mp3)
    ↓
WhisperX (diarization)
    ↓
output.vtt (timestamped, [SPEAKER_00]/[SPEAKER_01])
    ↓
02_preprocessing/remove_timestamps.py
    ↓
TRANSCRIPT (.txt, clean format)
    ↓
03_llm_scoring/wri_rating_{gptoss,qwen}.py → LLM rating (--prompt-file)
    ↓
wri_results_table.csv (28 items + 7 criteria + diagnosis)
    ↓
add_criteria_columns.py → Derive human reference thresholds
    ↓
check_criteria_and_diagnosis.py → Validate criterion & diagnosis rules
    ↓
llm_criteria_long.R → Agreement (Cohen's κ, ICC), diagnostic accuracy (ROC/AUC),
                       GLMM, figures, tables
```

## Input Format

**Transcript file** (plain text, `*.txt`):
```
[SPEAKER_00]: Interviewer question or statement
[SPEAKER_01]: Patient response
[SPEAKER_00]: Next question...
```

No timestamps. Speakers are:
- `[SPEAKER_00]` = clinician/interviewer
- `[SPEAKER_01]` = patient (rated)

**Example:** `examples/adhs_syn001/syn001-corrected.txt`

## WRI Scoring Logic

### Phase 1: Item Scores (28 items)
Each item rated on a 0–2 scale (or -1 if not assessable):
- 0 = absent
- 1 = mild
- 2 = moderate to severe
- -1 = not assessable

Items grouped into 7 criteria (K1–K7):
- **K1** (5 items): Attention disturbance
- **K2** (3 items): Hyperactivity / restlessness
- **K3** (3 items): Temperament ("hot temper")
- **K4** (4 items): Affective lability
- **K5** (3 items): Emotional overreactivity (stress intolerance)
- **K6** (5 items): Disorganization
- **K7** (5 items): Impulsivity

### Phase 2: Criterion Sums & Global Ratings
- **SUM**: sum of item scores (0–2 only; -1 excluded)
- **GB** (global rating): 0–4 scale (0 = none, 1 = mild, 2 = moderate, 3 = severe, 4 = very severe)

### Phase 3: Criterion Met & Diagnosis
A criterion is **met** if both:
1. Sum ≥ item count (K1≥5, K2≥3, K3≥3, K4≥4, K5≥3, K6≥5, K7≥5)
2. GB ≥ 2

**ADHD diagnosis** (1 = yes, 0 = no):
- K1 met AND K2 met AND ≥2 of K3–K7 met

### Phase 4: Uncertainty
For each sum, global rating, and diagnosis, an uncertainty score (0–100):
- 0 = clinician very likely agrees
- 100 = clinician very likely disagrees

## Output Format

**`wri_results_table.csv`** (after the scoring script):

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | str | Index: transcript ID |
| `item_1_1` … `item_7_5` | int | 28 item scores |
| `k1_sum` … `k7_sum` | int | Criterion sums |
| `k1_gb` … `k7_gb` | int | Criterion global ratings (0–4) |
| `k1_sum_erf` … `k7_sum_erf` | int | Sum threshold met (0/1) |
| `k1_gb_erf` … `k7_gb_erf` | int | GB threshold met (0/1) |
| `k1_erf` … `k7_erf` | int | Criterion met (0/1) |
| `diag` | int | ADHD diagnosis (0/1) |
| `k1_usum` … `k7_usum` | int | Uncertainty per sum (0–100) |
| `k1_ugb` … `k7_ugb` | int | Uncertainty per GB (0–100) |
| `diag_u` | int | Uncertainty of diagnosis (0–100) |

## Usage Examples

### Batch rating (all transcripts in a directory)
```bash
python 03_llm_scoring/wri_rating_gptoss.py \
    --input-folder /path/to/transcripts \
    --output-folder ./results \
    --prompt-file /path/to/local_prompt.txt \
    --model gpt-oss-120b \
    --temperature 0.75
```

The WRI prompt is not part of this repository (copyright); see `docs/prompt.md`.

### Backends
- **gpt-oss**: LM Studio SDK (`wri_rating_gptoss.py`), local server at `http://localhost:1234`
- **Qwen3**: HuggingFace transformers (`wri_rating_qwen.py`); the script also offers `lmstudio`, `openai` (vLLM) and `ollama` backends

### Model and temperature settings (paper)
The paper evaluated:
- **gpt-oss-20b** at temperatures 0, 0.75, 1.5
- **gpt-oss-120b** at temperatures 0, 0.75, 1.5
- **Qwen-32B** (vLLM backend) at temperature 0.75
- **Qwen-235B** (vLLM backend) at temperature 0.75

Default: `gpt-oss-20b` at temperature 0.75.

## File Structure

```
wri-adhm-llm/
├── README.md
├── LICENSE                           # MIT
├── requirements.txt
├── 01_transcription/whisperx_transcribe.sh
├── 02_preprocessing/remove_timestamps.py
├── 03_llm_scoring/
│   ├── wri_rating_gptoss.py          # gpt-oss via LM Studio
│   └── wri_rating_qwen.py            # Qwen3 via HF transformers
├── 04_qa/
│   ├── add_criteria_columns.py
│   ├── check_criteria_and_diagnosis.py
│   └── check_rules.py
├── prompts/EXAMPLE_PROMPT.md         # general description (real prompt not included)
├── examples/                         # synthetic transcript, VTT and CSVs
└── docs/                             # whisperx, pipeline, data_format, prompt
```

## Limitations & Important Notes

**NOT for clinical use.** This tool is for research purposes only:
- Agreement with human raters is moderate (Cohen's κ ≈ 0.48–0.67; ICC ≈ 0.83–0.85)
- Diagnostic accuracy is 0.83 (sensitivity 0.88, specificity 0.80)
- Uncertainty estimates are exploratory and should not guide clinical decisions
- The German prompt is the validated version (not included in this repository)

**LLM limitations:**
- Hallucinations and biases inherent to LLMs apply
- Model performance may vary with transcript quality, clarity, and language
- Low-resource settings or non-standard patient speech may reduce accuracy

## Requirements

- **Python 3.9+**
- **LM Studio** running locally at `http://localhost:1234` (or adapt endpoint in code)
- See `requirements.txt` for dependencies
- A local copy of the WRI prompt (`--prompt-file`)

## Authors

**Study & Pipeline:**
- Oliver Grimm (University of Frankfurt, Dept. Psychiatry)
- Christopher Landau

**Repo & Documentation:**
- Generated with Claude Code

## Citation

If you use this tool, please cite:
```bibtex
@software{grimm2026wri,
  title={WRI-ADHD LLM Rating Tool},
  author={Grimm, Oliver and Landau, Christopher},
  year={2026},
  url={https://github.com/biosemiotiker/wri-adhd-llm}
}
```

and the underlying study (when published).

## License

MIT – see `LICENSE`

## Issues & Contributions

This is a research release. For bugs or questions, open an issue on GitHub or contact the authors directly.
