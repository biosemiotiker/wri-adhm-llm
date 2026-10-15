# WRI Prompt Structure

## Overview

The WRI prompt is a **one-turn, four-phase prompt** (German, validated in the study).

**The prompt text itself is not part of this repository.** The Wender-Reimherr-Interview and its item texts are copyrighted. The scoring scripts therefore load the prompt from a local file:

```bash
python 03_llm_scoring/wri_rating_gptoss.py --input-folder <transcripts> --prompt-file <local_prompt.txt>
python 03_llm_scoring/wri_rating_qwen.py   --input-folder <transcripts> --prompt-file <local_prompt.txt>
```

Files matching `prompts/wri_prompt_*.txt` are git-ignored. The prompt is available from the authors on request, subject to the rights holder's permission. A general example is in [`prompts/EXAMPLE_PROMPT.md`](../prompts/EXAMPLE_PROMPT.md).

- **Models:** gpt-oss-20b/120b (LM Studio) and Qwen3-32B/235B (HF transformers)
- **Temperatures tested:** 0, 0.75, 1.5
- The transcript is appended to the prompt as `TRANSKRIPT: <text>`.

---

## Phases

1. **Item scoring:** 28 items in 7 criteria (K1–K7), each rated 0 (absent), 1 (mild), 2 (moderate to severe) or -1 (not assessable).
2. **Criterion sums and global ratings:** sum per criterion (`SUM`, -1 excluded) and a global rating (`GB`).
3. **Criterion fulfilment and diagnosis:** per-criterion fulfilment (`ERF`) and overall diagnosis (`DIAG`) derived from sums and global ratings by fixed rules.
4. **Uncertainty:** model-reported uncertainty (`USUM`, `UGB`, `DIAG_U`).

## Output Format

The model must answer in a strict `[KEY]=VALUE` format, one key per line, for example:

```
[1.1]=<score>
...
[SUM1]=<value>
[GB1]=<value>
[ERF1]=<0 or 1>
[DIAG]=<0 or 1>
```

Python recomputes sums, `ERF` and `DIAG` deterministically from the item scores and does not rely on the model's arithmetic. Inconsistencies are reported by the QA scripts in `04_qa/`.

## Parsing

`parse_*` functions in the scoring scripts read the keys with regular expressions. Values that cannot be parsed become `NaN`; `-1` entries are counted in the QC summary printed after each run.
