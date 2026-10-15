# WhisperX Transcription Pipeline

This guide covers the **prerequisite step** before WRI rating: converting audio files to speaker-diarized transcripts.

## Overview

**WhisperX** is a speech-to-text pipeline with automatic speaker diarization, producing WebVTT transcripts with `[SPEAKER_00]` (interviewer) and `[SPEAKER_01]` (patient) labels.

```
Audio file (.wav, .mp3, .m4a)
    ↓
WhisperX (speech-to-text + diarization)
    ↓
output.vtt (timestamped, diarized)
    ↓
remove_timestamps.py
    ↓
output-corrected.txt (clean format for wri_rating_gptoss.py)
```

## Installation

```bash
pip install openai-whisper transformers torch torchaudio pyannote.audio
```

**PyAnnote speaker diarization requires HuggingFace login:**
```bash
huggingface-cli login
# Paste your HuggingFace token (get from https://huggingface.co/settings/tokens)
```

## Usage

### Basic Command (German)
```bash
whisperx audio.wav \
    --model large-v3 \
    --language de \
    --diarize_model pyannote \
    --output_format vtt \
    --output_dir ./vtt_output
```

### Parameters
- `--model large-v3`: Multilingual model (large = best accuracy)
- `--language de`: German (forces language, improves accuracy)
- `--diarize_model pyannote`: Speaker diarization (2 speakers assumed)
- `--output_format vtt`: WebVTT format with timestamps
- `--output_dir`: Output directory

### Batch Processing
```bash
for file in *.wav; do
    whisperx "$file" --model large-v3 --language de --diarize_model pyannote --output_format vtt --output_dir ./vtt_output
done
```

## Output Format (VTT)

**Example output file:** `audio.vtt`

```
WEBVTT

00:00.000 --> 00:05.000
[SPEAKER_00]: Guten Tag. Ich bin heute hier, um mit dir einige Fragen zu stellen.

00:05.000 --> 00:07.000
[SPEAKER_01]: Ja, gerne.

00:07.000 --> 00:10.000
[SPEAKER_00]: Erste Frage: Hast du Schwierigkeiten, dich zu konzentrieren?

00:10.000 --> 00:15.000
[SPEAKER_01]: Ja, mir fällt es sehr schwer.
```

**Speaker mapping (fixed):**
- `[SPEAKER_00]` = Interviewer/clinician
- `[SPEAKER_01]` = Patient (rated)

## Converting VTT to TXT

The WRI rating script expects plain-text format (no timestamps). Use the included converter:

```bash
python 02_preprocessing/remove_timestamps.py input.vtt output-corrected.txt
```

**Input (VTT with timestamps):**
```
WEBVTT

00:00.000 --> 00:05.000
[SPEAKER_00]: Text...

00:05.000 --> 00:10.000
[SPEAKER_01]: Text...
```

**Output (clean TXT):**
```
[SPEAKER_00]: Text...
[SPEAKER_01]: Text...
```

### Manual Correction

After WhisperX, manually review and correct transcripts:

```bash
# Edit in your favorite text editor
nano output-corrected.txt

# Check format
grep "^\[SPEAKER" output-corrected.txt | head -5
```

**Expected format:**
- One line per speaker utterance
- Format: `[SPEAKER_00]: text` or `[SPEAKER_01]: text`
- No timestamps
- No empty lines between utterances (one per line)

## Quality Checks

### Speaker Consistency
Ensure interviewer and patient roles are correct:
```bash
grep "\[SPEAKER_00\]" output-corrected.txt | wc -l  # Should be ~50-100
grep "\[SPEAKER_01\]" output-corrected.txt | wc -l  # Should be ~50-100
```

### Content Verification
- Check for obvious transcription errors (common: homophones, technical terms)
- Verify speaker switches are logical (no consecutive [SPEAKER_00] or [SPEAKER_01] utterances without alternation)
- Manual review recommended for low-confidence segments

## Troubleshooting

### WhisperX says no GPU detected
```bash
# Install CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Diarization model won't download
- Ensure `huggingface-cli login` was run successfully
- Check internet connection
- PyAnnote model is ~300 MB, may take 1–2 min on first run

### Wrong speaker assignments
- WhisperX diarization assumes 2 speakers
- If 3rd party (observer, etc.) is present, manual post-correction required
- Re-diarization: re-run with adjusted model (if available)

### Timestamps in output after remove_timestamps.py
- Check input file format (should be valid VTT)
- Ensure the script is in `02_preprocessing/`
- Try: `python 02_preprocessing/remove_timestamps.py --help`

## Performance Notes

- **Model time:** large-v3 takes ~30–60 min per hour of audio (GPU-dependent)
- **Diarization time:** ~10–20 min per hour
- **Total:** ~1–1.5 hours per interview (45–60 min audio)
- **GPU:** Strongly recommended (NVIDIA CUDA 11.8+, 6–8GB VRAM minimum)

## Citation

**WhisperX:** Moreno, P., Aksen, E., Hou, S., & Schuster, M. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv preprint arXiv:2212.04356.

**PyAnnote (diarization):** Bredin, H., Yin, R., Coria, J. M., et al. (2021). *pyannote.audio: neural building blocks for speaker diarization*. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 7107-7111). IEEE.

## Next Steps

After WhisperX + timestamp removal:

```bash
# 1. Manually review and correct corrected.txt
# 2. Place in input directory structure:
#    adhs_<ID>/
#      └── <ID>-corrected.txt

# 3. Run WRI rating
python 03_llm_scoring/wri_rating_gptoss.py --prompt-file <lokaler_prompt.txt> --input-folder ./adhs_* --model "gpt-oss-20b"

# 4. Process results
python 04_qa/add_criteria_columns.py wri_results_table.csv
```

---

## Related Documentation

- `pipeline.md` – Full WRI pipeline overview
- `data_format.md` – Input/output format specifications
- `README.md` – Quick start and usage
