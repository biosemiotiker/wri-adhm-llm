#!/bin/bash

#
# whisperx_transcribe.sh
# ======================
#
# Transkribiert Audio-Dateien mit WhisperX (Whisper + Pyannote Diarization)
# und speichert das Ergebnis als WebVTT-Dateien.
#
# VERWENDUNG:
#    ./whisperx_transcribe.sh --input-dir /path/to/audio --output-dir /path/to/vtt
#    ./whisperx_transcribe.sh --input-dir audio/ --output-dir transcripts/
#
# ANFORDERUNGEN:
#    - WhisperX installiert: pip install whisperx
#    - HuggingFace Token (für Pyannote Diarization):
#      Setze HF_TOKEN=hf_... als Umgebungsvariable
#      oder huggingface-cli login
#    - CUDA GPU (optional, aber stark empfohlen)
#
# PARAMETER:
#    --input-dir     Ordner mit WAV/MP3/M4A-Dateien (erforderlich)
#    --output-dir    Zielordner für VTT-Dateien (erforderlich)
#    --model         Whisper Modell (default: large)
#    --language      ISO 639-1 Code (default: de für Deutsch)
#    --device        cpu oder cuda (default: cuda, fallback auf cpu)
#
# OUTPUT:
#    Pro Eingabedatei audio_id.{wav,mp3,m4a} wird erzeugt:
#        output_dir/audio_id.vtt  (WebVTT mit [SPEAKER_00], [SPEAKER_01])
#

set -e

# Defaults
MODEL="large"
LANGUAGE="de"
DEVICE="cuda"
INPUT_DIR=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unbekanntes Argument: $1"
            exit 1
            ;;
    esac
done

# Validierung
if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "FEHLER: --input-dir und --output-dir sind erforderlich."
    echo ""
    echo "VERWENDUNG:"
    echo "  $0 --input-dir /path/to/audio --output-dir /path/to/vtt"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "FEHLER: Input-Verzeichnis existiert nicht: $INPUT_DIR"
    exit 1
fi

# Output-Verzeichnis erstellen
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "WhisperX Transkription"
echo "=================================================="
echo "Input-Verzeichnis:  $INPUT_DIR"
echo "Output-Verzeichnis: $OUTPUT_DIR"
echo "Modell:             $MODEL"
echo "Sprache:            $LANGUAGE"
echo "Device:             $DEVICE"
echo ""

# Prüfe ob GPU verfügbar ist
if [[ "$DEVICE" == "cuda" ]]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNUNG: nvidia-smi nicht gefunden. Fallback auf CPU."
        DEVICE="cpu"
    fi
fi

# Zähle Audio-Dateien
AUDIO_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.m4a" \) | wc -l)
if [[ $AUDIO_COUNT -eq 0 ]]; then
    echo "WARNUNG: Keine Audio-Dateien gefunden in $INPUT_DIR"
    exit 0
fi

echo "Gefundene Audio-Dateien: $AUDIO_COUNT"
echo ""

# Verarbeite jede Audio-Datei
for audio_file in "$INPUT_DIR"/*.{wav,mp3,m4a}; do
    # Überspringe wenn Dateierweiterung nicht existiert
    [[ -e "$audio_file" ]] || continue

    filename=$(basename "$audio_file")
    basename_no_ext="${filename%.*}"
    output_file="$OUTPUT_DIR/$basename_no_ext.vtt"

    echo "Verarbeite: $filename"

    # WhisperX Aufruf mit Diarisierung (max 2 Sprecher)
    python -m whisperx \
        "$audio_file" \
        --model "$MODEL" \
        --language "$LANGUAGE" \
        --diarize \
        --max_speakers 2 \
        --output_format vtt \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --compute_type float32 \
        2>&1 | grep -v "^$" || true

    echo "  ✓ $basename_no_ext.vtt"
done

echo ""
echo "=================================================="
echo "Transkription abgeschlossen."
echo "Ergebnisse in: $OUTPUT_DIR"
echo "=================================================="
