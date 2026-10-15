import os
import re
import glob
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import pandas as pd
from tqdm import tqdm
import lmstudio as lms

# ===================================================================
# Defaults
# ===================================================================
DEFAULT_MODEL = "gpt-oss-20b"
DEFAULT_TEMPERATURE = 0.75
DEFAULT_TOP_P = 1.0
DEFAULT_CONTEXT_LENGTH = 40000

# ===================================================================
# NEUER EIN-TURN-PROMPT
# ===================================================================

# WICHTIG:
# Kopiere hier den vollständigen Prompt aus Prompt.docx hinein.
# Achte darauf, dass alle Backslashes und Sonderzeichen korrekt sind.
WRI_PROMPT = ""  # wird zur Laufzeit via --prompt-file geladen (Prompt aus Urheberrechtsgründen nicht im Repo)


def load_prompt(path):
    """Liest den WRI-Prompt aus einer lokalen Datei."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"Prompt-Datei nicht gefunden: {p}\n"
            "Der WRI-Prompt ist aus Urheberrechtsgründen nicht im Repository enthalten. "
            "Siehe prompts/EXAMPLE_PROMPT.md für Struktur und Ausgabeformat."
        )
    return p.read_text(encoding="utf-8")

# ===================================================================
# HILFSFUNKTION: ROBUSTES DATEILESEN
# ===================================================================

def read_text_file_robust(path: str) -> Tuple[Optional[str], Optional[str]]:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read(), enc
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Fehler beim Lesen von {path} mit {enc}: {e}")
            return None, None
    # Fallback mit Ersatzzeichen
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            print(f"Warnung: {path} mit Ersatzzeichen (utf-8, errors='replace') gelesen.")
            return f.read(), "utf-8-replace"
    except Exception as e:
        print(f"Fehler beim Fallback-Lesen von {path}: {e}")
        return None, None

# ===================================================================
# TRANSKRIPT-SUCHE
# ===================================================================

def find_transcripts_production(input_folder: str) -> Dict[str, str]:
    transcript_paths: Dict[str, str] = {}
    for entry in sorted(os.listdir(input_folder)):
        folder_path = os.path.join(input_folder, entry)
        if os.path.isdir(folder_path) and entry.lower().startswith("adhs"):
            id_match = re.match(r"^adhs_(.+?)(?:-corrected)?$", entry, re.IGNORECASE)
            patient_id = id_match.group(1) if id_match else entry
            matches = glob.glob(os.path.join(folder_path, "*corrected.txt"))
            if matches:
                transcript_paths[patient_id] = matches[0]
            else:
                print(f"Warnung: Kein 'corrected.txt' in: {folder_path}")
    return transcript_paths


def find_transcripts_test(test_folder: str) -> Dict[str, str]:
    transcript_paths: Dict[str, str] = {}
    matches = glob.glob(os.path.join(test_folder, "*-corrected.txt"))
    for filepath in sorted(matches):
        filename = os.path.basename(filepath)
        patient_id = re.sub(r"-corrected\.txt$", "", filename, flags=re.IGNORECASE)
        transcript_paths[patient_id] = filepath
    return transcript_paths
# ===================================================================
# PARSER FÜR EIN-TURN-OUTPUT → CSV
# ===================================================================

ITEM_KEYS: List[str] = [
    "1.1","1.2","1.3","1.4","1.5",
    "2.1","2.2","2.3",
    "3.1","3.2","3.3",
    "4.1","4.2","4.3","4.4",
    "5.1","5.2","5.3",
    "6.1","6.2","6.3","6.4","6.5",
    "7.1","7.2","7.3","7.4","7.5",
]

K_SUM_KEYS = [f"K{i}_SUM" for i in range(1, 8)]
K_GB_KEYS  = [f"K{i}_GB"  for i in range(1, 8)]

K_SUM_ERF_KEYS = [f"K{i}_SUM_ERF" for i in range(1, 8)]
K_GB_ERF_KEYS  = [f"K{i}_GB_ERF"  for i in range(1, 8)]
K_ERF_KEYS     = [f"K{i}_ERF"     for i in range(1, 8)]

K_USUM_KEYS = [f"K{i}_USUM" for i in range(1, 8)]
K_UGB_KEYS  = [f"K{i}_UGB"  for i in range(1, 8)]

ALL_KEYS_IN_ORDER: List[str] = (
    ITEM_KEYS +
    K_SUM_KEYS + K_GB_KEYS +
    K_SUM_ERF_KEYS + K_GB_ERF_KEYS + K_ERF_KEYS +
    ["DIAG"] +
    K_USUM_KEYS + K_UGB_KEYS +
    ["DIAG_U"]
)

def parse_wri_output_to_dict(text: str) -> Dict[str, Optional[int]]:
    values: Dict[str, Optional[int]] = {}
    for key in ALL_KEYS_IN_ORDER:
        pattern = r"\[" + re.escape(key) + r"\]\s*=\s*(-?\d+)"
        match = re.search(pattern, text)
        if match:
            try:
                values[key] = int(match.group(1))
            except ValueError:
                values[key] = None
        else:
            values[key] = None
    return values

def build_results_csv(output_folder: str) -> None:
    print("\n--- Erstelle wri_results_table.csv (Ein-Turn-Format) ---")

    wri_files = sorted(
        f for f in os.listdir(output_folder)
        if f.endswith(".txt")  # es gibt keine _confidence mehr
    )
    if not wri_files:
        print("Keine WRI-Output-Dateien gefunden.")
        return

    all_rows: List[Dict[str, Optional[int]]] = []

    for filename in wri_files:
        patient_id = os.path.splitext(filename)[0]
        wri_path = os.path.join(output_folder, filename)

        try:
            with open(wri_path, "r", encoding="utf-8") as f:
                wri_content = f.read()
        except UnicodeDecodeError:
            wri_content, enc_used = read_text_file_robust(wri_path)
            if wri_content is None:
                print(f"Fehler beim Lesen von {wri_path}, Datei wird übersprungen.")
                continue
        except Exception as e:
            print(f"Fehler beim Lesen von {wri_path}: {e}")
            continue

        parsed = parse_wri_output_to_dict(wri_content)
        row: Dict[str, Optional[int]] = {"patient_id": patient_id}

        # Items
        for key in ITEM_KEYS:
            row[f"item_{key.replace('.', '_')}"] = parsed[key]

        # Phase 2
        for k in range(1, 8):
            row[f"k{k}_sum"] = parsed[f"K{k}_SUM"]
            row[f"k{k}_gb"]  = parsed[f"K{k}_GB"]

        # Phase 3
        for k in range(1, 8):
            row[f"k{k}_sum_erf"] = parsed[f"K{k}_SUM_ERF"]
            row[f"k{k}_gb_erf"]  = parsed[f"K{k}_GB_ERF"]
            row[f"k{k}_erf"]     = parsed[f"K{k}_ERF"]
        row["diag"] = parsed["DIAG"]

        # Phase 4
        for k in range(1, 8):
            row[f"k{k}_usum"] = parsed[f"K{k}_USUM"]
            row[f"k{k}_ugb"]  = parsed[f"K{k}_UGB"]
        row["diag_u"] = parsed["DIAG_U"]

        all_rows.append(row)

    if not all_rows:
        print("Keine Daten für CSV gefunden.")
        return

    df = pd.DataFrame(all_rows).set_index("patient_id")

    csv_path = os.path.join(output_folder, "wri_results_table.csv")
    df.to_csv(csv_path)
    print(f"CSV gespeichert: {csv_path}")
    print(f"Dimensionen: {df.shape[0]} Probanden x {df.shape[1]} Spalten")

    # Qualitätskontrolle: NaN-Counts und -1-Counts
    nan_count = df.isna().sum().sum()
    minus_one_count = (df == -1).sum().sum()
    print(f"Qualitätskontrolle: {nan_count} nicht geparste Werte (NaN), "
          f"{minus_one_count} Einträge mit -1 (nicht beurteilbar oder laut Modell).")

# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Modus:         {'TEST' if TEST_MODE else 'PRODUKTION'}")
    print(f"Output-Ordner: {OUTPUT_FOLDER}")

    if TEST_MODE:
        transcript_paths = find_transcripts_test(TEST_FOLDER)
    else:
        transcript_paths = find_transcripts_production(INPUT_FOLDER)

    if not transcript_paths:
        source = TEST_FOLDER if TEST_MODE else INPUT_FOLDER
        print(f"Keine Transkripte gefunden in: {source}")
        return

    print(f"\n--- {len(transcript_paths)} Transkripte gefunden ---")
    print(f"    Modell: {MODEL_NAME} | Temp: {TEMPERATURE} | Top-p: {TOP_P} | Top-k: 0")

    _config = {"temperature": TEMPERATURE, "topP": TOP_P, "topK": 0}

    with lms.Client() as client:
        model = client.llm.model(MODEL_NAME, config={"contextLength": CONTEXT_LENGTH})

        for patient_id, filepath in tqdm(transcript_paths.items(),
                                         desc="Verarbeite Transkripte"):
            transcript_content, enc_used = read_text_file_robust(filepath)
            if transcript_content is None:
                print(f"Überspringe {filepath} wegen Lesefehler.")
                continue

            chat = lms.Chat()
            chat.add_user_message(
                f"{WRI_PROMPT}\n\nTRANSKRIPT:\n\n{transcript_content}"
            )

            try:
                response = model.respond(chat, config=_config)
            except Exception as e:
                print(f"Fehler beim Modellaufruf für {patient_id}: {e}")
                continue

            raw_content = (
                response.content if hasattr(response, "content") else str(response)
            )
            # LM Studio-spezifische Artefakte entfernen, wie im alten Skript
            clean_content = re.sub(
                r"<\|channel\|>.*?<\|end\|>", "", raw_content, flags=re.DOTALL
            )
            clean_content = re.sub(r"<\|[^|]+\|>", "", clean_content).strip()

            out_wri = os.path.join(OUTPUT_FOLDER, f"{patient_id}.txt")
            try:
                with open(out_wri, "w", encoding="utf-8") as f:
                    f.write(clean_content)
            except Exception as e:
                print(f"Fehler beim Schreiben von {out_wri}: {e}")

    print(f"\nFertig! {len(transcript_paths)} Auswertungen gespeichert in: {OUTPUT_FOLDER}")

    # CSV nach dem Lauf erstellen
    build_results_csv(OUTPUT_FOLDER)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="WRI-Auswertung mit gpt-oss-20b via LM Studio"
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Ordner mit Transkript-Dateien (z.B. data/ oder examples/)"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="results/",
        help="Output-Ordner für Ergebnisse (default: results/)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LM Studio Modellname (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature für Sampling (default: {DEFAULT_TEMPERATURE})"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Top-p für Sampling (default: {DEFAULT_TOP_P})"
    )
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Lokale Datei mit dem WRI-Prompt (nicht im Repo enthalten, siehe prompts/EXAMPLE_PROMPT.md)"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help=f"Context length in Tokens (default: {DEFAULT_CONTEXT_LENGTH})"
    )

    args = parser.parse_args()

    # Setze globale Variablen aus Argparse
    WRI_PROMPT = load_prompt(args.prompt_file)
    INPUT_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder
    MODEL_NAME = args.model
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    CONTEXT_LENGTH = args.context_length
    TEST_FOLDER = None  # Not used in CLI mode
    TEST_MODE = False

    main()
