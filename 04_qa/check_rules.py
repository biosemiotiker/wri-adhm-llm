#!/usr/bin/env python3
import csv
from pathlib import Path

# >>> HIER anpassen: Pfad zur Eingabe-CSV <<<
# Beispiel: wri_results_table.csv aus dem LLM-Scoring-Lauf
CSV_PATH = Path("./wri_results_table.csv")

# Ausgabe-CSV im gleichen Ordner, anderer Name
folder_name = CSV_PATH.parent.name
OUTPUT_PATH = CSV_PATH.parent / f"error_report_{folder_name}.csv"


def col_letter_to_index(col: str) -> int:
    """
    Wandelt eine Excel-Spaltenangabe (z.B. 'A', 'Z', 'AA', 'AH') in einen 0-basierten Index um.
    A -> 0, B -> 1, ..., Z -> 25, AA -> 26, AB -> 27, ...
    """
    col = col.strip().upper()
    result = 0
    for ch in col:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Ungültige Spaltenangabe: {col}")
        result = result * 26 + (ord(ch) - ord("A") + 1)
    return result - 1  # 0-basiert


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int_0_1(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        v = int(float(text))
    except ValueError:
        return None
    return v if v in (0, 1) else None


# Schwellwert-Regeln: Quelle (Spaltenbuchstabe), Schwellwert, Ziel (Spaltenbuchstabe)
threshold_rules = [
    {"src": "AD", "threshold": 5, "dst": "AR"},
    {"src": "AE", "threshold": 2, "dst": "AS"},
    {"src": "AF", "threshold": 3, "dst": "AU"},
    {"src": "AG", "threshold": 2, "dst": "AV"},
    {"src": "AH", "threshold": 3, "dst": "AX"},
    {"src": "AI", "threshold": 2, "dst": "AY"},
    {"src": "AJ", "threshold": 4, "dst": "BA"},
    {"src": "AK", "threshold": 2, "dst": "BB"},
    {"src": "AL", "threshold": 3, "dst": "BD"},
    {"src": "AM", "threshold": 2, "dst": "BE"},
    {"src": "AN", "threshold": 5, "dst": "BG"},
    {"src": "AO", "threshold": 2, "dst": "BH"},
    {"src": "AP", "threshold": 5, "dst": "BJ"},
    {"src": "AQ", "threshold": 2, "dst": "BK"},
]

# Kombinations-Regeln: L, M, N (alle als Spaltenbuchstaben)
pair_rules = [
    {"l": "AR", "m": "AS", "n": "AT"},
    {"l": "AU", "m": "AV", "n": "AW"},
    {"l": "AX", "m": "AY", "n": "AZ"},
    {"l": "BA", "m": "BB", "n": "BC"},
    {"l": "BD", "m": "BE", "n": "BF"},
    {"l": "BG", "m": "BH", "n": "BI"},
    {"l": "BJ", "m": "BK", "n": "BL"},
]


def check_file(csv_path: Path, output_path: Path):
    unrecht_1 = 0
    unrecht_0 = 0
    violations = []  # (zeilennummer, spaltenbuchstabe)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")

        # Zeile 1: Kopfzeile überspringen, aber Spaltenanzahl prüfen
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV-Datei ist leer.")

        # Datei-Zeilennummer: 2 = erste Datenzeile
        line_number = 2

        for row in reader:
            if line_number > 164:
                break

            # Sicherheitscheck: leer Zeilen o.ä.
            if not row:
                line_number += 1
                continue

            # 1) Schwellwert-Regeln
            for rule in threshold_rules:
                try:
                    src_idx = col_letter_to_index(rule["src"])
                    dst_idx = col_letter_to_index(rule["dst"])
                except ValueError:
                    continue  # ungültige Spaltenangabe -> ignorieren

                if src_idx >= len(row) or dst_idx >= len(row):
                    continue  # Zeile zu kurz -> ignorieren

                src_val = parse_float(row[src_idx])
                dst_val = parse_int_0_1(row[dst_idx])

                if src_val is None or dst_val is None:
                    continue  # ignorieren

                expected = 1 if src_val >= rule["threshold"] else 0

                if dst_val != expected:
                    if dst_val == 1 and expected == 0:
                        unrecht_1 += 1
                    elif dst_val == 0 and expected == 1:
                        unrecht_0 += 1
                    violations.append((line_number, rule["dst"]))

            # 2) Kombinations-Regeln (L, M -> N)
            for rule in pair_rules:
                try:
                    l_idx = col_letter_to_index(rule["l"])
                    m_idx = col_letter_to_index(rule["m"])
                    n_idx = col_letter_to_index(rule["n"])
                except ValueError:
                    continue

                if l_idx >= len(row) or m_idx >= len(row) or n_idx >= len(row):
                    continue

                l_val = parse_int_0_1(row[l_idx])
                m_val = parse_int_0_1(row[m_idx])
                n_val = parse_int_0_1(row[n_idx])

                if l_val is None or m_val is None or n_val is None:
                    continue  # ignorieren

                expected_n = 1 if (l_val == 1 and m_val == 1) else 0

                if n_val != expected_n:
                    if n_val == 1 and expected_n == 0:
                        # Unrecht 1 (z.B. (0,0,1), (1,0,1), (0,1,1))
                        unrecht_1 += 1
                    elif n_val == 0 and expected_n == 1:
                        # Unrecht 0 (z.B. (1,1,0))
                        unrecht_0 += 1
                    violations.append((line_number, rule["n"]))

            line_number += 1

    # Ausgabe-CSV schreiben
    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f, delimiter=",")
        writer.writerow(["Unrecht 1", "Unrecht 0"])
        writer.writerow([unrecht_1, unrecht_0])
        for line, col in violations:
            writer.writerow([line, col])


if __name__ == "__main__":
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {CSV_PATH}")
    check_file(CSV_PATH, OUTPUT_PATH)
    print(f"Fertig. Ergebnis geschrieben nach: {OUTPUT_PATH}")

