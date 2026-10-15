import pandas as pd
from pathlib import Path

# Schwellenwerte je Kriterium (Summenscore)
THRESHOLDS = {
    1: 5,
    2: 3,
    3: 3,
    4: 4,
    5: 3,
    6: 5,
    7: 5,
}


def add_human_columns(input_path: str):
    input_path = Path(input_path)

    # CSV einlesen
    df = pd.read_csv(
        input_path,
        sep=",",
        encoding="utf-8",
        na_values=["NA"],
        index_col=False,
    )  # [web:16]

    # 1. Kriterien-Spalten (Sum, GB, insgesamt erfüllt) berechnen/aktualisieren
    for k, thr in THRESHOLDS.items():
        sum_col = f"{k}_SC"
        gb_col = f"{k}_GB"
        sum_new = f"k{k}_sum_met_human"
        gb_new = f"k{k}_gb_met_human"
        overall_col = f"k{k}_erfuellt"

        # Falls "human"-Spalten schon existieren (z.B. erneuter Lauf), vorher entfernen
        for col in [sum_new, gb_new]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Summenscore-Bedingung: >= threshold, NA -> 0
        sum_flag = df[sum_col].ge(thr).fillna(False).astype(int)  # [web:23]

        # Globale Beurteilung: >= 2, NA -> 0
        gb_flag = df[gb_col].ge(2).fillna(False).astype(int)  # [web:23]

        # "insgesamt erfüllt": nur wenn beide Bedingungen erfüllt (1) sind
        overall_flag = ((sum_flag == 1) & (gb_flag == 1)).astype(int)  # [web:23]

        # Position des bestehenden "insgesamt erfüllt"-Flags finden
        insert_pos = df.columns.get_loc(overall_col)  # [web:13]

        # Neue Spalten direkt davor einfügen:
        # zuerst Summenscore, dann globale Beurteilung
        df.insert(insert_pos, sum_new, sum_flag)
        df.insert(insert_pos + 1, gb_new, gb_flag)

        # Bestehende "insgesamt erfüllt"-Spalte neu setzen
        df[overall_col] = overall_flag

    # 2. Diagnose neu berechnen: diagnosis_new
    # Annahme: bestehende Spalte heißt "diagnose" und ist die letzte Spalte.
    # Wenn der Name anders ist, hier anpassen.
    diagnosis_col = "diagnose"

    # Kriterien 1 und 2 müssen erfüllt sein
    crit12_ok = (df["k1_erfuellt"] == 1) & (df["k2_erfuellt"] == 1)

    # Kriterien 3-7: mindestens zwei erfüllt
    crit_3_to_7_cols = [f"k{i}_erfuellt" for i in range(3, 8)]
    # Summe der erfüllten Kriterien 3-7 pro Zeile
    crit_3_to_7_sum = df[crit_3_to_7_cols].fillna(0).astype(int).sum(axis=1)  # [web:62][web:66]
    crit_3_to_7_ok = crit_3_to_7_sum >= 2

    diagnosis_new = ((crit12_ok) & (crit_3_to_7_ok)).astype(int)

    # diagnosis_new direkt hinter "diagnose" einfügen
    diag_pos = df.columns.get_loc(diagnosis_col)
    df.insert(diag_pos + 1, "diagnosis_new", diagnosis_new)  # [web:13][web:9]

    # Ausgabedatei mit Suffix _added_columns
    output_path = input_path.with_name(input_path.stem + "_added_columns.csv")

    df.to_csv(output_path, index=False, encoding="utf-8")  # [web:30]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fügt Summenscore-, GB- und aktualisierte 'insgesamt erfüllt'-Spalten "
            "für 7 Kriterien hinzu und berechnet diagnosis_new."
        )
    )
    parser.add_argument(
        "input_path",
        help="Pfad zur Eingabe-CSV (z.B. examples/demo_data_synthetic.csv oder wri_results_table.csv)",
    )
    args = parser.parse_args()

    add_human_columns(args.input_path)

