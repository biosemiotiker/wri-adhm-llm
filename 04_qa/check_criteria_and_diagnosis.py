import pandas as pd
from pathlib import Path

# Schwellenwerte je Kriterium (Summenscore) – identisch zum Hauptskript
THRESHOLDS = {
    1: 5,
    2: 3,
    3: 3,
    4: 4,
    5: 3,
    6: 5,
    7: 5,
}


def check_file(input_path: str):
    input_path = Path(input_path)

    df = pd.read_csv(
        input_path,
        sep=",",
        encoding="utf-8",
        na_values=["NA"],
        index_col=False,
    )

    # Hier sammeln wir pro Problemtyp die Zeilen mit Verstoß
    issues = []

    # 1–3: Checks pro Kriterium
    for k, thr in THRESHOLDS.items():
        sum_col = f"{k}_SC"
        gb_col = f"{k}_GB"
        sum_new = f"k{k}_sum_met_human"
        gb_new = f"k{k}_gb_met_human"
        overall_col = f"k{k}_erfuellt"

        # Erwartete Flags nach Logik
        expected_sum_flag = df[sum_col].ge(thr).fillna(False).astype(int)
        expected_gb_flag = df[gb_col].ge(2).fillna(False).astype(int)
        expected_overall_flag = (
            (expected_sum_flag == 1) & (expected_gb_flag == 1)
        ).astype(int)

        # 1) Summenscore-Flag korrekt?
        mask_sum_mismatch = df[sum_new] != expected_sum_flag
        if mask_sum_mismatch.any():
            tmp = df.loc[mask_sum_mismatch, :].copy()
            tmp["issue_type"] = f"k{k}_sum_flag_mismatch"
            issues.append(tmp)

        # 2) GB-Flag korrekt?
        mask_gb_mismatch = df[gb_new] != expected_gb_flag
        if mask_gb_mismatch.any():
            tmp = df.loc[mask_gb_mismatch, :].copy()
            tmp["issue_type"] = f"k{k}_gb_flag_mismatch"
            issues.append(tmp)

        # 3) Insgesamt erfüllt korrekt?
        mask_overall_mismatch = df[overall_col] != expected_overall_flag
        if mask_overall_mismatch.any():
            tmp = df.loc[mask_overall_mismatch, :].copy()
            tmp["issue_type"] = f"k{k}_overall_flag_mismatch"
            issues.append(tmp)

    # 4: Diagnose-Regel prüfen (diagnosis_new vs. Regel)
    # Kriterien 1 und 2 erfüllt?
    crit12_ok = (df["k1_erfuellt"] == 1) & (df["k2_erfuellt"] == 1)

    # Kriterien 3-7: mindestens zwei erfüllt
    crit_3_to_7_cols = [f"k{i}_erfuellt" for i in range(3, 8)]
    crit_3_to_7_sum = df[crit_3_to_7_cols].fillna(0).astype(int).sum(axis=1)
    crit_3_to_7_ok = crit_3_to_7_sum >= 2

    expected_diagnosis = ((crit12_ok) & (crit_3_to_7_ok)).astype(int)

    mask_diag_rule_mismatch = df["diagnosis_new"] != expected_diagnosis
    if mask_diag_rule_mismatch.any():
        tmp = df.loc[mask_diag_rule_mismatch, :].copy()
        tmp["issue_type"] = "diagnosis_new_rule_mismatch"
        issues.append(tmp)

    # 5: Alte Diagnose vs. neue Diagnose vergleichen (optional, aber hilfreich)
    if "diagnose" in df.columns:
        mask_diag_old_vs_new = df["diagnose"] != df["diagnosis_new"]
        if mask_diag_old_vs_new.any():
            tmp = df.loc[mask_diag_old_vs_new, :].copy()
            tmp["issue_type"] = "diagnose_old_vs_new_differs"
            issues.append(tmp)

    # Zusammenführen aller Issues
    if issues:
        all_issues = pd.concat(issues, ignore_index=True)
    else:
        all_issues = pd.DataFrame(columns=df.columns.tolist() + ["issue_type"])

    # Kurze Zusammenfassung in der Konsole
    print("=== Validation Summary ===")
    if all_issues.empty:
        print("Keine Regelverletzungen gefunden.")
    else:
        # pro issue_type Anzahl Zeilen
        counts = all_issues["issue_type"].value_counts()
        for itype, count in counts.items():
            print(f"{itype}: {count} Zeilen")

    # Detailbericht als CSV speichern
    output_path = input_path.with_name(input_path.stem + "_validation_issues.csv")
    all_issues.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nDetailbericht gespeichert unter: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validiert Kriterien-Flags und Diagnose-Regeln."
    )
    parser.add_argument(
        "input_path",
        help=(
            "Pfad zur CSV-Datei mit hinzugefügten Kriterien- und Diagnosespalten "
            "(z.B. demo_data_added_columns.csv)"
        ),
    )
    args = parser.parse_args()

    check_file(args.input_path)

