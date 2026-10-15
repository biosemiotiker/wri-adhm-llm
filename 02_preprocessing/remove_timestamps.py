#!/usr/bin/env python3
"""
remove_timestamps.py
====================

Entfernt Zeitstempel aus WebVTT-Dateien und erzeugt bereinigtes Text-Format
für LLM-Eingabe. Behält Sprecherkennzeichnung ([SPEAKER_xx]) bei, entfernt
aber Timing-Linien und BOM.

VERWENDUNG:
    python remove_timestamps.py --input examples/syn001.vtt --output examples/adhs_syn001/syn001-corrected.txt
    python remove_timestamps.py --input examples/syn001.vtt --stdout

EINGABEFORMAT (WebVTT):
    WEBVTT

    00:00.000 --> 00:05.000
    [SPEAKER_00]: Guten Tag...

    00:05.000 --> 00:10.000
    [SPEAKER_01]: Hallo, danke...

AUSGABEFORMAT (Text):
    [SPEAKER_00]: Guten Tag...
    [SPEAKER_01]: Hallo, danke...

"""

import argparse
import re
import sys
from pathlib import Path


def remove_timestamps(vtt_content: str) -> str:
    """
    Entfernt Zeitstempel aus VTT und gibt bereinigte Text zurück.

    Args:
        vtt_content: Roher VTT-Dateiinhalt

    Returns:
        Bereinigte Text mit [SPEAKER_xx]: Zeilen
    """
    lines = vtt_content.split('\n')
    output = []

    for line in lines:
        # Überspringe WEBVTT-Header und leere Zeilen
        if line.strip() in ('WEBVTT', ''):
            continue

        # Überspringe Timestamp-Linien (Format: HH:MM:SS.mmm --> HH:MM:SS.mmm oder MM:SS.mmm --> MM:SS.mmm)
        if '-->' in line:
            continue

        # Behalte [SPEAKER_xx]: Zeilen
        if '[SPEAKER_' in line:
            output.append(line.rstrip())

    # Entferne mehrfache leere Zeilen
    result = []
    prev_empty = False
    for line in output:
        if line.strip():
            result.append(line)
            prev_empty = False
        elif not prev_empty:
            result.append('')
            prev_empty = True

    # Entferne trailing empty lines
    while result and not result[-1].strip():
        result.pop()

    return '\n'.join(result)


def main():
    parser = argparse.ArgumentParser(
        description='Entfernt Zeitstempel aus WebVTT-Transkripten für LLM-Eingabe'
    )
    parser.add_argument('--input', type=str, required=True, help='Input VTT-Datei')
    parser.add_argument('--output', type=str, default=None, help='Output Text-Datei (default: stdout)')
    parser.add_argument('--stdout', action='store_true', help='Ausgabe auf stdout (Kurzform für --output -)')

    args = parser.parse_args()

    # Lese Input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"FEHLER: Input-Datei nicht gefunden: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        # Lese mit UTF-8, ignoriere BOM falls vorhanden
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except Exception as e:
        print(f"FEHLER beim Lesen der Datei: {e}", file=sys.stderr)
        sys.exit(1)

    # Verarbeite
    cleaned = remove_timestamps(content)

    # Schreibe Output
    if args.stdout or args.output == '-':
        print(cleaned)
    elif args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Schreibe ohne BOM (normaler UTF-8)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            print(f"✓ Schrieb: {output_path}", file=sys.stderr)
        except Exception as e:
            print(f"FEHLER beim Schreiben der Datei: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(cleaned)


if __name__ == '__main__':
    main()
