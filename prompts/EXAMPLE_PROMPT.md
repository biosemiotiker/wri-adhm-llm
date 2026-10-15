# Example WRI Scoring Prompt (Deutsch)

Dieses Dokument zeigt die **Struktur und das Format** eines WRI-Scoring-Prompts, wie er in `03_llm_scoring/wri_rating_gptoss.py` und `wri_rating_qwen.py` verwendet wird.

## Prompt-Struktur

Der Prompt folgt einem **4-Phasen-Format**:

### Phase 1: Symptom-Items bewerten (0/1/2/-1)
**Eingabe:** Transkript eines Interviews  
**Aufgabe:** Bewerten Sie jedes der 28 WRI-Items

Beispiel für die WRI-Items (deutsche Domänen):
- **K1** (Aufmerksamkeit): Konzentrationsprobleme, Aufmerksamkeitsspanne, Ablenkbarkeit
- **K2** (Hyperaktivität): Innere Unruhe, überaktiv im Geist, motorische Unruhe
- **K3** (Temperament): Hitzkopfigkeit, reizbar, Wutausbrüche
- **K4** (Affektlabilität): Stimmungsschwankungen, emotionale Überreaktion
- **K5** (Desorganisation): Schusseligkeit, chaotische Verhältnisse, Organisationsprobleme
- **K6** (Stresstoleranz): Überfordert bei Druck, ungeduldig, niedrige Frustrationstoleranz
- **K7** (Impulsivität): Übereilte Entscheidungen, vorschnelle Worte, Ungeduld

### Phase 2: Kriterium-Summen (SUM) und Globalurteile (GB) berechnen
- **SUM**: Summe der Item-Punkte pro Kriterium
- **GB**: Globales Urteil (0–3 Skala) basierend auf Gesamteindruck des Kriteriums

### Phase 3: Erfüllung entscheiden
- Prüfe, ob SUM ≥ Schwellwert (z. B. SUM ≥ 3)
- Prüfe, ob GB ≥ Schwellwert (z. B. GB ≥ 2)
- **Kriterium erfüllt**: beide Bedingungen wahr
- **ADHS-Diagnose**: alle 7 Kriterien erfüllt

### Phase 4: Unsicherheit quantifizieren (optional)
Gibt Unsicherheitsscores für jedes Item und jedes Kriterium an.

## Output-Format

Alle Ausgaben folgen dem Schlüssel-Wert-Format:

```
[1.1]=2
[1.2]=1
...
[K1_SUM]=12
[K1_GB]=2
[K1_USUM]=5
[K1_UGB]=2
...
[DIAG_ERFUELLT]=1
[DIAG_USUM]=2
```

**Format-Regeln:**
- `[KEY]=VALUE` (eckige Klammern, keine runden)
- Großbuchstaben für Schlüssel
- Kein Leerzeichen im Schlüssel
- Ein Wert pro Zeile

## Implementierung

Die echten Prompts sind in den Scoring-Skripten als `WRI_PROMPT`-String implementiert:
- `03_llm_scoring/wri_rating_gptoss.py` (Zeilen ~80–320)
- `03_llm_scoring/wri_rating_qwen.py` (Zeilen ~82–320)

Die 235B-Variante hat zusätzliche Instruktionen zur Differentialdiagnose und Thinking-Mode.

## Urheberrecht

Das Wender-Reimherr-Interview ist ein publiziertes Messinstrument der Autoren Wender (1995) und in der deutschen Version von Retz-Junginger und Reimherr. Die Scoring-Regeln und Item-Definitionen sind geschützt. Dieser Beispiel-Prompt zeigt nur die **Struktur** und nutzt **allgemeine Beschreibungen** der WRI-Domänen. Für exakte Wortlaute siehe:

- Wender, P. H. (1995). "Attention-Deficit Hyperactivity Disorder in Adults". Oxford University Press.
- Deutsche Version und Scoring-Handbuch des Originalpublikums.

Die tatsächlich verwendeten Prompts im Publikationszusammenhang sollten gegen das offizielle Scoring-Manual validiert werden.
