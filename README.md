# Bartlett und Welch: Mitteln beruhigt das Spektrum, kostet aber Schärfe – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-bartlett-welch-demo.streamlit.app/)** -->

Dritte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf den Demos
zum **Periodogramm** und zu **Fenstern**. Ein simuliertes Signal (Hintergrund mit bekannter Dichte, bis zu zwei Linien) wird in Segmente geschnitten, gefenstert und gemittelt;
Streuung, Auflösung, Leckage und die Wirkung der Überlappung werden gegen die Wahrheit gemessen.

**Namensfalle:** Mit „Baum-Welch“ (`baum-welch-demo`, Hidden-Markov-Modelle) und „Spectral Clustering“ hat diese Demo nichts zu tun.

## Warum dieses Problem

Ein einzelnes Periodogramm streut so stark wie sein Wert, egal wie lang die Aufzeichnung ist (Variationskoeffizient 1 je innerem Bin bei weißem Gauß-Rauschen). Mitteln über Segmente beruhigt die Kurve, aber jedes Segment
ist kürzer und löst weniger fein auf. Das Verfahren tauscht Schärfe gegen Ruhe; wie günstig der Tausch ist, hängt vom Fenster je Segment und von der Überlappung ab.

## Modell

Hintergrund weiß oder mit AR-Pol („Resonanz“: ein Gipfel bei 200 Hz; „Steil“: 57 dB Spannweite), Varianz auf 1 normiert, plus null bis zwei Linien. Der Schätzer ist das arithmetische
Mittel der gefensterten Segment-Periodogramme (Dichte je Hz, zweiseitig); sein Erwartungswert ist exakt berechenbar (Fensterautokorrelation, Linien segmentweise) und hängt nicht von der
Überlappung ab. Die Streuung folgt für weißes Rauschen aus der Segmentkorrelation (Formel im 📐-Expander). Die Abtastrate ist fest 1000 Hz.

## Methodik

Vier Fenster (Rechteck, Hann, Hamming, Blackman), Überlappung 0 bis 87,5 %, Segmentlängen 32 bis n, Aufzeichnungen 1024 bis 16384. Experimente über feste Zufallsströme (Seeds ab 300000),
jede Zahl ist ein Mittel über 100 bis 600 Läufe. Das 95-%-Band ist ein χ²-Band mit 2·K<sub>eff</sub> Freiheitsgraden (K<sub>eff</sub> = äquivalente Zahl unabhängiger Mittel).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py` (Anteile bei 100 bis 600 Läufen: Standardfehler 2 bis 5 Prozentpunkte).

| Frage | Befund |
|---|---|
| Wie sinkt die Streuung bei Bartlett (Rechteck, ohne Überlappung, weißes Rauschen, n = 4096), K = 1 / 4 / 16 / 64? | Variationskoeffizient je Bin 0,99 / 0,50 / 0,25 / 0,13 (Theorie 1/√K); Streuung des Logarithmus 5,5 / 2,3 / 1,1 / 0,56 dB; Bias des Logarithmus −2,5 / −0,56 / −0,14 / −0,03 dB (Theorie ψ(K) − ln K). |
| Deckt das 95-%-Band mit ν = 2·K? | In 94,4 bis 95,1 % der Fälle (K = 1 bis 64). |
| Was bringt die Überlappung (Länge 256, n = 4096; ohne Überlappung 16 unabhängige Mittel)? | Hann: 50 % → 29,8 (Faktor 1,86), 62,5 % → 32,4, 75 % → 32,4; Rechteck: 50 % → 21,0 (Faktor 1,31), 75 % → 22,7; Blackman 75 % → 37,6. Die Formel trifft auf 2 %. |
| Gibt es ein Optimum (Resonanz, Hann 50 %, n = 4096)? | Gesamtfehler in dB bei Länge 32 / 64 / **128** / 256 / 512 / 1024 / 4096: 1,98 / 1,04 / **0,64** / 0,82 / 1,18 / 1,77 / 6,11. Der gemittelte Gipfel liegt bei Länge 128 um 1,7 dB, bei 32 um 5,7 dB zu tief. |
| Wie dicht dürfen zwei gleich starke Linien liegen (10 dB je Linie, Hann 50 %, n = 2048)? | Zuverlässig (100 %) getrennt ab etwa 2,5 Bins der Segmentlänge; darunter fällt die Auflösung stufenlos ab: 1,5 Bins in 28 % der Läufe, 2,5 Bins in 100 %; bei 2 Bins 79 %, bei 1 Bin 0 % (die Werte gehören zu verschiedenen Segmentlängen: 256 für 1,5 und 2,5 Bins, 128 für 2 Bins, 64 für 1 Bin). |
| Wie stark füllt sich das Tal bei 57 dB Spannweite (Länge 256)? | Bartlett mit Rechteck-Segmenten um 14,6 dB, Welch mit Hann (50 %) und Hann ohne Überlappung um 0,0 dB; das über 16 oder 29 Bins geglättete Rechteck-Periodogramm der ganzen Aufzeichnung um 4,2 bis 4,3 dB. |
| Ist Welch eine eigene Methode? | Bei mäßiger Dynamik (Hintergrund „Resonanz“) kaum mehr als eine Glättung: Bei gleicher Streuung (K<sub>eff</sub> 60 / 29 / 14) hat das über m Bins geglättete Rechteck-Periodogramm fast denselben Gesamtfehler (0,66 / 0,82 / 1,18 dB gegen 0,66 / 0,81 / 1,17); das Hann-Periodogramm braucht 2,4-fache Bandbreite und liegt am Gipfel 3,7 dB statt 1,7 dB tief. Bei hoher Dynamik (Hintergrund „Steil“) nicht: Welch 1,08 / 0,88 / 1,18 dB, das geglättete Rechteck-Periodogramm 2,37 / 2,29 / 2,41 dB (Leckage im Tal). |
| Hilft das Mitteln beim Finden einer Linie (−20 dB je Abtastwert, 99-%-Schwelle)? | Anteil gefunden bei K = 1 / 2 / 4 / 8 / 16 / 32: 50 / 69 / 64 / 64 / 47 / 27 %. Ein flaches Optimum bei wenigen Segmenten (die Reihenfolge von K = 2 / 4 / 8 liegt im Rauschen); K = 16 liegt gegenüber K = 1 im Rauschen, K = 32 deutlich darunter. Die Basis K = 1 ist ein Hann-Periodogramm, das die Ränder wegwirft: Ein Rechteck-Periodogramm der ganzen Aufzeichnung findet die Linie in rund 61 % der Läufe, Mitteln bringt dagegen höchstens einige Prozentpunkte. |

## Ehrliche Grenzen

- **Die Formel für K<sub>eff</sub> gilt für weißes Gauß-Rauschen** und für Frequenzen fern von 0 und Nyquist; bei farbigem Hintergrund ist die Streuung gemessen, die Formel Näherung.
- **Auflösung und Auffinden hängen vom Kriterium ab** (10 % Prominenz; 99-%-Quantil der Rauschmaxima) und von Phase und Lage der Linien; die Zahlen sind Anteile, keine Grenzen.
- **Das Optimum gilt für diese Resonanz** (Pol 0,97), nicht allgemein.
- **Bis zu zwei Linien, kein Trend, stationär.** Ein Trend leckt in alle Segmente; Nichtstationarität ist das Thema des Spektrogramms.
- **Arithmetisches Mittel:** ein Ausreißer in einem Segment verzerrt es; ein Median-Mittel ist robuster und nicht Teil dieser Demo.
- **Bleibt ein Rest der Aufzeichnung übrig** (n − L nicht durch die Schrittweite teilbar), wird er nicht genutzt.

## Verifikation

- **Kern gegen Handrechnung:** Hann der Länge 4 mit Schrittweite 2 und 8 Messwerten: ρ₁ = 1/36, K = 3, K<sub>eff</sub> = 2,893; Bartlett K = 2: K<sub>eff</sub> = 2; Welch einer Konstanten
  (2,667 / 0,667 / 0); Log-Bias bei K = 2: −1,174 dB.
- **Orakel (`tests/test_oracle_welch.py`):** Welch gegen `scipy.signal.welch` (alle vier Fenster, Überlappungen 0 / 50 / 75 %, gerade und ungerade Aufzeichnungslängen); Digamma, Trigamma, χ²-Quantile gegen `scipy`;
  Streufaktor gegen die exakte quadratische Form (2·tr(Q²)) und gegen eine große Simulation; Erwartungswert (Rauschen und Linien) gegen Simulation; Gipfelzählung gegen `scipy.signal.find_peaks`.
- **Generator** (`tests/test_generator.py`): wortgleich aus der ersten Demo.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder `st.slider` an den Grenzen; die Auswahlregler nur in Einzeltests), alle Fenster und Hintergründe, Permalink, Segmentlänge gegen Länge, Footer.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich) |
| `wbl_windows.py` | Fenster, Rauschbandbreite, Erwartungswert des gefensterten Periodogramms, Gipfelzählung |
| `wbl_welch.py` | Kern: Segmentmittel, Streufaktor, K<sub>eff</sub>, Log-Bias, χ²-Band, Glättung, Rechenaufwand |
| `wbl_constants.py` | Regler-Grenzen, feste Annahmen, Experimente, Presets |
| `wbl_evaluation.py` | Analyse, Experimente, Verdikte |
| `wbl_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `wbl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `wbl_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Generator, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Multitaper:** nutzt die ganze Aufzeichnung geschickter als das Zerschneiden; Thema der Folgedemo.
- **Konfidenzintervalle im Detail** (Jackknife, Whittle, Falschalarmschwellen): hier nur das χ²-Band; die Theorie folgt in einem späteren Stück.
- **Median-Mittel, Trendabzug, Zeitvariation:** nicht enthalten.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
