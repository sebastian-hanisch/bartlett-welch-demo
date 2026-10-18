"""Bartlett und Welch: Mitteln beruhigt das Spektrum, kostet aber Schärfe - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 3 der Reihe „Spektral- und Wavelet-Analyse“. Ein simuliertes Signal (Hintergrund mit bekannter Dichte, bis zu zwei Linien) wird in Segmente geschnitten,
gefenstert und gemittelt; gezeigt werden Streuung, Auflösung, Leckage und die Wirkung der Überlappung. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import wbl_constants as C
import wbl_evaluation as ev
from wbl_evaluation import Settings, analyse, de
from wbl_pdf_export import build_pdf
from wbl_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seg_options,
    sync_query_params,
)
from wbl_visualization import (
    build_detect,
    build_leakage,
    build_overlap,
    build_segments,
    build_spectrum,
    build_tradeoff,
    build_two_lines,
    build_variance,
)

st.set_page_config(page_title="Bartlett und Welch – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _experiments():
    return {"variance": ev.sweep_variance(), "overlap": ev.sweep_overlap(), "tradeoff": ev.sweep_tradeoff(), "two_lines": ev.sweep_two_lines(),
            "leakage": ev.sweep_leakage(), "smoothing": ev.sweep_smoothing(), "detect": ev.sweep_detect()}


st.title("📊 Bartlett und Welch: Mitteln beruhigt das Spektrum, kostet aber Schärfe")
st.markdown(
    """
Ein einzelnes **Periodogramm** streut bei weißem Gauß-Rauschen an inneren Bins so stark wie sein Wert (Demo 1). Wer die Aufzeichnung in **Segmente** schneidet, jedes für sich in ein Spektrum verwandelt und die Spektren
**mittelt**, beruhigt die Kurve: Die Streuung sinkt mit der Zahl der Segmente. Der Preis: Jedes Segment ist kürzer, die **Frequenzauflösung** wird im gleichen Maß gröber, Gipfel
verwischen, Täler füllen sich. **Bartlett** schneidet ohne Überlappung mit Rechteck; **Welch** überlappt die Segmente und gewichtet sie mit einem Fenster.

Dies ist die dritte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf **Periodogramm** und **Fenster** aufbauend; „Baum-Welch“ (Hidden-Markov-Modelle) ist etwas anderes. Alles ist
**simuliert**: Hintergrund, Linien und die wahre Dichte sind bekannt. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Mitteln", expanded=True):
    st.markdown(
        """
1. **Segmentieren:** Die Aufzeichnung der Länge n wird in K Segmente der Länge L geschnitten, im Abstand der Schrittweite (bei 50 % Überlappung L/2).
2. **Fenstern und transformieren:** Jedes Segment wird mit dem Fenster gewichtet und per FFT in ein Periodogramm verwandelt.
3. **Mitteln:** Die K Periodogramme werden Bin für Bin gemittelt. Sind sie unabhängig, sinkt die Streuung auf 1/√K des Werts.
4. **Überlappung:** Überlappte Segmente sind nicht unabhängig; das Fenster dämpft die Ränder, die der Nachbar dafür noch nutzt. Die **äquivalente Zahl unabhängiger Mittel** K<sub>eff</sub>
   steht zwischen K ohne Überlappung und K.
5. **Auflösung:** Das Fenster der Länge L hat eine Hauptkeule von rund 1,5 Bins · f<sub>s</sub>/L Breite: Zwei Linien werden ab etwa 2,5 Bins der Segmentlänge zuverlässig getrennt; darunter fällt der Anteil der Läufe, in denen sie getrennt erscheinen, stufenlos ab.
        """,
        unsafe_allow_html=True,
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Mitteln**")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Anzahl der Messwerte (Abtastrate 1000 Hz).")
    if st.session_state["seg_select"] > n:
        st.session_state["seg_select"] = n
    seg = st.select_slider("Segmentlänge L", options=seg_options(n), key="seg_select",
                           help="Länge eines Segments. L = n ist ein einziges Periodogramm über alles. Kürzer: mehr Segmente, ruhigere Kurve, gröbere Auflösung.")
    overlap = st.select_slider("Überlappung", options=list(C.OVERLAP_CHOICES), key="overlap_select", format_func=lambda v: f"{de(100 * v, 1)} %",
                               help="Anteil, um den aufeinanderfolgende Segmente überlappen. 0 % = Bartlett (mit Rechteck).")
    window = st.selectbox("Fenster", list(C.WINDOWS), key="window_select", help="Rechteck = kein Fenster (Bartlett). Die anderen blenden die Segmentränder sanft aus.")
    st.markdown("**Signal**")
    background = st.selectbox("Hintergrund", list(C.BACKGROUNDS), key="background_select",
                              help="Weiß: flach. Resonanz: ein Gipfel bei 200 Hz. Steil: 57 dB Spannweite, zeigt Leckage im Tal.")
    lines = st.radio("Linien", list(C.LINES), key="lines_select", horizontal=True, help="Reine Töne auf dem Hintergrund (Lage und Stärke bekannt).")
    if lines != "keine":
        line = st.slider("Linie 1 (Bruchteil der Abtastrate)", C.LINE_MIN, C.LINE_MAX, float(st.session_state["line_kept"]), step=C.LINE_STEP, format="%.4f",
                         help="Lage der ersten Linie; mal 1000 Hz ergibt die Frequenz.")
        st.session_state["line_kept"] = float(line)
        snr = st.slider("Linie 1 gegen Rauschen (dB je Abtastwert)", C.SNR_MIN, C.SNR_MAX, int(st.session_state["snr_kept"]),
                        help="Linienleistung gegen die Rauschvarianz 1. Bei −20 dB steckt die Linie 20 dB unter dem Rauschen eines Abtastwerts.")
        st.session_state["snr_kept"] = int(snr)
    else:
        line, snr = float(st.session_state["line_kept"]), int(st.session_state["snr_kept"])
    if lines == "zwei":
        gap = st.slider("Abstand der Linie 2 (Bins der Länge 256)", C.GAP_MIN, C.GAP_MAX, float(st.session_state["gap_kept"]), step=C.GAP_STEP,
                        help="Abstand der zweiten Linie in Vielfachen von f_s/256 (3,9 Hz).")
        st.session_state["gap_kept"] = float(gap)
        rel = st.slider("Linie 2 gegen Linie 1 (dB)", C.REL_MIN, C.REL_MAX, int(st.session_state["rel_kept"]), help="0 dB: gleich stark.")
        st.session_state["rel_kept"] = int(rel)
    else:
        gap, rel = float(st.session_state["gap_kept"]), int(st.session_state["rel_kept"])
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "seg_select": int(seg), "overlap_select": float(overlap), "window_select": window, "background_select": background,
                   "lines_select": lines, "line_kept": float(line), "gap_kept": float(gap), "rel_kept": int(rel), "snr_kept": int(snr),
                   "seed_input": int(seed)})

settings = Settings(window, int(n), int(seg), float(overlap), background, lines, float(line), float(gap), int(rel), int(snr), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Segmente", f"{a.k}", delta=f"≈ {de(a.k_eff, 1)} unabhängige Mittel", delta_color="off")
m2.metric("Streuung je Frequenz", f"±{de(a.std_db, 1)} dB", delta=f"statt ±{de(a.full_std_db, 1)} dB mit einem Periodogramm", delta_color="off")
m3.metric("Rauschbandbreite", f"{de(a.enbw_hz, 2)} Hz", delta=f"statt {de(a.full_enbw_hz, 2)} Hz über alles", delta_color="off")
m4.metric("Rechenaufwand", f"{a.ops:,}".replace(",", "."), delta=f"{de(a.ops / a.ops_full, 1)}-fach eines Periodogramms über alles", delta_color="off",
          help="Komplexe Multiplikationen der Segment-FFTs ohne Auffüllen (Radix 2).")
st.plotly_chart(build_spectrum(a, zoom=False), width="stretch", key="spectrum_full")
st.caption("Grau: ein Periodogramm über die ganze Aufzeichnung (selbes Fenster). Blau: Segmentmittel mit 95-%-Band (χ² mit 2·K<sub>eff</sub> Freiheitsgraden), orange: exakter Erwartungswert "
           "der Schätzung (zeigt den Verlust an Schärfe), grün gepunktet: wahre Dichte des Hintergrunds, rot gestrichelt: wahre Lage der Linien.", unsafe_allow_html=True)
if lines != "keine":
    st.plotly_chart(build_spectrum(a, zoom=True), width="stretch", key="spectrum_zoom")
    st.caption("Ausschnitt um die Linien.")
if a.peak_bias_db is not None:
    st.caption(f"Am Gipfel des Hintergrunds liegt der Erwartungswert des Segmentmittels {de(abs(a.peak_bias_db), 1)} dB {'unter' if a.peak_bias_db < 0 else 'über'} der wahren Dichte.")
with st.expander("Wie die Segmente liegen"):
    st.plotly_chart(build_segments(a), width="stretch", key="segments")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="bartlett_welch.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was kauft das Mitteln, und was kostet es?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption(f"Feste Zufallsströme (Seed {C.SWEEP_SEED} und folgende), unabhängig von den Reglern. Aufzeichnung n = {C.EXP_N}, Abtastrate {C.FS:g} Hz; „Läufe“ sind unabhängige Wiederholungen.")
exp = _experiments()

st.markdown("#### Die Streuung sinkt mit der Zahl der Segmente")
st.caption(f"Bartlett (Rechteck, ohne Überlappung), weißes Rauschen, {C.EXP_VAR_RUNS} Läufe; je Zahl der Segmente K.")
st.dataframe(pd.DataFrame([{"K": r["k"], "Länge L": r["seg"], "Variationskoeffizient": de(r["cv"], 3), "Theorie 1/√K": de(r["cv_theory"], 3),
                            "Bias des Logarithmus (dB)": de(r["bias_db"], 2), "Bias Theorie (dB)": de(r["bias_theory"], 2), "Streuung (dB)": de(r["std_db"], 2),
                            "Streuung Theorie (dB)": de(r["std_theory"], 2), "95-%-Band deckt": de(100 * r["coverage"], 1) + " %"} for r in exp["variance"]]),
             hide_index=True, width="stretch")
st.plotly_chart(build_variance(exp["variance"]), width="stretch", key="exp_variance")
kind, text = ev.verdict_variance(exp["variance"])
getattr(st, kind)(text)

st.markdown("#### Überlappung und Fenster")
st.caption(f"Weißes Rauschen, Segmentlänge {C.EXP_OVL_SEG}, {C.EXP_OVL_RUNS} Läufe. Gemessen wird K<sub>eff</sub> = 1/CV² und verglichen mit der exakten Formel.", unsafe_allow_html=True)
tbl = []
for ov in C.EXP_OVL_OVERLAPS:
    row = {"Überlappung": de(100 * ov, 1) + " %"}
    for name in C.WINDOWS:
        r = [x for x in exp["overlap"] if x["window"] == name and x["overlap"] == ov][0]
        row[name] = f"{de(r['k_eff'], 1)} ({de(r['k_eff_formula'], 1)})"
    row["Segmente K"] = [x for x in exp["overlap"] if x["overlap"] == ov][0]["k"]
    tbl.append(row)
st.dataframe(pd.DataFrame(tbl), hide_index=True, width="stretch")
st.caption("Zahl unabhängiger Mittel gemessen (Formel in Klammern). Ohne Überlappung sind es 16.")
st.plotly_chart(build_overlap(exp["overlap"]), width="stretch", key="exp_overlap")
kind, text = ev.verdict_overlap(exp["overlap"])
getattr(st, kind)(text)

st.markdown("#### Das Optimum: Streuung gegen Schärfe")
st.caption(f"Hintergrund mit Resonanz bei 200 Hz, Hann, 50 % Überlappung, {C.EXP_TRADE_RUNS} Läufe; Fehler der Schätzung in dB gegen die wahre Dichte.")
st.dataframe(pd.DataFrame([{"Segmentlänge": r["seg"], "Segmente": r["k"], "Rauschbandbreite (Hz)": de(r["enbw_hz"], 2), "Bias (dB)": de(r["bias_db"], 2),
                            "Streuung (dB)": de(r["std_db"], 2), "Gesamtfehler RMSE (dB)": de(r["rmse_db"], 2), "Gipfel im Mittel (dB)": de(r["peak_db"], 2)}
                           for r in exp["tradeoff"]]), hide_index=True, width="stretch")
st.plotly_chart(build_tradeoff(exp["tradeoff"]), width="stretch", key="exp_tradeoff")
kind, text = ev.verdict_tradeoff(exp["tradeoff"])
getattr(st, kind)(text)
st.caption(f"Zwei gleich starke Linien in weißem Rauschen (10 dB je Linie), n = {C.EXP_RES_N}, Hann, 50 %, {C.EXP_RES_RUNS} Läufe; Abstand in Bins der Länge 256. Aufgelöst heißt zwei Gipfel "
           "mit mindestens 10 % Prominenz.")
st.plotly_chart(build_two_lines(exp["two_lines"]), width="stretch", key="exp_two_lines")
kind, text = ev.verdict_two_lines(exp["two_lines"])
getattr(st, kind)(text)

st.markdown("#### Das Fenster je Segment: Leckage")
st.caption(f"Steiler Hintergrund (57 dB Spannweite), Segmentlänge {C.EXP_LEAK_SEG}, {C.EXP_LEAK_RUNS} Läufe; Mittelwert des Schätzers gegen die wahre Dichte. Gipfel: 150–250 Hz, "
           "Flanke: 300–350 Hz, Tal: 450–500 Hz.")
st.plotly_chart(build_leakage(exp["leakage"]), width="stretch", key="exp_leakage")
kind, text = ev.verdict_leakage(exp["leakage"])
getattr(st, kind)(text)
with st.expander("Glätten statt Segmentieren"):
    st.markdown("Statt die Aufzeichnung zu zerschneiden, kann man das Periodogramm der ganzen Aufzeichnung über benachbarte Bins mitteln. Verglichen wird bei gleicher exakter Streuung "
                "(gleiches K<sub>eff</sub>): Welch mit Hann gegen das Rechteck-Periodogramm und gegen das Hann-Periodogramm, jeweils über m Bins gemittelt.", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Verfahren": r["label"], "K_eff": de(r["k_eff"], 1), "Streuung (dB)": de(r["std_db"], 2), "Gipfel (dB)": de(r["peak_db"], 2),
                                "RMSE (dB)": de(r["rmse_db"], 2), "Bandbreite (Hz)": de(r["bandwidth_hz"], 1)} for r in exp["smoothing"]]), hide_index=True, width="stretch")
    kind, text = ev.verdict_smoothing(exp["smoothing"])
    getattr(st, kind)(text)

st.markdown("#### Hilft das Mitteln beim Finden einer Linie?")
st.caption(f"Eine Linie zwischen zwei Bins in weißem Rauschen, Hann, 50 % Überlappung, {C.EXP_DET_RUNS} Läufe. Gefunden heißt: Das Maximum des Segmentmittels übersteigt das "
           "99-%-Quantil der Maxima ohne Linie. Die Falschalarmtheorie ist Thema eines späteren Stücks.")
st.plotly_chart(build_detect(exp["detect"]), width="stretch", key="exp_detect")
kind, text = ev.verdict_detect(exp["detect"])
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Weißes Gauß-Rauschen** in der Formel für K<sub>eff</sub> | Bei gefärbtem Hintergrund ist die Streuung gemessen, die Formel nur Näherung. | Konfidenzintervalle |
| **Stationär** | Ändert sich das Spektrum über die Aufzeichnung, mittelt das Verfahren darüber hinweg. | Spektrogramm |
| **Zwei Linien, kein Trend** | Mehr Linien und ein Trend sind nicht untersucht; ein Trend leckt in alle Segmente. | Unterraumverfahren |
| **Arithmetisches Mittel** | Ein Ausreißer in einem Segment verzerrt das Mittel; ein Median-Mittel ist robuster, hier nicht behandelt. | – |
| **Segmentlänge Zweierpotenz, ganze Segmente** | Bleiben Messwerte am Ende übrig, werden sie nicht genutzt. | – |
| **Auflösung gegen Streuung** | Bei fester Aufzeichnungslänge verbessert Mitteln die Streuung nur auf Kosten der Auflösung; ein günstigerer Kompromiss braucht ein besseres Verfahren (Multitaper) oder zusätzliche Annahmen (parametrische Modelle). | Multitaper |
""",
    unsafe_allow_html=True,
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Welch-Schätzer.** Segmente $x^{(j)}_m = x[m + jD]$, $m = 0..L-1$, $j = 0..K-1$, $K = \lfloor (n-L)/D \rfloor + 1$, Fenster $w$:
$\hat S(f) = \dfrac{1}{K}\sum_{j} \dfrac{1}{f_s \sum_m w_m^2}\Big|\sum_m w_m x^{(j)}_m e^{-2\pi i f m/f_s}\Big|^2$. Bartlett: $D = L$, $w \equiv 1$.

**Streuung.** Für weißes Gauß-Rauschen ist $\mathrm{Var}(\hat S)/S^2 = \dfrac1K\Big(1 + 2\sum_{j=1}^{K-1}(1 - \tfrac{j}{K})\,\rho_j\Big)$ mit
$\rho_j = \big(\sum_m w_m w_{m+jD}\big)^2 / \big(\sum_m w_m^2\big)^2$; $K_\mathrm{eff} = 1/\text{Streufaktor}$. Hann mit 50 %: $\rho_1 = 1/36$.

**Logarithmus.** $\hat S \sim S\,\chi^2_{\nu}/\nu$ mit $\nu = 2K_\mathrm{eff}$: Bias von $10\log_{10}\hat S$ gleich $\tfrac{10}{\ln 10}(\psi(K_\mathrm{eff}) - \ln K_\mathrm{eff})$,
Streuung $\tfrac{10}{\ln 10}\sqrt{\psi'(K_\mathrm{eff})}$ ($-2{,}51$ dB und $5{,}57$ dB für $K_\mathrm{eff} = 1$).

**Konfidenzband.** $S \in \big[\nu\hat S/\chi^2_{\nu;0{,}975},\ \nu\hat S/\chi^2_{\nu;0{,}025}\big]$ mit 95 %.

**Erwartungswert (exakt).** Unabhängig von der Überlappung: $E\hat S(f) = \dfrac{1}{f_s\sum w^2}\Big(\dfrac1K\sum_j\Big|\sum_m w_m s^{(j)}_m e^{-i\omega m}\Big|^2 + \sum_{|\tau|<L} r(\tau)\,c_w(\tau)\cos(\omega\tau)\Big)$
mit den Linien $s$, der Autokovarianz $r$ des Hintergrunds und der Fensterautokorrelation $c_w(\tau) = \sum_m w_m w_{m+\tau}$. Leckage und die Glättung durch das kurze Fenster sind Bias.

**Quellen.** Lehrbuchwissen (Bartlett 1948; Welch 1967; Harris 1978; Percival und Walden 1993). Formeln hier per Simulation, Handrechnung und gegen scipy geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
