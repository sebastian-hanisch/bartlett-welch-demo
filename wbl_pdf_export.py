"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from wbl_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "β": "beta", "Σ": "Summe", "±": "+-", "√": "Wurzel aus "}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Bartlett und Welch: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    lines = {"keine": "keine Linie", "eine": f"eine Linie bei {de(s.f1_hz, 1)} Hz ({s.snr} dB je Abtastwert)",
             "zwei": f"zwei Linien bei {de(s.f1_hz, 1)} und {de(s.f2_hz, 1)} Hz ({s.snr} dB, zweite {s.rel} dB)"}[s.lines]
    pdf.multi_cell(0, 5, _s(f"Folge der Länge n = {s.n}, Segmentlänge {s.seg_eff}, Fenster {s.window}, Überlappung {de(100 * s.overlap, 1)} %, Hintergrund: {s.background}, "
                            f"{lines}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    out = [
        f"Segmente: {a.k}, äquivalente Zahl unabhängiger Mittel {de(a.k_eff, 1)} (weißes Rauschen, exakte Formel).",
        f"Streuung des Logarithmus je Frequenz: {de(a.std_db, 2)} dB gegen {de(a.full_std_db, 2)} dB beim Periodogramm über alles; Bias des Logarithmus {de(a.log_bias_db, 2)} dB.",
        f"Rauschbandbreite des Fensters: {de(a.enbw_hz, 2)} Hz gegen {de(a.full_enbw_hz, 2)} Hz für die ganze Aufzeichnung.",
        f"Rechenaufwand (komplexe Multiplikationen): {a.ops} gegen {a.ops_full} für ein Periodogramm über alles.",
    ]
    if a.peak_bias_db is not None:
        out.append(f"Erwartungswert am Gipfel des Hintergrunds gegen die wahre Dichte: {de(a.peak_bias_db, 2)} dB.")
    if a.resolved is not None:
        out.append(f"Zwei Linien im Segmentmittel aufgelöst: {'ja' if a.resolved else 'nein'}; im Periodogramm über alles: {'ja' if a.full_resolved else 'nein'}.")
    for line in out:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: bartlett-welch-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
