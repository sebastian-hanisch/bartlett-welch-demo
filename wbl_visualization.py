"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import wbl_constants as C
from wbl_evaluation import Analysis

EPS = 1e-30


def _db(v):
    return 10.0 * np.log10(np.maximum(np.asarray(v, dtype=float), EPS))


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_spectrum(a: Analysis, zoom: bool = False) -> go.Figure:
    """Spektrum in dB über der Frequenz in Hz; zoom: Ausschnitt um die Linien, sonst bis zur Nyquist-Frequenz."""
    s = a.settings
    lines = []
    if s.lines in ("eine", "zwei"):
        lines.append(s.f1_hz)
    if s.lines == "zwei":
        lines.append(s.f2_hz)
    if zoom and lines:
        pad = max(6.0 * C.FS / s.seg_eff, 0.5 * (max(lines) - min(lines)))
        lo, hi = max(0.0, min(lines) - pad), min(C.FS / 2, max(lines) + pad)
    else:
        lo, hi = 0.0, C.FS / 2
    sel = (a.f_hz >= lo) & (a.f_hz <= hi)
    sel_full = (a.f_full >= lo) & (a.f_full <= hi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.f_full[sel_full], y=_db(a.full[sel_full]), mode="lines", line=dict(color=C.COLOR_FULL, width=1), name="ein Periodogramm über alles",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.band_hi[sel]), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.band_lo[sel]), mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND,
                             name="95-%-Band um die Schätzung", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.est[sel]), mode="lines", line=dict(color=C.COLOR_WELCH, width=2), name="Segmentmittel",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.expected[sel]), mode="lines", line=dict(color=C.COLOR_EXPECTED, width=2.5), name="Erwartungswert (exakt)",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.f_hz[sel], y=_db(a.true_density[sel]), mode="lines", line=dict(color=C.COLOR_TRUE, width=2, dash="dot"),
                             name="wahre Dichte des Hintergrunds", hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    for pos in lines:
        if lo <= pos <= hi:
            fig.add_vline(x=pos, line=dict(color=C.COLOR_LINE, width=1.2, dash="dash"))
    fig.update_xaxes(title_text="Frequenz (Hz)", range=[lo, hi])
    fig.update_yaxes(title_text="Leistungsdichte (dB)")
    return _fixed(fig, height=400)


def build_segments(a: Analysis, limit: int = 8) -> go.Figure:
    """Die ersten Segmente: Lage und Fensterform über der Aufzeichnung (zeigt Länge, Überlappung und Gewichtung)."""
    s = a.settings
    seg, hop = s.seg_eff, s.hop
    shown = min(a.k, limit)
    fig = go.Figure()
    wn = a.w / a.w.max()
    for i in range(shown):
        start = i * hop
        fig.add_trace(go.Scatter(x=np.arange(start, start + seg), y=wn * 0.8 + (shown - 1 - i), mode="lines", line=dict(color=C.COLOR_WELCH, width=1.8),
                                 name=f"Segment {i + 1}", showlegend=False, hovertemplate=f"Segment {i + 1}, Messwert %{{x}}<extra></extra>"))
    fig.update_xaxes(title_text=f"Messwert Nr. (Aufzeichnung 0 bis {s.n - 1}; gezeigt: die ersten {shown} von {a.k} Segmenten)",
                     range=[0, min(s.n, (shown - 1) * hop + seg)])
    fig.update_yaxes(showticklabels=False, range=[-0.1, shown + 0.1])
    return _fixed(fig, height=60 + 28 * shown, showlegend=False)


def build_variance(rows) -> go.Figure:
    ks = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[r["std_theory"] for r in rows], mode="lines", line=dict(color="gray", dash="dot"), name="Theorie",
                             hovertemplate="K = %{x}: %{y:.2f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=ks, y=[r["std_db"] for r in rows], mode="markers+lines", name="gemessen", marker=dict(size=9),
                             hovertemplate="K = %{x}: %{y:.2f} dB<extra></extra>"))
    fig.update_xaxes(title_text="Zahl der Segmente K", type="log", tickvals=ks)
    fig.update_yaxes(title_text="Streuung des Logarithmus (dB)", range=[0, max(r["std_theory"] for r in rows) * 1.1])
    return _fixed(fig, height=320)


def build_overlap(rows) -> go.Figure:
    fig = go.Figure()
    for name in dict.fromkeys(r["window"] for r in rows):
        sel = [r for r in rows if r["window"] == name]
        fig.add_trace(go.Scatter(x=[100 * r["overlap"] for r in sel], y=[r["gain"] for r in sel], mode="lines+markers", name=name,
                                 hovertemplate=name + ", %{x:.1f} % Überlappung: Faktor %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Überlappung (%)")
    fig.update_yaxes(title_text="unabhängige Mittel gegen ohne Überlappung", range=[0.8, max(r["gain"] for r in rows) * 1.1])
    return _fixed(fig, height=340)


def build_tradeoff(rows) -> go.Figure:
    segs = [str(r["seg"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=segs, y=[r["rmse_db"] for r in rows], mode="lines+markers", name="Gesamtfehler (RMSE)", line=dict(width=3),
                             hovertemplate="Länge %{x}: %{y:.2f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=segs, y=[r["std_db"] for r in rows], mode="lines+markers", name="Streuung",
                             hovertemplate="Länge %{x}: %{y:.2f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=segs, y=[abs(r["peak_db"]) for r in rows], mode="lines+markers", name="Verlust am Gipfel (Betrag)",
                             hovertemplate="Länge %{x}: %{y:.2f} dB<extra></extra>"))
    fig.update_xaxes(title_text="Segmentlänge (Hann, 50 % Überlappung)", type="category")
    fig.update_yaxes(title_text="dB", range=[0, max(max(r["rmse_db"], abs(r["peak_db"])) for r in rows) * 1.1])
    return _fixed(fig, height=340)


def build_two_lines(rows) -> go.Figure:
    segs = [k for k in rows[0] if k not in ("delta", "runs")]
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(x=[str(s) for s in segs], y=[100 * r[s] for s in segs], mode="lines+markers",
                                 name=f"Abstand {r['delta']:g} Bins".replace(".", ","), hovertemplate="Länge %{x}: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Segmentlänge", type="category")
    fig.update_yaxes(title_text="Anteil aufgelöst (%)", range=[-3, 103])
    return _fixed(fig, height=340)


def build_leakage(rows) -> go.Figure:
    names = ["Gipfel", "Flanke", "Tal"]
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Bar(x=names, y=r["bands"], name=r["label"], hovertemplate="%{x}: %{y:.1f} dB<extra></extra>"))
    fig.update_yaxes(title_text="Mittelwert über wahrer Dichte (dB)")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=380)


def build_detect(rows) -> go.Figure:
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(x=[str(k) for k in r["k"]], y=[100 * v for v in r["share"]], mode="lines+markers", name=f"{r['snr']} dB",
                                 hovertemplate="K = %{x}: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Zahl der Segmente K (Hann, 50 % Überlappung)", type="category")
    fig.update_yaxes(title_text="Linie gefunden (%)", range=[-3, 103])
    return _fixed(fig, height=340)
