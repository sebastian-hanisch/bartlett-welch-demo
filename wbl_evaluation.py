"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import signal_generator as G
import wbl_constants as C
import wbl_welch as H
import wbl_windows as W

DB = H.DB


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def percent(share: float) -> str:
    """Anteil als Prozentzahl mit Dezimalkomma, ohne überflüssige Nachkommastelle (50 %, 62,5 %)."""
    return f"{100.0 * share:g}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    window: str = C.DEFAULT_WINDOW
    n: int = C.DEFAULT_N
    seg: int = C.DEFAULT_SEG
    overlap: float = C.DEFAULT_OVERLAP
    background: str = C.DEFAULT_BACKGROUND
    lines: str = C.DEFAULT_LINES
    line: float = C.DEFAULT_LINE
    gap: float = C.DEFAULT_GAP
    rel: int = C.DEFAULT_REL
    snr: int = C.DEFAULT_SNR
    seed: int = C.DEFAULT_SEED

    @property
    def seg_eff(self) -> int:
        """Segmentlänge, nicht länger als die Folge."""
        return min(self.seg, self.n)

    @property
    def hop(self) -> int:
        return H.hop_for(self.seg_eff, self.overlap)

    @property
    def f1_hz(self) -> float:
        return self.line * C.FS

    @property
    def f2_hz(self) -> float:
        return self.f1_hz + self.gap * C.FS / C.GAP_REF


def noise_spec(background: str):
    """(Pole, Innovations-Sigma, weißes Sigma) eines Hintergrunds, auf Varianz 1 normiert."""
    poles = C.BACKGROUNDS[background]
    if poles:
        return poles, G.ar_unit_sigma(G.ar_from_poles(poles)), 0.0
    return (), 0.0, 1.0


def amplitude(snr_db: float) -> float:
    return math.sqrt(2.0 * 10.0 ** (snr_db / 10.0))


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    k: int                        # Zahl der Segmente
    k_eff: float                  # äquivalente Zahl unabhängiger Mittel (weißes Rauschen)
    f_hz: np.ndarray              # Frequenzachse der Welch-Schätzung
    x: np.ndarray
    w: np.ndarray
    est: np.ndarray               # Welch-Schätzung (Leistungsdichte)
    expected: np.ndarray          # exakter Erwartungswert der Schätzung
    true_density: np.ndarray      # wahre Dichte des Hintergrunds
    band_lo: np.ndarray           # 95-%-Band um die Schätzung (Dichte)
    band_hi: np.ndarray
    f_full: np.ndarray            # Periodogramm der ganzen Aufzeichnung (selbes Fenster, ohne Auffüllen)
    full: np.ndarray
    enbw_hz: float                # Rauschbandbreite des Segmentfensters in Hz
    full_enbw_hz: float
    std_db: float                 # Streuung des Logarithmus je Frequenz nach der Formel (K_eff)
    full_std_db: float
    log_bias_db: float
    peak_bias_db: float | None    # Erwartungswert gegen wahre Dichte am Gipfel des Hintergrunds (nur mit Pol)
    resolved: bool | None         # zwei Linien im Welch-Mittel aufgelöst (None, wenn nicht zwei Linien)
    full_resolved: bool | None
    ops: int
    ops_full: int


def build_signal(s: Settings):
    rng = G.SplitMix64(s.seed)
    ph = 2.0 * math.pi * rng.uniform(2)
    poles, ar_sigma, white = noise_spec(s.background)
    a1 = amplitude(s.snr)
    a2 = a1 * 10.0 ** (s.rel / 20.0)
    lines = []
    if s.lines in ("eine", "zwei"):
        lines.append((s.f1_hz, a1, float(ph[0])))
    if s.lines == "zwei":
        lines.append((s.f2_hz, a2, float(ph[1])))
    sig = G.make_signal(s.n, C.FS, lines, poles, ar_sigma, white, rng)
    return sig, (poles, ar_sigma, white)


def analyse(s: Settings) -> Analysis:
    sig, (poles, ar_sigma, white) = build_signal(s)
    seg, hop = s.seg_eff, s.hop
    w = W.window(s.window, seg)
    est, k = H.welch(sig.x, seg, hop, w, C.FS, C.PAD * seg)
    f_hz = np.arange(len(est)) * C.FS / (C.PAD * seg)
    r = G.noise_autocov(poles, ar_sigma, white, seg)
    expected = W.expected_noise_spectrum(f_hz, C.FS, w, r)
    if s.lines != "keine":
        expected = expected + H.welch(sig.clean, seg, hop, w, C.FS, C.PAD * seg)[0]
    true_density = G.true_psd(f_hz, C.FS, poles, ar_sigma, white)
    k_eff = H.k_effective(w, hop, k)
    lo_f, hi_f = H.confidence_factors(k_eff)
    wf = W.window(s.window, s.n)
    full = np.abs(np.fft.rfft(sig.x * wf)) ** 2 / (C.FS * (wf ** 2).sum())
    f_full = np.arange(len(full)) * C.FS / s.n
    peak_bias = None
    if poles:
        f_peak = C.FS * poles[0][0]
        e_pk = float(W.expected_noise_spectrum([f_peak], C.FS, w, r)[0])
        peak_bias = DB * math.log(e_pk / float(G.true_psd([f_peak], C.FS, poles, ar_sigma, white)[0]))
    resolved = full_resolved = None
    if s.lines == "zwei":
        resolved = bool(W.two_lines_resolved(est, C.PAD, s.f1_hz * seg / C.FS, s.f2_hz * seg / C.FS))
        full_resolved = bool(W.two_lines_resolved(full, 1, s.f1_hz * s.n / C.FS, s.f2_hz * s.n / C.FS))
    return Analysis(s, k, k_eff, f_hz, sig.x, w, est, expected, true_density, lo_f * est, hi_f * est, f_full, full, W.enbw(w) * C.FS / seg,
                    W.enbw(wf) * C.FS / s.n, H.log_std_db(k_eff), H.log_std_db(1.0), H.log_bias_db(k_eff), peak_bias, resolved, full_resolved,
                    H.ops_welch(seg, k), H.ops_radix2(s.n))


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def _inner(n_bins: int) -> slice:
    """Bins mit zwei Freiheitsgraden eines Segment-Spektrums mit n_bins = seg / 2 + 1 Werten: ohne Gleichanteil und Nyquist."""
    return slice(1, n_bins - 1)


def _detect_slice(n_bins: int) -> slice:
    """Bins, in denen das Maximum für die Entdeckung gesucht wird: ohne die zwei Randbins an jedem Ende (Gleichanteil, Nyquist, Leckage der Ränder)."""
    return slice(2, n_bins - 2)


def _white_batch(runs: int, n: int, seed: int) -> np.ndarray:
    return G.make_noise_batch(runs, n, (), 0.0, 1.0, G.SplitMix64(seed))


def sweep_variance(ks=C.EXP_VAR_K, n: int = C.EXP_N, runs: int = C.EXP_VAR_RUNS) -> list[dict]:
    """Bartlett (Rechteck, ohne Überlappung), weißes Rauschen: Variationskoeffizient, Bias und Streuung des Logarithmus, Deckung des 95-%-Bands je Segmentzahl."""
    x = _white_batch(runs, n, C.SWEEP_SEED)
    s_true = 1.0 / C.FS
    rows = []
    for k in ks:
        seg = n // k
        est, kk = H.welch(x, seg, seg, np.ones(seg), C.FS)
        inner = est[:, _inner(est.shape[1])]
        lg = DB * np.log(inner / s_true)
        lo_f, hi_f = H.confidence_factors(float(kk))
        covered = (inner * lo_f <= s_true) & (s_true <= inner * hi_f)
        rows.append({"k": kk, "seg": seg, "cv": float((inner.std(axis=0) / inner.mean(axis=0)).mean()), "cv_theory": 1.0 / math.sqrt(kk),
                     "bias_db": float(lg.mean()), "bias_theory": H.log_bias_db(kk), "std_db": float(lg.std(axis=0).mean()), "std_theory": H.log_std_db(kk),
                     "coverage": float(covered.mean())})
    return rows


def sweep_overlap(windows=C.WINDOWS, overlaps=C.EXP_OVL_OVERLAPS, seg: int = C.EXP_OVL_SEG, n: int = C.EXP_N, runs: int = C.EXP_OVL_RUNS) -> list[dict]:
    """Weißes Rauschen, feste Segmentlänge: unabhängige Mittel K_eff = 1 / CV^2 (gemessen) und aus der exakten Formel, je Fenster und Überlappung."""
    x = _white_batch(runs, n, C.SWEEP_SEED + 1)
    k0 = n // seg
    rows = []
    for name in windows:
        w = W.window(name, seg)
        for ov in overlaps:
            hop = H.hop_for(seg, ov)
            est, k = H.welch(x, seg, hop, w, C.FS)
            inner = est[:, _inner(est.shape[1])]
            cv2 = float(((inner.std(axis=0) / inner.mean(axis=0)) ** 2).mean())
            rows.append({"window": name, "overlap": ov, "k": k, "k_eff": 1.0 / cv2, "k_eff_formula": H.k_effective(w, hop, k), "gain": 1.0 / cv2 / k0})
    return rows


def sweep_tradeoff(segs=C.EXP_TRADE_SEGS, n: int = C.EXP_N, runs: int = C.EXP_TRADE_RUNS, background: str = C.EXP_TRADE_BACKGROUND) -> list[dict]:
    """Hintergrund mit Resonanz, Hann 50 %: Fehler der Schätzung in dB gegen die wahre Dichte und Verlust am Gipfel (exakt) je Segmentlänge."""
    poles, ar_sigma, white = noise_spec(background)
    x = G.make_noise_batch(runs, n, poles, ar_sigma, white, G.SplitMix64(C.SWEEP_SEED + 2))
    r = G.noise_autocov(poles, ar_sigma, white, n)
    f_peak = C.FS * poles[0][0]
    s_peak = float(G.true_psd([f_peak], C.FS, poles, ar_sigma, white)[0])
    rows = []
    for seg in segs:
        w = W.window("Hann", seg)
        hop = H.hop_for(seg, 0.5)
        est, k = H.welch(x, seg, hop, w, C.FS)
        f = np.arange(est.shape[1]) * C.FS / seg
        true = G.true_psd(f, C.FS, poles, ar_sigma, white)
        sl = _inner(est.shape[1])
        err = DB * np.log(est[:, sl] / true[sl])
        peak = DB * math.log(float(W.expected_noise_spectrum([f_peak], C.FS, w, r[:seg])[0]) / s_peak)
        rows.append({"seg": seg, "k": k, "enbw_hz": W.enbw(w) * C.FS / seg, "bias_db": float(err.mean()), "std_db": float(err.std(axis=0).mean()),
                     "rmse_db": float(math.sqrt((err ** 2).mean())), "peak_db": peak})
    return rows


def sweep_two_lines(segs=C.EXP_RES_SEGS, deltas=C.EXP_RES_DELTAS, n: int = C.EXP_RES_N, runs: int = C.EXP_RES_RUNS, snr: float = C.EXP_RES_SNR) -> list[dict]:
    """Zwei gleich starke Linien in weißem Rauschen (SNR je Linie), Hann 50 %: Anteil der Läufe, in denen das Welch-Mittel zwei Gipfel zeigt, je Segmentlänge.
    Abstand in Bins der Länge 256."""
    rng = G.SplitMix64(C.SWEEP_SEED + 3)
    t = np.arange(n)
    sigma = math.sqrt(0.5 / 10.0 ** (snr / 10.0))
    rows = []
    for d in deltas:
        f1 = (40.0 + 3.0 * rng.uniform(runs)) / C.GAP_REF
        f2 = f1 + d / C.GAP_REF
        ph1, ph2 = 2.0 * math.pi * rng.uniform(runs), 2.0 * math.pi * rng.uniform(runs)
        x = (np.cos(2.0 * np.pi * f1[:, None] * t[None, :] + ph1[:, None]) + np.cos(2.0 * np.pi * f2[:, None] * t[None, :] + ph2[:, None])
             + sigma * rng.normal(runs * n).reshape(runs, n))
        row = {"delta": d, "runs": runs}
        for seg in segs:
            w = W.window("Hann", seg)
            est, _ = H.welch(x, seg, H.hop_for(seg, 0.5), w, 1.0, C.PAD * seg)
            row[seg] = float(np.mean([W.two_lines_resolved(est[i], C.PAD, f1[i] * seg, f2[i] * seg) for i in range(runs)]))
        rows.append(row)
    return rows


def _band_bias(f_frac, est_mean, true, bands) -> list[float]:
    out = []
    for lo, hi in bands:
        sel = (f_frac >= lo) & (f_frac <= hi)
        out.append(float(DB * np.log(est_mean[sel] / true[sel]).mean()))
    return out


LEAK_BANDS = ((0.15, 0.25), (0.30, 0.35), (0.45, 0.4999))      # Gipfel, Flanke, Tal (Bruchteile der Abtastrate)


def sweep_leakage(seg: int = C.EXP_LEAK_SEG, n: int = C.EXP_N, runs: int = C.EXP_LEAK_RUNS, smooth=C.EXP_LEAK_SMOOTH) -> list[dict]:
    """Steiler Hintergrund: Mittelwert des Schätzers gegen die wahre Dichte in dB (Gipfel, Flanke, Tal) für Welch mit Rechteck und Hann, Bartlett mit Hann und das über
    die ganze Aufzeichnung geglättete Rechteck-Periodogramm."""
    poles, ar_sigma, white = noise_spec("Steil")
    x = G.make_noise_batch(runs, n, poles, ar_sigma, white, G.SplitMix64(C.SWEEP_SEED + 4))
    rows = []
    f = np.arange(seg // 2 + 1) / seg
    true = G.true_psd(f * C.FS, C.FS, poles, ar_sigma, white)
    for label, name, ov in (("Bartlett: Rechteck, 0 % Überlappung", "Rechteck", 0.0), ("Welch: Hann, 50 % Überlappung", "Hann", 0.5),
                            ("Hann, 0 % Überlappung", "Hann", 0.0)):
        est, k = H.welch(x, seg, H.hop_for(seg, ov), W.window(name, seg), C.FS)
        rows.append({"label": label, "bands": _band_bias(f, est.mean(axis=0), true, LEAK_BANDS)})
    p0 = np.abs(np.fft.rfft(x, axis=1)) ** 2 / (C.FS * n)
    f_all = np.arange(n // 2 + 1) / n
    true_all = G.true_psd(f_all * C.FS, C.FS, poles, ar_sigma, white)
    for m in smooth:
        sm = H.smooth_periodogram(p0, m)
        rows.append({"label": f"Periodogramm der ganzen Aufzeichnung, {m} Bins gemittelt", "bands": _band_bias(f_all, sm.mean(axis=0), true_all, LEAK_BANDS)})
    return rows


def sweep_smoothing(segs=C.EXP_SMOOTH_SEGS, n: int = C.EXP_N, runs: int = C.EXP_LEAK_RUNS, background: str = C.EXP_TRADE_BACKGROUND) -> list[dict]:
    """Welch (Hann 50 %) gegen das über die ganze Aufzeichnung geglättete Periodogramm bei gleicher exakter Streuung (K_eff): Streuung, Fehler am Gipfel und
    Rauschbandbreite. Geglättet wird einmal mit Rechteck (m = K_eff) und einmal mit Hann (m so, dass K_eff stimmt)."""
    poles, ar_sigma, white = noise_spec(background)
    x = G.make_noise_batch(runs, n, poles, ar_sigma, white, G.SplitMix64(C.SWEEP_SEED + 5))
    f_peak = poles[0][0]
    f_all = np.fft.rfftfreq(n)
    true_all = G.true_psd(f_all * C.FS, C.FS, poles, ar_sigma, white)
    pk_all = int(np.argmin(np.abs(f_all - f_peak)))
    rows = []

    def summarise(label, est, true, sl, pk, k_eff, bw_hz):
        err = DB * np.log(est[:, sl] / true[sl])
        peak = DB * math.log(float(est[:, pk].mean()) / float(true[pk]))
        rows.append({"label": label, "k_eff": k_eff, "std_db": float(err.std(axis=0).mean()), "peak_db": peak, "rmse_db": float(math.sqrt((err ** 2).mean())),
                     "bandwidth_hz": bw_hz})

    for seg in segs:
        w = W.window("Hann", seg)
        hop = H.hop_for(seg, 0.5)
        est, k = H.welch(x, seg, hop, w, C.FS, C.SMOOTH_PAD * seg)           # 5-fach aufgefüllt: die Frequenz des Gipfels (0,2) liegt auf dem Raster
        k_eff = H.k_effective(w, hop, k)
        f = np.arange(est.shape[1]) / (C.SMOOTH_PAD * seg)
        true = G.true_psd(f * C.FS, C.FS, poles, ar_sigma, white)
        summarise(f"Welch, Segmentlänge {seg}, Hann, 50 %", est, true, slice(C.SMOOTH_PAD, est.shape[1] - C.SMOOTH_PAD), int(np.argmin(np.abs(f - f_peak))), k_eff,
                  W.enbw(w) * C.FS / seg)
        p0 = np.abs(np.fft.rfft(x, axis=1)) ** 2 / (C.FS * n)
        m0 = max(1, int(round(k_eff)))
        edge = slice(m0, len(f_all) - m0)
        summarise(f"Ganze Aufzeichnung, Rechteck, {m0} Bins gemittelt", H.smooth_periodogram(p0, m0), true_all, edge, pk_all, float(m0),
                  m0 * C.FS / n + W.enbw(np.ones(n)) * C.FS / n)
        wl = W.window("Hann", n)
        pl = np.abs(np.fft.rfft(x * wl, axis=1)) ** 2 / (C.FS * (wl ** 2).sum())
        m1 = min(range(1, 400), key=lambda mm: abs(H.smoothing_k_effective(wl, mm) - k_eff))
        edge1 = slice(m1, len(f_all) - m1)
        summarise(f"Ganze Aufzeichnung, Hann, {m1} Bins gemittelt", H.smooth_periodogram(pl, m1), true_all, edge1, pk_all, H.smoothing_k_effective(wl, m1),
                  m1 * C.FS / n + W.enbw(wl) * C.FS / n)
    return rows


def sweep_detect(ks=C.EXP_DET_K, snrs=C.EXP_DET_SNR, n: int = C.EXP_N, runs: int = C.EXP_DET_RUNS, q: float = C.EXP_DET_QUANTILE) -> list[dict]:
    """Eine Linie zwischen zwei Bins in weißem Rauschen, Hann 50 %: Anteil der Läufe, in denen das Maximum des Welch-Mittels die Schwelle (q-Quantil der Maxima
    ohne Linie) überschreitet, je Segmentzahl und SNR (je Abtastwert)."""
    noise = _white_batch(runs, n, C.SWEEP_SEED + 6)
    rng = G.SplitMix64(C.SWEEP_SEED + 7)
    ph = 2.0 * math.pi * rng.uniform(runs)
    t = np.arange(n)
    plans = []
    for k in ks:
        seg = n if k == 1 else (2 * n // (k + 1)) // 2 * 2
        hop = seg // 2
        est0, kk = H.welch(noise, seg, hop, W.window("Hann", seg), 1.0)
        thr = float(np.quantile(est0[:, _detect_slice(est0.shape[1])].max(axis=1), q))
        plans.append((seg, hop, kk, thr, (seg // 4 + 0.37) / seg))
    rows = []
    for snr in snrs:
        row = {"snr": snr, "runs": runs, "k": [p[2] for p in plans], "share": []}
        for seg, hop, kk, thr, f0 in plans:
            x = amplitude(snr) * np.cos(2.0 * np.pi * f0 * t[None, :] + ph[:, None]) + noise
            est, _ = H.welch(x, seg, hop, W.window("Hann", seg), 1.0)
            row["share"].append(float((est[:, _detect_slice(est.shape[1])].max(axis=1) > thr).mean()))
        rows.append(row)
    return rows


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    ratio = a.enbw_hz / a.full_enbw_hz
    base = (f"{a.k} Segmente der Länge {s.seg_eff} ({percent(s.overlap)} % Überlappung) ergeben etwa {de(a.k_eff, 1)} unabhängige Mittel: Die Streuung je Frequenz sinkt von "
            f"±{de(a.full_std_db, 1)} dB auf ±{de(a.std_db, 1)} dB, die Rauschbandbreite steigt von {de(a.full_enbw_hz, 2)} auf {de(a.enbw_hz, 2)} Hz (Faktor {de(ratio, 0)}).")
    if s.lines == "zwei":
        if a.resolved:
            return "success", base + " Die beiden Linien erscheinen im Mittel noch als **zwei Gipfel**."
        if a.full_resolved:
            return "warning", base + " Die beiden Linien **verschmelzen** im Mittel, das lange Periodogramm trennt sie."
        return "warning", base + " Die beiden Linien sind auch im langen Periodogramm **nicht getrennt**."
    return "info", base


def verdict_variance(rows: list[dict]) -> tuple[str, str]:
    r1, rl = rows[0], rows[-1]
    return "success", (f"Bei {r1['k']} Segment fällt der Variationskoeffizient je Bin nicht unter {de(r1['cv'], 2)}; bei {rl['k']} Segmenten liegt er bei {de(rl['cv'], 3)} "
                       f"(Theorie 1/√K = {de(rl['cv_theory'], 3)}), die Streuung des Logarithmus bei {de(rl['std_db'], 2)} dB statt {de(r1['std_db'], 2)} dB. "
                       f"**Mitteln senkt die Streuung wie 1/√K**; das 95-%-Band deckt in {de(100 * rl['coverage'], 1)} % der Fälle.")


def _overlap_row(rows, window, overlap):
    return [r for r in rows if r["window"] == window and r["overlap"] == overlap][0]


def verdict_overlap(rows: list[dict]) -> tuple[str, str]:
    hann, rect = _overlap_row(rows, "Hann", 0.5), _overlap_row(rows, "Rechteck", 0.5)
    h75 = _overlap_row(rows, "Hann", 0.75)
    return "info", (f"Mit 50 % Überlappung liefert Hann {de(hann['k_eff'], 1)} unabhängige Mittel (Faktor {de(hann['gain'], 2)} gegen ohne Überlappung), das Rechteck nur "
                    f"{de(rect['k_eff'], 1)} (Faktor {de(rect['gain'], 2)}); mehr als 62,5 % bringt bei Hann nichts mehr (75 %: {de(h75['k_eff'], 1)}). "
                    "**Überlappung holt zurück, was das Fenster an den Rändern verschenkt** (beim Rechteck gewinnt sie trotzdem).")


def verdict_tradeoff(rows: list[dict]) -> tuple[str, str]:
    best = min(rows, key=lambda r: r["rmse_db"])
    full = rows[-1]
    short = rows[0]
    return "warning", (f"Der Fehler hat ein Minimum: bei Segmentlänge {best['seg']} ({best['k']} Segmente) liegt er bei {de(best['rmse_db'], 2)} dB. Ein Periodogramm über alles "
                       f"streut mit {de(full['rmse_db'], 1)} dB, zu kurze Segmente ({short['seg']}) verwischen den Gipfel um {de(abs(short['peak_db']), 1)} dB. "
                       "**Es gibt kein „je mehr Segmente, desto besser“.**")


def verdict_two_lines(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["delta"] == 4.0][0]
    return "info", (f"Zwei Linien im Abstand von 4 Bins der Länge 256 erscheinen bei Segmentlänge 256 in {de(100 * r[256], 0)} % der Läufe als zwei Gipfel, bei 128 "
                    f"(2 Bins) in {de(100 * r[128], 0)} %, bei 64 (1 Bin) in {de(100 * r[64], 0)} %: **ab etwa 2,5 Bins der Segmentlänge zuverlässig getrennt, darunter fällt die Auflösung stufenlos ab.**")


def verdict_leakage(rows: list[dict]) -> tuple[str, str]:
    bart, welch = rows[0], rows[1]
    return "success", (f"Im Tal liegt Bartlett mit Rechteck-Segmenten um {de(bart['bands'][2], 1)} dB über der wahren Dichte, mit Hann-Segmenten um {de(welch['bands'][2], 1)} dB: "
                       "**Das Fenster je Segment senkt die Leckage, die Überlappung nicht.**")


def verdict_smoothing(rows: list[dict]) -> tuple[str, str]:
    w, rect, hann = rows[0], rows[1], rows[2]
    return "info", (f"Bei gleicher Streuung ({de(w['std_db'], 2)} dB) hat das mit Rechteck geglättete lange Periodogramm fast denselben Gipfelfehler ({de(rect['peak_db'], 1)} statt "
                    f"{de(w['peak_db'], 1)} dB); mit Hann braucht es {de(hann['bandwidth_hz'] / w['bandwidth_hz'], 1)}-fach so viel Bandbreite ({de(hann['peak_db'], 1)} dB). "
                    "**Bei mäßiger Dynamik ist Welch kaum besser als ein geglättetes Periodogramm**; bei hoher Dynamik (Steil) schon, weil das Rechteck leckt.")


def verdict_detect(rows: list[dict]) -> tuple[str, str]:
    r = [x for x in rows if x["snr"] == -20][0]
    sh = dict(zip(r["k"], r["share"]))
    best_k = max(sh, key=sh.get)
    return "warning", (f"Bei −20 dB je Abtastwert wird die Linie mit einem Periodogramm in {de(100 * sh[r['k'][0]], 0)} % der Läufe gefunden, mit {best_k} Segmenten in "
                       f"{de(100 * sh[best_k], 0)} %, mit {r['k'][-1]} Segmenten nur in {de(100 * sh[r['k'][-1]], 0)} %: **Mitteln hilft beim Finden einer Linie wenig "
                       "(gegen ein Rechteck-Periodogramm der ganzen Aufzeichnung höchstens einige Prozentpunkte), und zu viele Segmente schaden.**")
