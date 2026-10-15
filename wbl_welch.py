"""Kern: Segmentmittel (Bartlett, Welch), exakter Streufaktor, äquivalente Zahl unabhängiger Mittel, Konfidenzband, Glättung des Periodogramms.

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy. Digamma, Trigamma und die χ²-Quantile sind selbst gerechnet (die Tests prüfen sie gegen scipy).
"""

import math

import numpy as np

DB = 10.0 / math.log(10.0)


def n_segments(n: int, seg: int, hop: int) -> int:
    """Zahl der vollständigen Segmente der Länge seg im Abstand hop in einer Folge der Länge n."""
    if seg > n or hop < 1:
        raise ValueError("Segment länger als die Folge oder Schrittweite < 1")
    return (n - seg) // hop + 1


def hop_for(seg: int, overlap: float) -> int:
    """Schrittweite in Abtastwerten für den Überlappungsanteil overlap in [0, 1)."""
    return max(1, int(round(seg * (1.0 - overlap))))


def welch(x, seg: int, hop: int, w, fs: float, nfft: int | None = None):
    """Mittel der gefensterten Segment-Periodogramme (Leistungsdichte je Hz, zweiseitig, Bins 0 .. nfft / 2) und Zahl der Segmente.

    x darf mehrere Läufe haben (letzte Achse = Zeit). Skalierung je Segment: |X|^2 / (fs sum w^2); Rechteck ohne Überlappung ist Bartlett.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(w) != seg:
        raise ValueError("Fensterlänge ungleich Segmentlänge")
    n = x.shape[-1]
    k = n_segments(n, seg, hop)
    segs = np.lib.stride_tricks.sliding_window_view(x, seg, axis=-1)[..., ::hop, :]
    p = np.abs(np.fft.rfft(segs * w, nfft or seg, axis=-1)) ** 2 / (fs * (w ** 2).sum())
    return p.mean(axis=-2), k


def segment_correlation(w, hop: int, j: int) -> float:
    """Korrelation der Periodogramme zweier Segmente im Abstand j Schritten für weißes Gauß-Rauschen: (sum w_m w_{m + j hop})^2 / (sum w^2)^2."""
    w = np.asarray(w, dtype=float)
    sh = j * hop
    if sh >= len(w):
        return 0.0
    return float(((w[:len(w) - sh] * w[sh:]).sum() / (w ** 2).sum()) ** 2)


def variance_factor(w, hop: int, k: int) -> float:
    """Var(Mittel) / S^2 für weißes Gauß-Rauschen: (1 / K) (1 + 2 sum_{j<K} (1 - j / K) rho_j). Ohne Überlappung (Bartlett): 1 / K.

    Gilt für Frequenzen fern von 0 und der Nyquist-Frequenz; der Spiegelterm |E[X_a X_b]|^2 der reellen Folge ist weggelassen (bei Länge 16 unter 2e-4 relativ)."""
    total = 1.0
    for j in range(1, k):
        total += 2.0 * (1.0 - j / k) * segment_correlation(w, hop, j)
    return total / k


def k_effective(w, hop: int, k: int) -> float:
    """Äquivalente Zahl unabhängiger Mittel: 1 / Streufaktor (= 1 / CV^2)."""
    return 1.0 / variance_factor(w, hop, k)


def digamma(x: float) -> float:
    """Digamma-Funktion für x > 0 (Rekursion bis x >= 15, dann asymptotische Reihe)."""
    r = 0.0
    while x < 15.0:
        r -= 1.0 / x
        x += 1.0
    x2 = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - x2 * (1.0 / 12 - x2 * (1.0 / 120 - x2 * (1.0 / 252 - x2 * (1.0 / 240))))


def trigamma(x: float) -> float:
    """Trigamma-Funktion für x > 0."""
    r = 0.0
    while x < 15.0:
        r += 1.0 / (x * x)
        x += 1.0
    x2 = 1.0 / (x * x)
    return r + 1.0 / x + 0.5 * x2 + (1.0 / x) * x2 * (1.0 / 6 - x2 * (1.0 / 30 - x2 * (1.0 / 42 - x2 * (1.0 / 30))))


def log_bias_db(k: float) -> float:
    """Mittelwert von 10 log10 des Mittels von k Exponentialwerten gegen 10 log10 des Erwartungswerts: DB (psi(k) - ln k); k = 1: -2,51 dB."""
    return DB * (digamma(k) - math.log(k))


def log_std_db(k: float) -> float:
    """Standardabweichung von 10 log10 des Mittels: DB sqrt(psi'(k)); k = 1: 5,57 dB."""
    return DB * math.sqrt(trigamma(k))


def _gamma_p(a: float, x: float) -> float:
    """Regularisierte untere unvollständige Gammafunktion P(a, x)."""
    if x <= 0.0:
        return 0.0
    lead = math.exp(-x + a * math.log(x) - math.lgamma(a))
    if x < a + 1.0:
        term = total = 1.0 / a
        for i in range(1, 2000):
            term *= x / (a + i)
            total += term
            if term < total * 1e-16:
                break
        return total * lead
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - lead * h


def chi2_cdf(x: float, nu: float) -> float:
    return _gamma_p(0.5 * nu, 0.5 * x)


def chi2_ppf(p: float, nu: float) -> float:
    """Quantil der χ²-Verteilung mit nu (auch nicht ganzen) Freiheitsgraden, per Intervallhalbierung."""
    lo, hi = 0.0, nu + 12.0 * math.sqrt(2.0 * nu) + 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def confidence_factors(k_eff: float, level: float = 0.95) -> tuple[float, float]:
    """(untere, obere) Faktoren, mit denen das Mittel zu multiplizieren ist, damit [lo * est, hi * est] die wahre Dichte mit Wahrscheinlichkeit `level` deckt:
    est ~ S chi2_nu / nu mit nu = 2 k_eff, also S in [nu / q_hi, nu / q_lo] * est."""
    nu = 2.0 * k_eff
    a = 0.5 * (1.0 - level)
    return nu / chi2_ppf(1.0 - a, nu), nu / chi2_ppf(a, nu)


def smooth_periodogram(p, m: int) -> np.ndarray:
    """Gleitender Mittelwert über m benachbarte Bins (letzte Achse, Mitte, außerhalb null)."""
    p = np.atleast_2d(np.asarray(p, dtype=float))
    ker = np.full(m, 1.0 / m)
    return np.array([np.convolve(row, ker, mode="same") for row in p])


def smoothing_k_effective(w, m: int) -> float:
    """Äquivalente Zahl unabhängiger Mittel beim Mitteln von m benachbarten Bins eines mit w gefensterten Periodogramms der Länge len(w) (weißes Rauschen):
    m / (1 + 2 sum_{j<m} (1 - j / m) rho_j), rho_j = |DFT(w^2)_j|^2 / (sum w^2)^2."""
    w = np.asarray(w, dtype=float)
    rho = np.abs(np.fft.fft(w ** 2)) ** 2 / (w ** 2).sum() ** 2
    total = 1.0
    for j in range(1, m):
        total += 2.0 * (1.0 - j / m) * float(rho[j])
    return m / total


def ops_radix2(n: int):
    """Komplexe Multiplikationen einer Radix-2-FFT: (n / 2) log2 n; nur für Zweierpotenzen, sonst None."""
    if n < 2 or n & (n - 1):
        return None
    return n // 2 * int(round(math.log2(n)))


def ops_welch(seg: int, k: int):
    """Komplexe Multiplikationen aller Segment-FFTs (ohne Auffüllen): k mal Radix-2 der Länge seg."""
    o = ops_radix2(seg)
    return None if o is None else k * o
