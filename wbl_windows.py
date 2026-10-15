"""Fenster, Kennzahlen, exakter Erwartungswert des gefensterten Periodogramms und Gipfelzählung (aus Stück 2 der Reihe, auf vier Fenster gekürzt).

Reine Funktionen, kein Zufall, kein Streamlit. Fenster sind periodisch (DFT-gerade), wie sie die Spektralanalyse braucht.
"""

import math

import numpy as np

NAMES = ("Rechteck", "Hann", "Hamming", "Blackman")
COSINE_SUM = {
    "Hann": (0.5, 0.5),
    "Hamming": (0.54, 0.46),
    "Blackman": (0.42, 0.5, 0.08),
}


def window(name: str, n: int) -> np.ndarray:
    """Periodisches Fenster der Länge n."""
    if name == "Rechteck":
        return np.ones(n)
    if name in COSINE_SUM:
        m = np.arange(n)
        w = np.zeros(n)
        for k, a in enumerate(COSINE_SUM[name]):
            w += (-1) ** k * a * np.cos(2.0 * np.pi * k * m / n)
        return w
    raise ValueError(f"unbekanntes Fenster: {name}")


def enbw(w) -> float:
    """Äquivalente Rauschbandbreite in Bins: n sum(w^2) / (sum w)^2."""
    w = np.asarray(w, dtype=float)
    return float(len(w) * (w ** 2).sum() / w.sum() ** 2)


def expected_noise_spectrum(f_hz, fs: float, w, r) -> np.ndarray:
    """E[gefenstertes Periodogramm eines Segments] für Rauschen mit Autokovarianz r (Lags 0 .. len(w) - 1), Leistungsdichte (Skala 1 / (fs sum w^2)).

    E|X|^2 = sum_tau r(tau) c_w(tau) cos(w tau), beide Vorzeichen von tau, mit der Fensterautokorrelation c_w(tau) = sum w_m w_{m+tau}.
    """
    w = np.asarray(w, dtype=float)
    r = np.asarray(r, dtype=float)
    n = len(w)
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    om = 2.0 * np.pi * f / fs
    c = np.correlate(w, w, mode="full")[n - 1:]
    tau = np.arange(n)
    wts = r[:n] * c
    out = np.empty(len(f))
    for lo in range(0, len(f), 256):
        cs = np.cos(np.outer(om[lo:lo + 256], tau))
        out[lo:lo + 256] = wts[0] + 2.0 * (cs[:, 1:] @ wts[1:])
    return out / (fs * (w ** 2).sum())


def count_peaks(y, rel_prominence: float, reference: float | None = None) -> int:
    """Zahl der Gipfel mit Prominenz >= rel_prominence * reference (Standard: Maximum von y). Gipfel sind innere Punkte, die höher als beide Nachbarn sind
    (bei Plateaus der erste Punkt); die Ränder zählen nie.

    Prominenz wie bei scipy.signal.find_peaks ohne Fensterlänge: Höhe über der höheren der beiden Basen; Basis = tiefster Wert bis zum nächsten höheren
    Punkt auf dieser Seite, ohne einen solchen bis zum Rand.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0
    ref = float(y.max()) if reference is None else float(reference)
    count = 0
    for i in range(1, len(y) - 1):
        if not (y[i] > y[i - 1] and y[i] >= y[i + 1]):
            continue
        j = i - 1
        while j > 0 and y[j] <= y[i]:
            j -= 1
        left = y[j:i + 1].min() if y[j] > y[i] else y[0:i + 1].min()
        k = i + 1
        while k < len(y) - 1 and y[k] <= y[i]:
            k += 1
        right = y[i:k + 1].min() if y[k] > y[i] else y[i:].min()
        if y[i] - max(left, right) >= rel_prominence * ref:
            count += 1
    return count


def two_lines_resolved(spec, pad: int, f1_bins: float, f2_bins: float, rel_prominence: float = 0.1) -> bool:
    """Zwei Linien gelten als aufgelöst, wenn das (aufgefüllte) Spektrum zwischen f1 - d/2 und f2 + d/2 mindestens zwei Gipfel mit Prominenz >= 10 % des Maximums hat."""
    d = abs(f2_bins - f1_bins)
    lo = max(int(math.floor((min(f1_bins, f2_bins) - d / 2) * pad)), 0)
    hi = int(math.ceil((max(f1_bins, f2_bins) + d / 2) * pad)) + 1
    return count_peaks(np.asarray(spec)[lo:hi], rel_prominence, reference=float(np.max(spec))) >= 2
