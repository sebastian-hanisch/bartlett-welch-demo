"""Orakel (scipy und unabhängige Referenzen): Welch-Schätzer, Streufaktor als quadratische Form, Erwartungswert, Verteilungsfunktionen, Gipfelzählung."""

import math

import numpy as np
import pytest
from scipy import signal as sps
from scipy import special, stats

import signal_generator as G
import wbl_constants as C
import wbl_evaluation as ev
import wbl_welch as H
import wbl_windows as W


@pytest.mark.parametrize("name", C.WINDOWS)
@pytest.mark.parametrize("overlap", [0.0, 0.5, 0.75])
@pytest.mark.parametrize("n,seg", [(1000, 128), (777, 64)])
def test_welch_equals_scipy_welch(name, overlap, n, seg):
    x = G.SplitMix64(5).normal(n)
    w = W.window(name, seg)
    hop = H.hop_for(seg, overlap)
    est, k = H.welch(x, seg, hop, w, 2.0)
    f, p = sps.welch(x, fs=2.0, window=w, nperseg=seg, noverlap=seg - hop, detrend=False, scaling="density", return_onesided=True)
    p = p.copy()
    p[1:-1] /= 2.0                                      # scipy: einseitig (verdoppelt), hier zweiseitig
    assert est == pytest.approx(p, rel=1e-10, abs=1e-14) and k == (n - seg) // hop + 1


def test_windows_equal_scipy_periodic_windows():
    for name, ref in (("Rechteck", "boxcar"), ("Hann", "hann"), ("Hamming", "hamming"), ("Blackman", "blackman")):
        for n in (8, 31, 256):
            assert W.window(name, n) == pytest.approx(sps.get_window(ref, n, fftbins=True), abs=1e-12)


def test_digamma_trigamma_equal_scipy():
    # Toleranz 1e-13 (die abgebrochene Reihe ab x = 15 fehlt höchstens um 1,5e-14): fängt jede Änderung eines Reihenkoeffizienten bis zum x^-7-Glied
    for x in (0.7, 1.0, 1.5, 2.0, 5.5, 14.5, 14.99, 15.0, 15.01, 15.5, 20.3, 100.0, 1000.0):
        assert H.digamma(x) == pytest.approx(float(special.digamma(x)), abs=1e-13)
        assert H.trigamma(x) == pytest.approx(float(special.polygamma(1, x)), abs=1e-13)


def test_chi2_quantiles_and_distribution_equal_scipy():
    for nu in (2.0, 3.5, 16.0, 119.4, 600.0):
        for p in (0.025, 0.5, 0.975):
            assert H.chi2_ppf(p, nu) == pytest.approx(float(stats.chi2.ppf(p, nu)), rel=1e-9)
        for x in (0.5 * nu, nu, 1.5 * nu):
            assert H.chi2_cdf(x, nu) == pytest.approx(float(stats.chi2.cdf(x, nu)), abs=1e-10)


def _quadratic_form_variance(w, hop, k, n, omega):
    """Var des Mittels der Segment-Periodogramme bei weißem Gauß-Rauschen (Varianz 1) als exakte quadratische Form: x^T Q x mit x ~ N(0, I) hat Varianz 2 tr(Q^2)."""
    seg = len(w)
    q = np.zeros((n, n))
    for j in range(k):
        v = np.zeros(n, dtype=complex)
        v[j * hop:j * hop + seg] = w * np.exp(-1j * omega * np.arange(seg))
        q += np.real(np.outer(v, v.conj()))
    q /= k * (w ** 2).sum()
    return 2.0 * float(np.trace(q @ q))


@pytest.mark.parametrize("name,hop,seg,n", [("Hann", 8, 16, 64), ("Rechteck", 8, 16, 64), ("Hamming", 4, 16, 48), ("Blackman", 12, 16, 64), ("Hann", 16, 16, 64)])
def test_variance_factor_equals_the_exact_quadratic_form_at_inner_frequencies(name, hop, seg, n):
    w = W.window(name, seg)
    k = H.n_segments(n, seg, hop)
    omega = 2.0 * math.pi * 3 / seg                      # Bin 3 von 8: weder 0 noch Nyquist
    exact = _quadratic_form_variance(w, hop, k, n, omega)
    # Die Formel lässt den Spiegelterm |E[X_a X_b]|^2 weg (nur nahe 0 und Nyquist spürbar): bei Länge 16 und Bin 3 höchstens 2e-4 relativ
    assert H.variance_factor(w, hop, k) == pytest.approx(exact, rel=5e-4)


def test_variance_factor_equals_a_large_simulation_within_three_standard_errors():
    n, seg, hop, runs = 64, 16, 8, 40000
    w = W.window("Hann", seg)
    x = G.SplitMix64(11).normal(runs * n).reshape(runs, n)
    est, k = H.welch(x, seg, hop, w, 1.0)
    cv2 = (est[:, 3].var() / est[:, 3].mean() ** 2)
    f = H.variance_factor(w, hop, k)
    assert abs(cv2 - f) < 3.0 * f * math.sqrt(2.0 / runs) * 2.0      # Standardfehler einer Varianz: sqrt(2 / runs) mal Wert, großzügig


@pytest.mark.parametrize("background", ["weiß", "Resonanz", "Steil"])
def test_expected_noise_spectrum_equals_the_simulated_mean(background):
    seg, hop, runs = 32, 16, 6000
    poles, ar_sigma, white = ev.noise_spec(background)
    w = W.window("Hann", seg)
    x = G.make_noise_batch(runs, 256, poles, ar_sigma, white, G.SplitMix64(21))
    est, k = H.welch(x, seg, hop, w, C.FS)
    f = np.arange(seg // 2 + 1) * C.FS / seg
    exp = W.expected_noise_spectrum(f, C.FS, w, G.noise_autocov(poles, ar_sigma, white, seg))
    se = est.std(axis=0) / math.sqrt(runs)
    inner = slice(1, seg // 2)
    assert np.all(np.abs(est.mean(axis=0)[inner] - exp[inner]) < 4.5 * se[inner])


def test_expected_value_with_lines_equals_the_simulated_mean():
    s = ev.Settings(n=1024, seg=64, overlap=0.5, lines="zwei", gap=8.0, snr=0, rel=-6, background="Resonanz", seed=9)
    a = ev.analyse(s)
    poles, ar_sigma, white = ev.noise_spec(s.background)
    sig, _ = ev.build_signal(s)
    runs = 3000
    x = sig.clean[None, :] + G.make_noise_batch(runs, s.n, poles, ar_sigma, white, G.SplitMix64(33))
    est, _ = H.welch(x, s.seg_eff, s.hop, a.w, C.FS, C.PAD * s.seg_eff)
    se = est.std(axis=0) / math.sqrt(runs)
    inner = slice(8, len(est[0]) - 8)
    assert np.all(np.abs(est.mean(axis=0)[inner] - a.expected[inner]) < 4.5 * se[inner] + 1e-12)


def test_count_peaks_equals_scipy_find_peaks():
    rng = np.random.default_rng(3)
    for _ in range(60):
        y = np.abs(np.convolve(rng.normal(size=80), np.ones(5) / 5, mode="same"))
        thr = 0.1 * y.max()
        ref = len(sps.find_peaks(y, prominence=thr)[0])
        assert W.count_peaks(y, 0.1) == ref
