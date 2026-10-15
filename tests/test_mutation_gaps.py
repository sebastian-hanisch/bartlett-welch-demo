"""Gezielte Tests für Stellen, die der Mutationstest als ungeprüft meldete (Grenzen, Randfälle, unabhängige Referenzfassungen)."""

import numpy as np
import pytest

import wbl_evaluation as ev
import wbl_welch as H
import wbl_windows as W


def test_variance_factor_with_two_overlapping_segments_keeps_the_correlation_term(hann4):
    # n = 6, L = 4, Schrittweite 2: K = 2, Faktor (1 / 2) (1 + 2 (1 - 1 / 2) rho_1) mit rho_1 = 1 / 36
    assert H.n_segments(6, 4, 2) == 2
    assert H.variance_factor(hann4, 2, 2) == pytest.approx(0.5 * (1.0 + 2.0 * 0.5 / 36.0), abs=1e-15)


def test_variance_factor_matches_an_independent_loop_for_a_window_with_long_correlation():
    w = np.array([1.0, 0.7, 0.2, 0.9, 0.4, 0.8, 0.3, 0.6])
    for hop in (1, 2, 3):
        k = H.n_segments(40, 8, hop)
        s2 = (w ** 2).sum()
        total = 0.0
        for a in range(k):
            for b in range(k):
                sh = abs(a - b) * hop
                rho = 1.0 if sh == 0 else (((w[:8 - sh] * w[sh:]).sum() / s2) ** 2 if sh < 8 else 0.0)
                total += rho
        assert H.variance_factor(w, hop, k) == pytest.approx(total / k ** 2, rel=1e-12)


def test_segment_correlation_vanishes_when_the_shift_reaches_the_window_length():
    w = np.array([1.0, 2.0, 3.0, 4.0])
    assert H.segment_correlation(w, 2, 2) == 0.0 and H.segment_correlation(w, 4, 1) == 0.0
    assert H.segment_correlation(w, 3, 1) == pytest.approx((4.0 / 30.0) ** 2, abs=1e-15)      # nur w_0 w_3 = 4


def test_smoothing_effective_number_matches_an_independent_reference_with_long_correlation():
    w = np.zeros(16)
    w[0], w[-1] = 1.0, 0.5            # w^2 = (1, 0, ..., 0, 0,25): alle Nachbarkorrelationen sind nicht null
    s2 = (w ** 2).sum()
    for m in (1, 2, 5, 9):
        rho = [abs(sum(w[i] ** 2 * np.exp(-2j * np.pi * j * i / 16) for i in range(16))) ** 2 / s2 ** 2 for j in range(m)]
        ref = m / (1.0 + 2.0 * sum((1.0 - j / m) * rho[j] for j in range(1, m)))
        assert H.smoothing_k_effective(w, m) == pytest.approx(ref, rel=1e-12)


def test_chi2_quantile_inverts_the_distribution_function_to_rounding():
    for nu in (2.0, 800.0):
        q = H.chi2_ppf(0.975, nu)
        assert H.chi2_cdf(q, nu) == pytest.approx(0.975, abs=1e-14)


def test_gamma_series_and_continued_fraction_agree_at_the_switch_point():
    a = 7.0
    below = H._gamma_p(a, a + 1.0 - 1e-9)
    above = H._gamma_p(a, a + 1.0 + 1e-9)
    assert above - below == pytest.approx(0.0, abs=1e-6) and 0.0 < below < above < 1.0
    assert H._gamma_p(a, a + 1.0) == pytest.approx(0.5 * (below + above), abs=1e-8)


def test_band_bias_includes_the_lower_band_edge_and_the_upper_one():
    f = np.array([0.1, 0.2, 0.3, 0.4])
    est = np.array([1.0, 10.0, 100.0, 1000.0])
    true = np.ones(4)
    out = ev._band_bias(f, est, true, ((0.2, 0.3), (0.1, 0.1), (0.35, 0.45)))
    assert out == pytest.approx([15.0, 0.0, 30.0], abs=1e-9)


def test_detect_slice_skips_two_bins_at_each_end():
    assert ev._detect_slice(10) == slice(2, 8)
    assert list(range(10))[ev._detect_slice(10)] == [2, 3, 4, 5, 6, 7]


def test_ops_radix2_only_for_powers_of_two():
    for n in (3, 5, 6, 7, 9, 10, 12, 14, 24, 100, 1000):
        assert H.ops_radix2(n) is None
    for n in (2, 4, 8, 1024):
        assert H.ops_radix2(n) is not None
