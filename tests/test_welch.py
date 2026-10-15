"""Kern: Segmentierung, Mittel, Streufaktor, Logarithmus-Bias, χ²-Band, Glättung, Rechenaufwand; Mini-Instanz von Hand."""

import math

import numpy as np
import pytest

import wbl_welch as H
import wbl_windows as W


def test_segment_counts():
    assert H.n_segments(8, 4, 2) == 3 and H.n_segments(8, 4, 4) == 2 and H.n_segments(4, 4, 1) == 1
    assert H.n_segments(7, 4, 2) == 2                              # ein Rest von einem Messwert bleibt ungenutzt
    with pytest.raises(ValueError):
        H.n_segments(3, 4, 1)
    with pytest.raises(ValueError):
        H.n_segments(8, 4, 0)


def test_hop_for_overlap():
    assert H.hop_for(256, 0.0) == 256 and H.hop_for(256, 0.5) == 128 and H.hop_for(256, 0.625) == 96 and H.hop_for(256, 0.875) == 32
    assert H.hop_for(32, 0.99) == 1


def test_welch_of_a_constant_by_hand(hann4):
    # Segment (1,1,1,1), Hann (0; 0,5; 1; 0,5): X_0 = 2, X_1 = -1, X_2 = 0; Summe der Quadrate 1,5, fs = 1
    est, k = H.welch(np.ones(8), 4, 2, hann4, 1.0)
    assert k == 3
    assert est == pytest.approx([4.0 / 1.5, 1.0 / 1.5, 0.0], abs=1e-12)


def test_welch_averages_the_segment_periodograms_independently_computed(hann4):
    x = np.array([1.0, -2.0, 0.5, 3.0, 2.0, -1.0, 0.0, 4.0])
    est, k = H.welch(x, 4, 2, hann4, 2.0)
    ref = np.zeros(3)
    for start in (0, 2, 4):
        ref += np.abs(np.fft.fft(x[start:start + 4] * hann4)[:3]) ** 2 / (2.0 * 1.5)
    assert k == 3 and est == pytest.approx(ref / 3, abs=1e-12)


def test_welch_zero_padding_and_run_axis(hann4):
    x = np.arange(16, dtype=float).reshape(2, 8)
    est, k = H.welch(x, 4, 2, hann4, 1.0, nfft=16)
    assert est.shape == (2, 9) and k == 3
    single, _ = H.welch(x[1], 4, 2, hann4, 1.0, nfft=16)
    assert est[1] == pytest.approx(single, abs=1e-12)


def test_welch_rejects_a_window_of_the_wrong_length(hann4):
    with pytest.raises(ValueError):
        H.welch(np.ones(8), 8, 4, hann4, 1.0)


def test_segment_correlation_of_hann_with_half_overlap(hann4):
    assert H.segment_correlation(hann4, 2, 1) == pytest.approx(1.0 / 36.0, abs=1e-15)      # (0,25 / 1,5)^2
    assert H.segment_correlation(hann4, 2, 2) == 0.0                                        # Verschiebung 4 >= L
    assert H.segment_correlation(hann4, 1, 1) == pytest.approx(((0 * 0.5 + 0.5 * 1 + 1 * 0.5) / 1.5) ** 2, abs=1e-15)
    assert H.segment_correlation(np.ones(4), 2, 1) == pytest.approx(0.25, abs=1e-15)       # Rechteck, 50 %: (2 / 4)^2


def test_variance_factor_and_effective_number_by_hand(hann4):
    f = H.variance_factor(hann4, 2, 3)
    assert f == pytest.approx((1.0 / 3.0) * (1.0 + 2.0 * (2.0 / 3.0) / 36.0), abs=1e-15)
    assert H.k_effective(hann4, 2, 3) == pytest.approx(2.892857142857143, abs=1e-12)
    assert H.variance_factor(np.ones(4), 4, 2) == 0.5 and H.k_effective(np.ones(4), 4, 2) == 2.0     # Bartlett: 1 / K
    assert H.variance_factor(hann4, 2, 1) == 1.0


def test_digamma_trigamma_and_log_moments():
    assert H.digamma(1.0) == pytest.approx(-0.5772156649015329, abs=1e-10)
    assert H.digamma(0.5) == pytest.approx(-1.9635100260214235, abs=1e-10)
    assert H.trigamma(1.0) == pytest.approx(math.pi ** 2 / 6.0, abs=1e-10)
    assert H.log_bias_db(1.0) == pytest.approx(-2.5068, abs=2e-4)
    assert H.log_std_db(1.0) == pytest.approx(5.5700, abs=1e-4)
    assert H.log_bias_db(2.0) == pytest.approx(10.0 / math.log(10.0) * (1.0 - 0.5772156649 - math.log(2.0)), abs=1e-8)
    assert H.log_bias_db(64.0) > H.log_bias_db(1.0) and H.log_std_db(64.0) < H.log_std_db(1.0)


def test_chi2_quantiles_for_two_degrees_of_freedom_are_exponential():
    assert H.chi2_cdf(2.0, 2.0) == pytest.approx(1.0 - math.exp(-1.0), abs=1e-12)
    assert H.chi2_ppf(0.95, 2.0) == pytest.approx(-2.0 * math.log(0.05), abs=1e-9)
    assert H.chi2_ppf(0.5, 2.0) == pytest.approx(2.0 * math.log(2.0), abs=1e-9)
    assert H.chi2_cdf(0.0, 4.0) == 0.0 and H.chi2_cdf(-1.0, 4.0) == 0.0


def test_confidence_factors_for_one_effective_segment():
    lo, hi = H.confidence_factors(1.0)
    assert lo == pytest.approx(2.0 / (-2.0 * math.log(0.025)), abs=1e-9)
    assert hi == pytest.approx(2.0 / (-2.0 * math.log(0.975)), abs=1e-7)
    lo90, hi90 = H.confidence_factors(1.0, 0.90)
    assert lo < lo90 < 1.0 < hi90 < hi
    lo_k, hi_k = H.confidence_factors(50.0)
    assert 1.0 - lo_k < 1.0 - lo and hi_k - 1.0 < hi - 1.0         # mehr Mittel, schmaleres Band


def test_smooth_periodogram_edges_and_identity():
    out = H.smooth_periodogram([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    assert out.shape == (1, 5) and out[0] == pytest.approx([1.0, 2.0, 3.0, 4.0, 3.0], abs=1e-12)
    assert H.smooth_periodogram([[1.0, 5.0], [2.0, 4.0]], 1) == pytest.approx(np.array([[1.0, 5.0], [2.0, 4.0]]))


def test_smoothing_effective_number():
    assert H.smoothing_k_effective(np.ones(64), 7) == pytest.approx(7.0, abs=1e-12)           # Rechteck: Nachbarbins unkorreliert
    assert H.smoothing_k_effective(np.ones(64), 1) == 1.0
    hann = W.window("Hann", 64)
    m = H.smoothing_k_effective(hann, 10)
    assert 4.0 < m < 7.0                                                                       # Nachbarbins der Hann-Periodogramme sind korreliert
    rho = np.abs(np.fft.fft(hann ** 2)) ** 2 / (hann ** 2).sum() ** 2
    assert rho[1] == pytest.approx(0.4444, abs=1e-3) and rho[2] == pytest.approx(0.0278, abs=1e-3)       # (2/3)^2 und (1/6)^2


def test_operation_counts():
    assert H.ops_radix2(256) == 128 * 8 and H.ops_radix2(2) == 1 and H.ops_radix2(100) is None and H.ops_radix2(1) is None
    assert H.ops_welch(256, 31) == 31 * 128 * 8 and H.ops_welch(100, 3) is None
