"""Randfälle: kleinste und größte Länge, Segment gleich Aufzeichnung, höchste Überlappung, alle Fenster und Hintergründe, Linien an den Reglergrenzen."""

import numpy as np
import pytest

import wbl_constants as C
import wbl_evaluation as ev


@pytest.mark.parametrize("window", C.WINDOWS)
@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_window_and_background_runs_with_two_lines(window, background):
    a = ev.analyse(ev.Settings(window=window, background=background, n=1024, seg=64, lines="zwei"))
    assert np.isfinite(a.est).all() and np.isfinite(a.expected).all() and (a.expected > 0).all() and (a.band_hi >= a.band_lo).all()


def test_segment_equal_to_the_record_is_one_periodogram():
    a = ev.analyse(ev.Settings(n=1024, seg=1024, overlap=0.875))
    assert a.k == 1 and a.k_eff == pytest.approx(1.0) and a.std_db == pytest.approx(a.full_std_db)
    assert a.est[::C.PAD] == pytest.approx(a.full, rel=1e-9)


def test_segment_longer_than_the_record_is_cut_to_n():
    s = ev.Settings(n=1024, seg=4096)
    assert s.seg_eff == 1024 and ev.analyse(s).k == 1


def test_shortest_segment_with_the_highest_overlap_of_the_longest_record():
    a = ev.analyse(ev.Settings(n=16384, seg=32, overlap=0.875))
    assert a.k == (16384 - 32) // 4 + 1 and np.isfinite(a.est).all()
    assert a.ops == a.k * 16 * 5


@pytest.mark.parametrize("line", [C.LINE_MIN, C.LINE_MAX])
@pytest.mark.parametrize("gap", [C.GAP_MIN, C.GAP_MAX])
def test_lines_at_the_slider_limits_stay_below_nyquist(line, gap):
    s = ev.Settings(line=line, gap=gap, lines="zwei")
    assert 0 < s.f1_hz < s.f2_hz < C.FS / 2
    assert np.isfinite(ev.analyse(s).est).all()


def test_extreme_levels_and_seed_limits():
    for rel in (C.REL_MIN, C.REL_MAX):
        for snr in (C.SNR_MIN, C.SNR_MAX):
            assert np.isfinite(ev.analyse(ev.Settings(lines="zwei", rel=rel, snr=snr)).est).all()
    for seed in (0, C.SEED_MAX):
        assert np.isfinite(ev.analyse(ev.Settings(seed=seed)).est).all()


def test_no_lines_means_no_line_resolution_and_the_expected_value_is_the_noise_term():
    a = ev.analyse(ev.Settings(lines="keine"))
    assert a.resolved is None and a.full_resolved is None
    a1 = ev.analyse(ev.Settings(lines="eine"))
    assert a1.resolved is None and a1.expected.max() > a.expected.max()


def test_peak_bias_exists_only_with_a_resonance():
    assert ev.analyse(ev.Settings(background="weiß")).peak_bias_db is None
    assert ev.analyse(ev.Settings(background="Resonanz")).peak_bias_db < 0
