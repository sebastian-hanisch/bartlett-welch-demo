"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Experiment-Überschriften) ist hier an Rechnung oder Messung gebunden."""

import math

import numpy as np
import pytest

import wbl_constants as C
import wbl_evaluation as ev
import wbl_welch as H
import wbl_windows as W


def preset(name, **over):
    return ev.Settings(**{**C.PRESETS[name], **over})


def test_calm_noise_preset_numbers():
    a = ev.analyse(preset("Rauschen beruhigen"))
    assert a.k == 32 and a.k_eff == pytest.approx(32.0) and round(a.full_std_db, 1) == 5.6 and round(a.std_db, 1) == 0.8


def test_resonance_preset_is_the_optimum_of_the_tradeoff_experiment():
    a = ev.analyse(preset("Resonanz: das Optimum"))
    best = min(ev.sweep_tradeoff(), key=lambda r: r["rmse_db"])
    assert a.k == 63 and a.settings.seg_eff == best["seg"] == 128
    assert round(abs(a.peak_bias_db), 1) == 1.7 and a.peak_bias_db < 0


def test_two_lines_preset_merge_in_the_mean_and_separate_in_the_long_periodogram():
    s = preset("Zwei Linien: Schärfe geht verloren")
    assert s.gap * s.seg_eff / C.GAP_REF == 1.5                                     # 1,5 Bins der Segmentlänge
    a = ev.analyse(s)
    assert a.resolved is False and a.full_resolved is True


def test_steep_preset_valley_is_filled_by_about_fifteen_decibels_with_the_rectangle_and_not_with_hann():
    s = preset("Steiler Hintergrund: Leckage")
    a = ev.analyse(s)
    valley = a.f_hz >= 450.0
    assert 10 * math.log10(float(np.mean(a.expected[valley] / a.true_density[valley]))) == pytest.approx(14.7, abs=0.2)
    h = ev.analyse(ev.Settings(**{**C.PRESETS["Steiler Hintergrund: Leckage"], "window": "Hann", "overlap": 0.5}))
    assert abs(10 * math.log10(float(np.mean(h.expected[valley] / h.true_density[valley])))) < 0.05


def test_weak_line_preset_rises_about_six_decibels_and_shorter_segments_lower_it():
    s = preset("Schwache Linie")
    assert ev.analyse(s).k == 7 and s.lines == "eine" and s.snr == -20

    def rise(seg):
        a = ev.analyse(ev.Settings(**{**C.PRESETS["Schwache Linie"], "seg": seg}))
        return 10 * math.log10(float(np.max(a.est / a.true_density)))
    assert rise(1024) == pytest.approx(6.5, abs=0.3)
    assert rise(1024) > rise(256) > rise(64)


def test_first_screen_claims_about_the_window_and_resolution():
    # „Hauptkeule von rund 1,5 Bins" (Hann, Rauschbandbreite) und „ab etwa 2,5 Bins getrennt" (Hann 100 % ab 2,5 Bins der Segmentlänge)
    assert W.enbw(W.window("Hann", 256)) == pytest.approx(1.5, abs=0.01)
    rows = ev.sweep_two_lines()
    r25 = [r for r in rows if r["delta"] == 2.5][0]
    assert r25[256] == 1.0 and r25[128] == 0.0                                       # 2,5 Bins getrennt, 1,25 Bins nicht
    r4 = [r for r in rows if r["delta"] == 4.0][0]
    assert 0.5 < r4[128] < 0.95 and r4[64] == 0.0                                    # 2 Bins: „nicht zuverlässig"


def test_averaging_lowers_the_scatter_as_one_over_root_k():
    rows = ev.sweep_variance()
    for r in rows:
        assert r["cv"] == pytest.approx(r["cv_theory"], rel=0.05)
        assert r["std_db"] == pytest.approx(r["std_theory"], rel=0.02)
        assert 0.935 < r["coverage"] < 0.965
    assert rows[0]["cv"] > 0.95 and rows[-1]["cv"] < 0.14


def test_hann_with_half_overlap_gets_nearly_twice_the_independent_means_and_the_rectangle_one_third_more():
    rows = ev.sweep_overlap()
    gain = {(r["window"], r["overlap"]): r["gain"] for r in rows}
    assert 1.7 < gain[("Hann", 0.5)] < 2.0 and 1.2 < gain[("Rechteck", 0.5)] < 1.45
    assert gain[("Hann", 0.75)] == pytest.approx(gain[("Hann", 0.625)], abs=0.1) and gain[("Hann", 0.875)] < 2.1
    assert gain[("Rechteck", 0.0)] == pytest.approx(1.0, abs=0.03)
    assert H.k_effective(W.window("Hann", 256), 128, 31) == pytest.approx(29.418, abs=1e-3)


def test_rectangle_segments_leak_into_the_valley_and_hann_segments_do_not():
    rows = ev.sweep_leakage()
    assert rows[0]["bands"][2] > 14.0 and abs(rows[1]["bands"][2]) < 0.1 and abs(rows[2]["bands"][2]) < 0.1
    assert rows[0]["bands"][2] > rows[3]["bands"][2] > 4.0                           # das lange Rechteck-Periodogramm leckt weniger, aber merklich


def test_welch_is_a_smoothing_at_equal_scatter():
    rows = ev.sweep_smoothing()
    for i in (0, 3, 6):
        w, rect, hann = rows[i], rows[i + 1], rows[i + 2]
        assert w["std_db"] == pytest.approx(rect["std_db"], abs=0.05) and w["std_db"] == pytest.approx(hann["std_db"], abs=0.06)
        assert w["peak_db"] == pytest.approx(rect["peak_db"], abs=0.2)
        assert hann["bandwidth_hz"] > 1.9 * w["bandwidth_hz"] and hann["peak_db"] < rect["peak_db"] - 0.2


def test_averaging_helps_finding_a_line_only_a_little_and_too_many_segments_hurt():
    r = [x for x in ev.sweep_detect() if x["snr"] == -20][0]
    share = dict(zip(r["k"], r["share"]))
    assert max(share.values()) < share[1] + 0.25 and share[2] > share[1] and share[32] < share[1] - 0.15
    assert share[16] < share[8] and r["runs"] >= 500


def test_detection_baseline_with_the_rectangle_is_about_sixty_one_percent_and_averaging_gains_only_a_few_points():
    """README: Die Basis K = 1 ist ein Hann-Periodogramm; ein Rechteck-Periodogramm der ganzen Aufzeichnung findet die Linie in rund 61 % der Läufe (gleicher Aufbau)."""
    n, runs = C.EXP_N, C.EXP_DET_RUNS
    noise = ev._white_batch(runs, n, C.SWEEP_SEED + 6)
    ph = 2.0 * math.pi * ev.G.SplitMix64(C.SWEEP_SEED + 7).uniform(runs)
    t = np.arange(n)
    w = W.window("Rechteck", n)
    est0, _ = H.welch(noise, n, n // 2, w, 1.0)
    sl = ev._detect_slice(est0.shape[1])
    thr = float(np.quantile(est0[:, sl].max(axis=1), C.EXP_DET_QUANTILE))
    x = ev.amplitude(-20) * np.cos(2.0 * np.pi * ((n // 4 + 0.37) / n) * t[None, :] + ph[:, None]) + noise
    est, _ = H.welch(x, n, n // 2, w, 1.0)
    rect_share = float((est[:, sl].max(axis=1) > thr).mean())
    assert 100 * rect_share == pytest.approx(61, abs=2)
    share = dict(zip([x for x in ev.sweep_detect() if x["snr"] == -20][0]["k"], [x for x in ev.sweep_detect() if x["snr"] == -20][0]["share"]))
    assert max(share.values()) - rect_share < 0.10 and share[1] < rect_share - 0.08


def test_smoothing_with_steep_background_welch_beats_the_smoothed_rectangle_periodogram():
    """README: bei hoher Dynamik (Steil) Welch 1,08 / 0,88 / 1,18 dB gegen 2,37 / 2,29 / 2,41 dB beim geglätteten Rechteck-Periodogramm."""
    rows = ev.sweep_smoothing(background="Steil")
    welch = [rows[i]["rmse_db"] for i in (0, 3, 6)]
    rect = [rows[i]["rmse_db"] for i in (1, 4, 7)]
    assert [round(v, 2) for v in welch] == pytest.approx([1.08, 0.88, 1.18], abs=0.03)
    assert [round(v, 2) for v in rect] == pytest.approx([2.37, 2.29, 2.41], abs=0.03)
    assert all(r > 1.9 * w for r, w in zip(rect[:2], welch[:2]))


# --- README: die Zahlen der Befunde-Tabelle stehen im Text und stimmen mit den Messungen überein --------------------------------------------------


def _readme():
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def _row(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())][0]


def test_readme_variance_numbers():
    rows = ev.sweep_variance()
    pick = [r for r in rows if r["k"] in (1, 4, 16, 64)]
    text = _readme()
    assert " / ".join(ev.de(r["cv"], 2) for r in pick) in text
    assert ev.de(pick[0]["std_db"], 1) + " / " + ev.de(pick[1]["std_db"], 1) + " / " + ev.de(pick[2]["std_db"], 1) + " / " + ev.de(pick[3]["std_db"], 2) in text
    assert "−" + ev.de(abs(pick[0]["bias_db"]), 1) + " / −" + ev.de(abs(pick[1]["bias_db"]), 2) + " / −" + ev.de(abs(pick[2]["bias_db"]), 2) + " / −" + ev.de(abs(pick[3]["bias_db"]), 2) in text
    lo, hi = min(r["coverage"] for r in rows), max(r["coverage"] for r in rows)
    assert f"{ev.de(100 * lo, 1)} bis {ev.de(100 * hi, 1)} %" in text


def test_readme_overlap_numbers():
    rows = ev.sweep_overlap()
    text = _readme()
    h50, h625, h75 = (_row(rows, window="Hann", overlap=o) for o in (0.5, 0.625, 0.75))
    r50, r75 = _row(rows, window="Rechteck", overlap=0.5), _row(rows, window="Rechteck", overlap=0.75)
    b75 = _row(rows, window="Blackman", overlap=0.75)
    assert f"50 % → {ev.de(h50['k_eff'], 1)} (Faktor {ev.de(h50['gain'], 2)}), 62,5 % → {ev.de(h625['k_eff'], 1)}, 75 % → {ev.de(h75['k_eff'], 1)}" in text
    assert f"50 % → {ev.de(r50['k_eff'], 1)} (Faktor {ev.de(r50['gain'], 2)}), 75 % → {ev.de(r75['k_eff'], 1)}; Blackman 75 % → {ev.de(b75['k_eff'], 1)}" in text
    worst = max(abs(r["k_eff"] / r["k_eff_formula"] - 1.0) for r in rows if r["overlap"] > 0)
    assert worst < 0.025 and "auf 2 %" in text


def test_readme_tradeoff_and_two_lines_numbers():
    rows = ev.sweep_tradeoff()
    text = _readme()
    pick = [r for r in rows if r["seg"] in (32, 64, 128, 256, 512, 1024, 4096)]
    assert " / ".join(("**" + ev.de(r["rmse_db"], 2) + "**") if r["seg"] == 128 else ev.de(r["rmse_db"], 2) for r in pick) in text
    assert f"bei Länge 128 um {ev.de(abs(_row(rows, seg=128)['peak_db']), 1)} dB, bei 32 um {ev.de(abs(_row(rows, seg=32)['peak_db']), 1)} dB" in text
    two = ev.sweep_two_lines()
    assert f"1,5 Bins in {ev.de(100 * _row(two, delta=1.5)[256], 0)} % der Läufe, 2,5 Bins in {ev.de(100 * _row(two, delta=2.5)[256], 0)} %" in text
    assert f"bei 2 Bins {ev.de(100 * _row(two, delta=4.0)[128], 0)} %, bei 1 Bin {ev.de(100 * _row(two, delta=4.0)[64], 0)} %" in text


def test_readme_leakage_smoothing_and_detection_numbers():
    text = _readme()
    leak = ev.sweep_leakage()
    assert f"um {ev.de(leak[0]['bands'][2], 1)} dB, Welch mit Hann (50 %) und Hann ohne Überlappung um {ev.de(abs(leak[1]['bands'][2]), 1)} dB" in text
    assert f"der ganzen Aufzeichnung um {ev.de(leak[4]['bands'][2], 1)} bis {ev.de(leak[3]['bands'][2], 1)} dB" in text
    sm = ev.sweep_smoothing()
    welch_rmse = " / ".join(ev.de(sm[i]["rmse_db"], 2) for i in (0, 3, 6))
    rect_rmse = " / ".join(ev.de(sm[i]["rmse_db"], 2) for i in (1, 4, 7))
    assert f"({rect_rmse} dB gegen {welch_rmse})" in text
    assert f"{ev.de(sm[2]['bandwidth_hz'] / sm[0]['bandwidth_hz'], 1)}-fache Bandbreite und liegt am Gipfel {ev.de(abs(sm[2]['peak_db']), 1)} dB statt {ev.de(abs(sm[0]['peak_db']), 1)} dB tief" in text
    det = _row(ev.sweep_detect(), snr=-20)
    assert " / ".join(ev.de(100 * v, 0) for v in det["share"]) + " %" in text
