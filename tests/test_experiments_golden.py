"""Festwerte der Experimente (feste Ströme ab Seed 300000): fangen Vorzeichen, Exponenten, Normierungen und Lagen, die Toleranz-Tests übersehen.

Die Werte stammen aus dem ersten Lauf des geprüften Codes (Handrechnung und scipy-Orakel stehen in test_welch.py und test_oracle_welch.py)."""

import pytest

import wbl_constants as C
import wbl_evaluation as ev

TOL = dict(abs=2e-4)


def col(rows, key):
    return [r[key] for r in rows]


def test_variance_sweep_fixed_values():
    rows = ev.sweep_variance()
    assert col(rows, "k") == [1, 2, 4, 8, 16, 32, 64] and col(rows, "seg") == [4096, 2048, 1024, 512, 256, 128, 64]
    assert col(rows, "cv") == pytest.approx([0.9923, 0.7032, 0.4976, 0.3514, 0.2501, 0.1774, 0.1273], **TOL)
    assert col(rows, "bias_db") == pytest.approx([-2.5049, -1.1705, -0.5647, -0.2751, -0.138, -0.0685, -0.0336], **TOL)
    assert col(rows, "std_db") == pytest.approx([5.539, 3.4651, 2.3042, 1.5736, 1.1036, 0.7771, 0.5551], **TOL)
    assert col(rows, "coverage") == pytest.approx([0.9502, 0.9508, 0.9496, 0.9513, 0.9484, 0.9478, 0.944], **TOL)
    assert col(rows, "cv_theory") == pytest.approx([1.0, 0.7071, 0.5, 0.3536, 0.25, 0.1768, 0.125], **TOL)
    assert col(rows, "bias_theory") == pytest.approx([-2.5068, -1.1742, -0.5647, -0.2766, -0.1368, -0.068, -0.0338], abs=3e-3)
    assert col(rows, "std_theory") == pytest.approx([5.5700, 3.4880, 2.3140, 1.5850, 1.1030, 0.7740, 0.5450], abs=3e-3)


def test_overlap_sweep_fixed_values():
    rows = ev.sweep_overlap()
    keff = {w: col([r for r in rows if r["window"] == w], "k_eff") for w in C.WINDOWS}
    formula = {w: col([r for r in rows if r["window"] == w], "k_eff_formula") for w in C.WINDOWS}
    assert keff["Rechteck"] == pytest.approx([16.0036, 19.0859, 21.0328, 22.0709, 22.6882, 23.103], **TOL)
    assert keff["Hann"] == pytest.approx([15.9464, 21.1454, 29.82, 32.3603, 32.4453, 32.2347], **TOL)
    assert keff["Hamming"] == pytest.approx([15.9466, 21.1609, 28.4052, 29.6938, 29.5982, 29.4038], **TOL)
    assert keff["Blackman"] == pytest.approx([15.9171, 21.1057, 30.9042, 36.445, 37.6299, 37.4072], **TOL)
    assert formula["Hann"] == pytest.approx([16.0, 20.9977, 29.4184, 31.9146, 31.9612, 31.7447], **TOL)
    assert formula["Rechteck"] == pytest.approx([16.0, 18.766, 20.8913, 21.7958, 22.5174, 22.881], **TOL)
    assert col([r for r in rows if r["window"] == "Hann"], "k") == [16, 21, 31, 41, 61, 121]
    assert [r["gain"] for r in rows if r["window"] == "Hann" and r["overlap"] == 0.5] == pytest.approx([29.82 / 16.0], **TOL)


def test_tradeoff_sweep_fixed_values():
    rows = ev.sweep_tradeoff()
    assert col(rows, "seg") == [32, 64, 128, 256, 512, 1024, 2048, 4096] and col(rows, "k") == [255, 127, 63, 31, 15, 7, 3, 1]
    assert col(rows, "bias_db") == pytest.approx([1.1711, 0.2946, 0.0587, -0.0494, -0.1472, -0.3272, -0.7746, -2.5011], **TOL)
    assert col(rows, "std_db") == pytest.approx([0.3004, 0.401, 0.5655, 0.8039, 1.1648, 1.7325, 2.7399, 5.5406], **TOL)
    assert col(rows, "rmse_db") == pytest.approx([1.9809, 1.0416, 0.6423, 0.8162, 1.1801, 1.7721, 2.8621, 6.1148], **TOL)
    assert col(rows, "peak_db") == pytest.approx([-5.708, -3.4478, -1.7255, -0.6774, -0.2107, -0.0569, -0.0146, -0.0037], **TOL)
    assert col(rows, "enbw_hz") == pytest.approx([46.875, 23.4375, 11.71875, 5.859375, 2.9296875, 1.46484375, 0.732421875, 0.3662109375], abs=1e-9)


def test_two_lines_sweep_fixed_values():
    rows = ev.sweep_two_lines()
    expect = {1.5: [1.0, 1.0, 1.0, 0.28, 0.0, 0.0], 2.5: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0], 4.0: [1.0, 1.0, 1.0, 1.0, 0.79, 0.0], 6.0: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]}
    for r in rows:
        assert [r[s] for s in C.EXP_RES_SEGS] == pytest.approx(expect[r["delta"]], abs=1e-9)


def test_leakage_sweep_fixed_values():
    rows = ev.sweep_leakage()
    assert [r["label"] for r in rows][:3] == ["Bartlett: Rechteck, 0 % Überlappung", "Welch: Hann, 50 % Überlappung", "Hann, 0 % Überlappung"]
    assert rows[0]["bands"] == pytest.approx([1.7574, 9.5029, 14.5802], **TOL)
    assert rows[1]["bands"] == pytest.approx([0.3145, -0.0079, 0.0054], **TOL)
    assert rows[2]["bands"] == pytest.approx([0.3305, 0.0018, 0.0127], **TOL)
    assert rows[3]["bands"] == pytest.approx([0.2362, 1.7625, 4.2744], **TOL)
    assert rows[4]["bands"] == pytest.approx([0.397, 1.7563, 4.2282], **TOL)


def test_smoothing_sweep_fixed_values():
    rows = ev.sweep_smoothing()
    expect = [("Welch, Segmentlänge 128, Hann, 50 %", 59.734, 0.5633, -1.7006, 0.6583, 11.719),
              ("Ganze Aufzeichnung, Rechteck, 60 Bins gemittelt", 60.0, 0.5731, -1.8513, 0.6647, 14.893),
              ("Ganze Aufzeichnung, Hann, 116 Bins gemittelt", 59.923, 0.5998, -3.6568, 1.0909, 28.687),
              ("Welch, Segmentlänge 256, Hann, 50 %", 29.418, 0.8012, -0.6218, 0.8114, 5.859),
              ("Ganze Aufzeichnung, Rechteck, 29 Bins gemittelt", 29.0, 0.8117, -0.5804, 0.8194, 7.324),
              ("Ganze Aufzeichnung, Hann, 57 Bins gemittelt", 29.581, 0.8098, -1.6827, 0.8675, 14.282),
              ("Welch, Segmentlänge 512, Hann, 50 %", 14.261, 1.1554, -0.1057, 1.1682, 2.93),
              ("Ganze Aufzeichnung, Rechteck, 14 Bins gemittelt", 14.0, 1.1749, -0.1586, 1.1846, 3.662),
              ("Ganze Aufzeichnung, Hann, 27 Bins gemittelt", 14.155, 1.1597, -0.3978, 1.1738, 6.958)]
    assert len(rows) == len(expect)
    for r, (label, k_eff, std, peak, rmse, bw) in zip(rows, expect):
        assert r["label"] == label
        assert (r["k_eff"], r["std_db"], r["peak_db"], r["rmse_db"], r["bandwidth_hz"]) == pytest.approx((k_eff, std, peak, rmse, bw), abs=2e-3)


def test_detect_sweep_fixed_values():
    rows = ev.sweep_detect()
    expect = {-24: [0.0583, 0.0817, 0.0683, 0.0583, 0.035, 0.0283], -22: [0.1967, 0.2883, 0.2167, 0.1967, 0.1367, 0.0717],
              -20: [0.5033, 0.6883, 0.645, 0.6417, 0.4717, 0.2733], -18: [0.8883, 0.9667, 0.97, 0.9683, 0.915, 0.8217]}
    for r in rows:
        assert r["k"] == [1, 2, 4, 8, 16, 32] and r["runs"] == C.EXP_DET_RUNS
        assert r["share"] == pytest.approx(expect[r["snr"]], **TOL)
