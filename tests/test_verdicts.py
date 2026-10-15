"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import wbl_evaluation as ev


def make_analysis(**kw):
    base = dict(settings=ev.Settings(seg=256, overlap=0.5), k=31, k_eff=29.4, f_hz=np.zeros(1), x=np.zeros(1), w=np.zeros(1), est=np.zeros(1), expected=np.zeros(1),
                true_density=np.zeros(1), band_lo=np.zeros(1), band_hi=np.zeros(1), f_full=np.zeros(1), full=np.zeros(1), enbw_hz=5.859, full_enbw_hz=0.366,
                std_db=0.81, full_std_db=5.57, log_bias_db=-0.05, peak_bias_db=None, resolved=None, full_resolved=None, ops=13888, ops_full=24576)
    base.update(kw)
    return ev.Analysis(**base)


def test_main_verdict_without_two_lines():
    kind, text = ev.verdict_main(make_analysis())
    assert kind == "info"
    assert text == ("31 Segmente der Länge 256 (50 % Überlappung) ergeben etwa 29,4 unabhängige Mittel: Die Streuung je Frequenz sinkt von ±5,6 dB auf ±0,8 dB, "
                    "die Rauschbandbreite steigt von 0,37 auf 5,86 Hz (Faktor 16).")


def test_main_verdict_for_two_lines_in_the_three_cases():
    s = ev.Settings(seg=128, overlap=0.5, lines="zwei")
    kind, text = ev.verdict_main(make_analysis(settings=s, resolved=True, full_resolved=True))
    assert kind == "success" and "**zwei Gipfel**" in text
    kind, text = ev.verdict_main(make_analysis(settings=s, resolved=False, full_resolved=True))
    assert kind == "warning" and "**verschmelzen**" in text and "das lange Periodogramm trennt sie" in text
    kind, text = ev.verdict_main(make_analysis(settings=s, resolved=False, full_resolved=False))
    assert kind == "warning" and "**nicht getrennt**" in text


def test_main_verdict_uses_the_effective_segment_length_and_overlap_in_percent():
    s = ev.Settings(n=1024, seg=4096, overlap=0.875)
    _, text = ev.verdict_main(make_analysis(settings=s, k=1, k_eff=1.0))
    assert "1 Segmente der Länge 1024 (87,5 % Überlappung)" in text


def test_percent_formatting():
    assert ev.percent(0.5) == "50" and ev.percent(0.625) == "62,5" and ev.percent(0.0) == "0" and ev.percent(0.875) == "87,5"


def test_variance_verdict():
    rows = [{"k": 1, "cv": 0.99, "cv_theory": 1.0, "std_db": 5.54, "coverage": 0.95}, {"k": 64, "cv": 0.1273, "cv_theory": 0.125, "std_db": 0.551, "coverage": 0.944}]
    kind, text = ev.verdict_variance(rows)
    assert kind == "success"
    assert "Bei 1 Segment fällt der Variationskoeffizient je Bin nicht unter 0,99; bei 64 Segmenten liegt er bei 0,127 (Theorie 1/√K = 0,125)" in text
    assert "bei 0,55 dB statt 5,54 dB" in text and "deckt in 94,4 % der Fälle" in text


def test_overlap_verdict_picks_the_right_rows():
    rows = [{"window": w, "overlap": o, "k_eff": k, "gain": k / 16.0} for w, o, k in
            (("Hann", 0.5, 30.1), ("Rechteck", 0.5, 21.3), ("Hann", 0.75, 32.7), ("Hann", 0.0, 16.0), ("Rechteck", 0.75, 23.0))]
    kind, text = ev.verdict_overlap(rows)
    assert kind == "info"
    assert "liefert Hann 30,1 unabhängige Mittel (Faktor 1,88 gegen ohne Überlappung), das Rechteck nur 21,3 (Faktor 1,33)" in text and "75 %: 32,7" in text


def test_tradeoff_verdict_names_the_minimum_not_the_first_or_last_row():
    rows = [{"seg": 32, "k": 255, "rmse_db": 1.98, "peak_db": -5.7}, {"seg": 128, "k": 63, "rmse_db": 0.64, "peak_db": -1.7}, {"seg": 4096, "k": 1, "rmse_db": 6.1, "peak_db": 0.0}]
    kind, text = ev.verdict_tradeoff(rows)
    assert kind == "warning"
    assert "bei Segmentlänge 128 (63 Segmente) liegt er bei 0,64 dB" in text and "streut mit 6,1 dB" in text and "(32) verwischen den Gipfel um 5,7 dB" in text


def test_two_lines_verdict_reads_the_rows_for_four_bins():
    rows = [{"delta": 2.5, 256: 1.0, 128: 0.0, 64: 0.0}, {"delta": 4.0, 256: 1.0, 128: 0.79, 64: 0.0}]
    kind, text = ev.verdict_two_lines(rows)
    assert kind == "info" and "bei Segmentlänge 256 in 100 % der Läufe" in text and "bei 128 (2 Bins) in 79 %, bei 64 (1 Bin) in 0 %" in text


def test_leakage_verdict_uses_the_valley_column():
    rows = [{"label": "a", "bands": [1.0, 2.0, 14.6]}, {"label": "b", "bands": [0.3, 0.0, 0.0054]}]
    kind, text = ev.verdict_leakage(rows)
    assert kind == "success" and "Bartlett mit Rechteck-Segmenten um 14,6 dB" in text and "mit Hann-Segmenten um 0,0 dB" in text


def test_smoothing_verdict_compares_rows_zero_one_two():
    rows = [{"std_db": 0.56, "peak_db": -1.7, "bandwidth_hz": 11.7}, {"std_db": 0.57, "peak_db": -1.9, "bandwidth_hz": 14.9}, {"std_db": 0.6, "peak_db": -3.7, "bandwidth_hz": 28.7}]
    kind, text = ev.verdict_smoothing(rows)
    assert kind == "info"
    assert "Bei gleicher Streuung (0,56 dB)" in text
    assert "(-1,9 statt -1,7 dB)" in text and "2,5-fach so viel Bandbreite (-3,7 dB)" in text


def test_detect_verdict_reads_minus_twenty_decibels():
    rows = [{"snr": -22, "k": [1, 2, 4], "share": [0.2, 0.3, 0.2]}, {"snr": -20, "k": [1, 2, 4], "share": [0.50, 0.69, 0.30]}]
    kind, text = ev.verdict_detect(rows)
    assert kind == "warning"
    assert "mit einem Periodogramm in 50 % der Läufe gefunden, mit 2 Segmenten in 69 %, mit 4 Segmenten nur in 30 %" in text
