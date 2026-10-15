"""AppTest: Auswahlfelder, Permalink, Segmentlänge gegen Länge, abhängige Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import wbl_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 240


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


@pytest.mark.parametrize("window", C.WINDOWS)
def test_every_window_runs(window):
    at = _run()
    at.selectbox(key="window_select").set_value(window).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_background_runs(background):
    at = _run()
    at.selectbox(key="background_select").set_value(background).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def _labels(at):
    return [x.label for x in at.slider]


def test_line_controls_appear_only_when_there_are_lines():
    at = _run({"ln": "keine"})
    assert not any(label.startswith(("Linie 1", "Abstand", "Linie 2")) for label in _labels(at))
    at.radio(key="lines_select").set_value("eine").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    labels = _labels(at)
    assert "Linie 1 (Bruchteil der Abtastrate)" in labels and "Linie 1 gegen Rauschen (dB je Abtastwert)" in labels and "Abstand der Linie 2 (Bins der Länge 256)" not in labels
    at.radio(key="lines_select").set_value("zwei").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    labels = _labels(at)
    assert "Abstand der Linie 2 (Bins der Länge 256)" in labels and "Linie 2 gegen Linie 1 (dB)" in labels


def test_line_values_survive_switching_the_lines_away_and_back():
    at = _run({"ln": "zwei", "f": "0.3", "d": "7.5", "rel": "-12", "snr": "-5"})
    by_label = {x.label: x for x in at.slider}
    assert by_label["Linie 1 (Bruchteil der Abtastrate)"].value == 0.3 and by_label["Abstand der Linie 2 (Bins der Länge 256)"].value == 7.5
    at.radio(key="lines_select").set_value("keine").run(timeout=TIMEOUT)
    assert not at.exception and not any(label.startswith("Linie") for label in _labels(at))
    at.radio(key="lines_select").set_value("zwei").run(timeout=TIMEOUT)
    by_label = {x.label: x for x in at.slider}
    assert by_label["Linie 1 (Bruchteil der Abtastrate)"].value == 0.3 and by_label["Abstand der Linie 2 (Bins der Länge 256)"].value == 7.5
    assert by_label["Linie 2 gegen Linie 1 (dB)"].value == -12 and by_label["Linie 1 gegen Rauschen (dB je Abtastwert)"].value == -5


def test_shrinking_the_record_pulls_the_segment_length_down():
    at = _run({"n": "16384", "seg": "8192"})
    assert at.select_slider(key="seg_select").value == 8192
    at.select_slider(key="n_select").set_value(1024).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.select_slider(key="seg_select").value == 1024
    assert max(at.select_slider(key="seg_select").options) == "1024" or max(int(o) for o in at.select_slider(key="seg_select").options) == 1024


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"n": "2048", "seg": "128", "ov": "0.75", "w": "Blackman", "bg": "Steil", "ln": "zwei", "f": "9", "d": "0", "rel": "-999", "snr": "999", "seed": "42"})
    assert at.select_slider(key="n_select").value == 2048 and at.select_slider(key="seg_select").value == 128
    assert at.select_slider(key="overlap_select").value == 0.75
    assert at.selectbox(key="window_select").value == "Blackman" and at.selectbox(key="background_select").value == "Steil"
    assert at.radio(key="lines_select").value == "zwei"
    by_label = {x.label: x for x in at.slider}
    assert by_label["Linie 1 (Bruchteil der Abtastrate)"].value == C.LINE_MAX and by_label["Abstand der Linie 2 (Bins der Länge 256)"].value == C.GAP_MIN
    assert by_label["Linie 2 gegen Linie 1 (dB)"].value == C.REL_MIN and by_label["Linie 1 gegen Rauschen (dB je Abtastwert)"].value == C.SNR_MAX
    at2 = _run({"n": "3000", "w": "Gauß", "ov": "0.3", "bg": "unsinn", "ln": "drei"})
    assert at2.select_slider(key="n_select").value == C.DEFAULT_N and at2.selectbox(key="window_select").value == C.DEFAULT_WINDOW
    assert at2.select_slider(key="overlap_select").value == C.DEFAULT_OVERLAP and at2.radio(key="lines_select").value == C.DEFAULT_LINES


def test_permalink_with_a_segment_longer_than_the_record_is_clamped():
    at = _run({"n": "1024", "seg": "8192"})
    assert at.select_slider(key="seg_select").value == 1024


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Zwei Linien: Schärfe geht verloren").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.radio(key="lines_select").value == "zwei" and at.select_slider(key="n_select").value == 2048 and at.select_slider(key="seg_select").value == 128


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("label", ["Abstand der Linie 2 (Bins der Länge 256)", "Linie 2 gegen Linie 1 (dB)"])
@pytest.mark.parametrize("edge", ["min", "max"])
def test_second_line_sliders_at_their_limits(label, edge):
    at = _run({"ln": "zwei"})
    slider = [x for x in at.slider if x.label == label][0]
    value = slider.min if edge == "min" else slider.max
    slider.set_value(int(value) if isinstance(slider.value, int) else value).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
