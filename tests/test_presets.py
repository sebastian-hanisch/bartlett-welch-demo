"""Presets: Vollständigkeit, gültige Werte, Standardwerte; Segmentlänge höchstens n."""

import wbl_constants as C
import wbl_evaluation as ev
import wbl_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["n"] in C.N_CHOICES and p["seg"] in C.SEG_CHOICES and p["seg"] <= p["n"] and p["overlap"] in C.OVERLAP_CHOICES, name
        assert p["window"] in C.WINDOWS and p["background"] in C.BACKGROUNDS and p["lines"] in C.LINES
        assert C.LINE_MIN <= p["line"] <= C.LINE_MAX and C.GAP_MIN <= p["gap"] <= C.GAP_MAX and C.REL_MIN <= p["rel"] <= C.REL_MAX
        assert C.SNR_MIN <= p["snr"] <= C.SNR_MAX and 0 <= p["seed"] <= C.SEED_MAX


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.n in C.N_CHOICES and s.seg in P.seg_options(s.n) and s.overlap in C.OVERLAP_CHOICES and s.window in C.WINDOWS and s.background in C.BACKGROUNDS


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_names_are_short_enough_for_the_buttons():
    assert max(len(n) for n in C.PRESETS) <= 36


def test_segment_options_are_powers_of_two_up_to_n():
    assert P.seg_options(1024) == [32, 64, 128, 256, 512, 1024]
    assert P.seg_options(16384)[-1] == 16384 and len(P.seg_options(16384)) == 10


def test_second_line_never_leaves_the_band():
    for line in (C.LINE_MIN, C.LINE_MAX):
        assert ev.Settings(line=line, gap=C.GAP_MAX).f2_hz < C.FS / 2
