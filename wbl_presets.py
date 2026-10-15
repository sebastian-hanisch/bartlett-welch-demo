"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle gezeichneten Regler sind in jeder Einstellung wirksam: Die Regler der Linien erscheinen nur, wenn es Linien gibt (Linie 1 und Stärke ab „eine“, Abstand und
Stärke der Linie 2 bei „zwei“). Ihre Werte stehen nicht im Zustand eines Reglers, sondern unter `*_kept` (kein Frontend/Backend-Auseinanderlaufen: der Regler bekommt sie als
Startwert). Die Segmentlänge ist durch die Länge n begrenzt: `clamp_seg` hält den Zustand gültig, die Auswahl zeigt nur Längen bis n.
"""

import random
from dataclasses import dataclass

import streamlit as st

import wbl_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_select": SettingSpec("n", int, C.DEFAULT_N, choices=C.N_CHOICES),
    "seg_select": SettingSpec("seg", int, C.DEFAULT_SEG, choices=C.SEG_CHOICES),
    "overlap_select": SettingSpec("ov", float, C.DEFAULT_OVERLAP, choices=C.OVERLAP_CHOICES),
    "window_select": SettingSpec("w", str, C.DEFAULT_WINDOW, choices=C.WINDOWS),
    "background_select": SettingSpec("bg", str, C.DEFAULT_BACKGROUND, choices=tuple(C.BACKGROUNDS)),
    "lines_select": SettingSpec("ln", str, C.DEFAULT_LINES, choices=C.LINES),
    "line_kept": SettingSpec("f", float, C.DEFAULT_LINE, C.LINE_MIN, C.LINE_MAX),
    "gap_kept": SettingSpec("d", float, C.DEFAULT_GAP, C.GAP_MIN, C.GAP_MAX),
    "rel_kept": SettingSpec("rel", int, C.DEFAULT_REL, C.REL_MIN, C.REL_MAX),
    "snr_kept": SettingSpec("snr", int, C.DEFAULT_SNR, C.SNR_MIN, C.SNR_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_select", "seg": "seg_select", "overlap": "overlap_select", "window": "window_select", "background": "background_select",
               "lines": "lines_select", "line": "line_kept", "gap": "gap_kept", "rel": "rel_kept", "snr": "snr_kept", "seed": "seed_input"}


def seg_options(n: int) -> list[int]:
    """Segmentlängen, die in eine Folge der Länge n passen."""
    return [s for s in C.SEG_CHOICES if s <= n]


def clamp_seg():
    """Segmentlänge höchstens n (nach Permalink, Preset oder geänderter Länge)."""
    n, seg = st.session_state["n_select"], st.session_state["seg_select"]
    if seg > n:
        st.session_state["seg_select"] = n


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default
    clamp_seg()


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]
    clamp_seg()


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
