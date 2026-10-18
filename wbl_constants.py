"""Konstanten: Regler-Grenzen, feste Annahmen, Hintergründe, Experimente, Farben und Presets (Kern: wbl_welch.py, wbl_windows.py)."""

import wbl_windows as W

FS = 1000.0                                  # Abtastrate in Hz (feste Annahme)
N_CHOICES = (1024, 2048, 4096, 8192, 16384)
DEFAULT_N = 4096
SEG_MIN = 32
SEG_CHOICES = tuple(2 ** k for k in range(5, 15))        # 32 .. 16384 (je Länge n nur die Werte bis n)
DEFAULT_SEG = 256
OVERLAP_CHOICES = (0.0, 0.25, 0.5, 0.625, 0.75, 0.875)
DEFAULT_OVERLAP = 0.5
PAD = 4                                      # Auffüllfaktor der Segment-FFT für die Darstellung (ändert Streuung und Auflösung nicht)
WINDOWS = W.NAMES
DEFAULT_WINDOW = "Hann"
LINES = ("keine", "eine", "zwei")
DEFAULT_LINES = "eine"
LINE_MIN, LINE_MAX, LINE_STEP, DEFAULT_LINE = 0.05, 0.43, 0.0005, 0.35    # Linie 1 als Bruchteil der Abtastrate
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = 0.5, 16.0, 0.25, 4.0            # Abstand der Linie 2 in Bins der Länge 256 (Vielfache von f_s / 256)
GAP_REF = 256
REL_MIN, REL_MAX, DEFAULT_REL = -40, 0, 0                                  # Stärke der Linie 2 gegen Linie 1 in dB
SNR_MIN, SNR_MAX, DEFAULT_SNR = -30, 30, 10                                # Linie 1 gegen Rauschvarianz 1, in dB je Abtastwert
SEED_MAX, DEFAULT_SEED = 999999, 36

# Pole (Frequenz als Bruchteil der Abtastrate, Radius) je Hintergrund; Varianz auf 1 normiert
BACKGROUNDS = {"weiß": (), "Resonanz": ((0.2, 0.97),), "Steil": ((0.2, 0.99), (0.0, 0.9), (0.0, 0.9))}
DEFAULT_BACKGROUND = "Resonanz"

# Experimente (feste Ströme)
SWEEP_SEED = 300000
EXP_N = 4096
EXP_VAR_K = (1, 2, 4, 8, 16, 32, 64)
EXP_VAR_RUNS = 200
EXP_OVL_SEG = 256
EXP_OVL_OVERLAPS = (0.0, 0.25, 0.5, 0.625, 0.75, 0.875)
EXP_OVL_RUNS = 200
EXP_TRADE_SEGS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
EXP_TRADE_RUNS = 150
EXP_TRADE_BACKGROUND = "Resonanz"
EXP_RES_N = 2048
EXP_RES_SEGS = (2048, 1024, 512, 256, 128, 64)
EXP_RES_DELTAS = (1.5, 2.5, 4.0, 6.0)          # Abstand in Bins von L = 256
EXP_RES_RUNS = 100
EXP_RES_SNR = 10
EXP_LEAK_SEG = 256
EXP_LEAK_RUNS = 200
EXP_LEAK_SMOOTH = (16, 29)
EXP_SMOOTH_SEGS = (128, 256, 512)
SMOOTH_PAD = 5                               # Auffüllfaktor im Glättungsexperiment: die Gipfelfrequenz 0,2 liegt dann auf dem Raster
EXP_DET_K = (1, 2, 4, 8, 16, 32)
EXP_DET_SNR = (-24, -22, -20, -18)
EXP_DET_RUNS = 600
EXP_DET_QUANTILE = 0.99

COLOR_FULL = "rgba(150,150,150,0.55)"
COLOR_WELCH = "rgba(76,120,168,0.95)"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_EXPECTED = "#E8833A"
COLOR_TRUE = "#2F6B65"
COLOR_LINE = "#B03A2E"

PRESETS = {
    "Rauschen beruhigen": {"n": 4096, "seg": 128, "overlap": 0.0, "window": "Rechteck", "background": "weiß", "lines": "keine", "line": 0.2, "gap": 4.0,
                           "rel": 0, "snr": 10, "seed": DEFAULT_SEED},
    "Resonanz: das Optimum": {"n": 4096, "seg": 128, "overlap": 0.5, "window": "Hann", "background": "Resonanz", "lines": "keine", "line": 0.2, "gap": 4.0,
                              "rel": 0, "snr": 10, "seed": DEFAULT_SEED},
    "Zwei Linien: Schärfe geht verloren": {"n": 2048, "seg": 128, "overlap": 0.5, "window": "Hann", "background": "weiß", "lines": "zwei", "line": 0.2,
                                           "gap": 3.0, "rel": 0, "snr": 10, "seed": DEFAULT_SEED},
    "Steiler Hintergrund: Leckage": {"n": 4096, "seg": 256, "overlap": 0.0, "window": "Rechteck", "background": "Steil", "lines": "keine", "line": 0.2,
                                     "gap": 4.0, "rel": 0, "snr": 10, "seed": DEFAULT_SEED},
    "Schwache Linie": {"n": 4096, "seg": 1024, "overlap": 0.5, "window": "Hann", "background": "weiß", "lines": "eine", "line": 0.2, "gap": 4.0,
                       "rel": 0, "snr": -20, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Rauschen beruhigen": "Weißes Rauschen, 32 Segmente ohne Überlappung: Die graue Kurve (ein Periodogramm über alles) streut um ±5,6 dB, die blaue nur um ±0,8 dB.",
    "Resonanz: das Optimum": "Ein Gipfel im Hintergrund, 63 Segmente der Länge 128 mit Hann und 50 % Überlappung: kleinster Gesamtfehler. Der Gipfel liegt im Mittel 1,7 dB zu tief; kürzere Segmente verwischen ihn weiter, längere streuen mehr.",
    "Zwei Linien: Schärfe geht verloren": "Zwei gleich starke Linien, 1,5 Bins der Segmentlänge auseinander: Das lange Periodogramm (grau) trennt sie in diesem Beispiel, das Welch-Mittel (blau) nicht (bei 1,5 Bins gelingt Letzteres nur in einem Teil der Läufe).",
    "Steiler Hintergrund: Leckage": "Rechteck-Segmente ohne Überlappung bei 57 dB Spannweite: Die Nebenkeulen füllen das Tal um etwa 15 dB. Mit Hann liegt die Schätzung auf der wahren Dichte.",
    "Schwache Linie": "Eine Linie 20 dB unter dem Rauschen je Abtastwert, 7 Segmente mit Hann: Sie ragt je nach Zufall etwa 5 bis 7 dB über die Rauschdichte (hier 6,5 dB). Kürzere Segmente senken ihren Abstand zum Rauschen, weil die Linienleistung je Bin mit der Länge sinkt.",
}
