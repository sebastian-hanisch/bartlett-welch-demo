"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hann4():
    """Periodisches Hann-Fenster der Länge 4 von Hand: w = 0,5 - 0,5 cos(2 pi m / 4) = (0; 0,5; 1; 0,5), Summe der Quadrate 1,5."""
    return np.array([0.0, 0.5, 1.0, 0.5])
