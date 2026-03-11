"""Shared test fixtures."""

import numpy as np


def cubic_lattice(a: float) -> np.ndarray:
    """Return a cubic lattice matrix with side length *a*."""
    return np.eye(3) * a
