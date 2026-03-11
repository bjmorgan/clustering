"""Cluster composition analysis using hill-system formulae."""

from __future__ import annotations

from collections import Counter

import numpy as np


def _hill_formula(counts: Counter[str]) -> str:
    """Format element counts as a hill-system formula string.

    Hill order: C first, then H, then remaining elements alphabetically.
    Counts of 1 are omitted (e.g. ``"CH4"`` not ``"C1H4"``).
    """
    parts: list[str] = []

    def _append(symbol: str) -> None:
        n = counts[symbol]
        parts.append(symbol if n == 1 else f"{symbol}{n}")

    # Carbon first, then hydrogen.
    if "C" in counts:
        _append("C")
    if "H" in counts:
        _append("H")

    # Remaining elements alphabetically.
    for symbol in sorted(counts):
        if symbol not in ("C", "H"):
            _append(symbol)

    return "".join(parts)


def cluster_composition(
    species: list[str],
    labels: np.ndarray,
) -> Counter[str]:
    """Count cluster formulae across all clusters in a frame.

    Args:
        species: Species labels, length ``n_atoms``.
        labels: Cluster label per atom from :func:`find_clusters`.

    Returns:
        Counter mapping canonical formula string (hill order) to the
        number of times that formula appears.
        e.g. ``Counter({"CO2": 3, "H2O": 5})``.
    """
    n_clusters = int(labels.max()) + 1 if len(labels) > 0 else 0
    formula_counts: Counter[str] = Counter()

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        element_counts: Counter[str] = Counter()
        for i in np.nonzero(mask)[0]:
            element_counts[species[i]] += 1
        formula = _hill_formula(element_counts)
        formula_counts[formula] += 1

    return formula_counts
