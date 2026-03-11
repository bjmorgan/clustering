"""Tests for composition analysis."""

from collections import Counter

import numpy as np

from md_clusters import cluster_composition
from md_clusters.composition import _hill_formula


class TestHillFormula:
    """Test hill-order formula generation."""

    def test_co2(self):
        assert _hill_formula(Counter({"C": 1, "O": 2})) == "CO2"

    def test_h2o(self):
        assert _hill_formula(Counter({"H": 2, "O": 1})) == "H2O"

    def test_ch4(self):
        assert _hill_formula(Counter({"C": 1, "H": 4})) == "CH4"

    def test_single_element(self):
        assert _hill_formula(Counter({"Fe": 1})) == "Fe"

    def test_no_carbon(self):
        """Without carbon, H is not given priority -- all alphabetical."""
        assert _hill_formula(Counter({"H": 2, "O": 1})) == "H2O"

    def test_complex_formula(self):
        """Multiple elements with carbon present."""
        counts = Counter({"C": 2, "H": 6, "O": 1, "N": 1})
        assert _hill_formula(counts) == "C2H6NO"


class TestClusterComposition:
    """Test formula counting across clusters."""

    def test_simple_composition(self):
        species = ["C", "O", "O", "C", "O", "O"]
        labels = np.array([0, 0, 0, 1, 1, 1])

        result = cluster_composition(species, labels)
        assert result == Counter({"CO2": 2})

    def test_mixed_clusters(self):
        species = ["C", "O", "O", "H", "H", "O"]
        labels = np.array([0, 0, 0, 1, 1, 1])

        result = cluster_composition(species, labels)
        assert result == Counter({"CO2": 1, "H2O": 1})

    def test_isolated_atoms(self):
        species = ["C", "O", "Fe"]
        labels = np.array([0, 1, 2])

        result = cluster_composition(species, labels)
        assert result == Counter({"C": 1, "O": 1, "Fe": 1})

    def test_empty(self):
        result = cluster_composition([], np.array([], dtype=int))
        assert result == Counter()

    def test_returns_counter(self):
        """Return type is Counter, supporting arithmetic and most_common."""
        species = ["C", "O", "O"]
        labels = np.array([0, 0, 0])
        result = cluster_composition(species, labels)
        assert isinstance(result, Counter)
