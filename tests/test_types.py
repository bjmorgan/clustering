"""Tests for BondSpec."""

import pytest

from md_clusters import BondSpec


class TestBondSpec:
    """Tests for BondSpec construction and validation."""

    def test_species_sorted(self):
        spec = BondSpec(species=("O", "C"), max_length=1.6)
        assert spec.species == ("C", "O")

    def test_species_already_sorted(self):
        spec = BondSpec(species=("C", "O"), max_length=1.6)
        assert spec.species == ("C", "O")

    def test_max_length_positive(self):
        with pytest.raises(ValueError, match="max_length must be positive"):
            BondSpec(species=("C", "O"), max_length=0.0)

    def test_max_length_negative(self):
        with pytest.raises(ValueError, match="max_length must be positive"):
            BondSpec(species=("C", "O"), max_length=-1.0)

    def test_min_length_negative(self):
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            BondSpec(species=("C", "O"), max_length=1.6, min_length=-0.1)

    def test_min_exceeds_max(self):
        with pytest.raises(ValueError, match="min_length.*must not exceed"):
            BondSpec(species=("C", "O"), max_length=1.0, min_length=1.5)

    def test_frozen(self):
        spec = BondSpec(species=("C", "O"), max_length=1.6)
        with pytest.raises(AttributeError):
            spec.max_length = 2.0
