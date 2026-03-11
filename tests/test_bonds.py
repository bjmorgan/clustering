"""Tests for bond detection."""

import numpy as np
import pytest

from md_clusters import BondSpec, find_bonds
from tests.conftest import cubic_lattice


class TestFindBondsSimple:
    """Test bond detection on simple geometries in a large cell."""

    def test_two_bonded_atoms(self):
        """Two atoms 1.0 A apart, spec allows up to 1.5 A."""
        species = ["C", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]
        lattice = cubic_lattice(20.0)

        adj = find_bonds(species, coords, specs, lattice)

        assert adj.shape == (2, 2)
        assert adj[0, 1]
        assert adj[1, 0]  # Symmetric.
        assert not adj[0, 0]  # No self-bonds.

    def test_two_atoms_too_far(self):
        """Two atoms 3.0 A apart, spec allows up to 1.5 A."""
        species = ["C", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]
        lattice = cubic_lattice(20.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj.nnz == 0

    def test_species_mismatch(self):
        """Two close atoms but wrong species for the spec."""
        species = ["C", "C"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]
        lattice = cubic_lattice(20.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj.nnz == 0

    def test_wildcard_species(self):
        """Wildcard spec matches any species."""
        species = ["Zr", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        specs = [BondSpec(species=("*", "*"), max_length=1.5)]
        lattice = cubic_lattice(20.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj[0, 1]

    def test_min_length_filtering(self):
        """Bond shorter than min_length is excluded."""
        species = ["C", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        specs = [BondSpec(species=("C", "O"), max_length=1.5, min_length=0.8)]
        lattice = cubic_lattice(20.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj.nnz == 0

    def test_first_match_wins(self):
        """First matching spec claims the pair; later specs don't add it."""
        species = ["C", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # First spec: min_length too high, doesn't match.
        # Second spec: matches.
        specs = [
            BondSpec(species=("C", "O"), max_length=1.5, min_length=1.2),
            BondSpec(species=("C", "O"), max_length=1.5),
        ]
        lattice = cubic_lattice(20.0)

        adj = find_bonds(species, coords, specs, lattice)
        # The pair is not claimed by spec 0 (too short), so spec 1 picks it up.
        assert adj[0, 1]

    def test_empty_species(self):
        adj = find_bonds([], np.zeros((0, 3)), [BondSpec(("C", "O"), 1.5)], cubic_lattice(20.0))
        assert adj.shape == (0, 0)

    def test_empty_specs(self):
        adj = find_bonds(["C"], np.array([[0.0, 0.0, 0.0]]), [], cubic_lattice(20.0))
        assert adj.shape == (1, 1)
        assert adj.nnz == 0


class TestFindBondsPBC:
    """Test bond detection across periodic boundaries."""

    def test_bonded_across_boundary(self):
        """Two atoms near opposite faces of a 10 A cubic cell."""
        species = ["C", "O"]
        # Atom 0 at x=0.5, atom 1 at x=9.5 -> MIC distance = 1.0 A.
        coords = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]
        lattice = cubic_lattice(10.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj[0, 1]

    def test_not_bonded_direct_but_close_in_mic(self):
        """Atoms far apart in direct space but close under MIC."""
        species = ["C", "O"]
        coords = np.array([[0.2, 5.0, 5.0], [9.9, 5.0, 5.0]])
        # Direct distance = 9.7, MIC distance = 0.3.
        specs = [BondSpec(species=("C", "O"), max_length=0.5)]
        lattice = cubic_lattice(10.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj[0, 1]

    def test_mic_violation_raises(self):
        """Bond spec longer than inscribed sphere radius raises."""
        species = ["C", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # Cell is 4 A cubic -> inscribed radius = 2.0 A.
        specs = [BondSpec(species=("C", "O"), max_length=2.5)]
        lattice = cubic_lattice(4.0)

        with pytest.raises(ValueError, match="inscribed sphere"):
            find_bonds(species, coords, specs, lattice)
