"""Tests for the atomic cluster analysis module."""

import numpy as np
import pytest
from scipy import sparse

from clustering import (
    BondSpec,
    FrameResult,
    _hill_formula,
    build_pair_masks,
    cluster_composition,
    find_bonds,
    find_clusters,
)
from collections import Counter


# ---------------------------------------------------------------------------
# BondSpec
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# find_bonds — non-periodic (large cell, no wrapping)
# ---------------------------------------------------------------------------


def _cubic_lattice(a: float) -> np.ndarray:
    """Return a cubic lattice matrix with side length *a*."""
    return np.eye(3) * a


class TestFindBondsSimple:
    """Test bond detection on simple geometries in a large cell."""

    def test_two_bonded_atoms(self):
        """Two atoms 1.0 A apart, spec allows up to 1.5 A."""
        species = ["C", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]
        lattice = _cubic_lattice(20.0)

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
        lattice = _cubic_lattice(20.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj.nnz == 0

    def test_species_mismatch(self):
        """Two close atoms but wrong species for the spec."""
        species = ["C", "C"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]
        lattice = _cubic_lattice(20.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj.nnz == 0

    def test_wildcard_species(self):
        """Wildcard spec matches any species."""
        species = ["Zr", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        specs = [BondSpec(species=("*", "*"), max_length=1.5)]
        lattice = _cubic_lattice(20.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj[0, 1]

    def test_min_length_filtering(self):
        """Bond shorter than min_length is excluded."""
        species = ["C", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        specs = [BondSpec(species=("C", "O"), max_length=1.5, min_length=0.8)]
        lattice = _cubic_lattice(20.0)

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
        lattice = _cubic_lattice(20.0)

        adj = find_bonds(species, coords, specs, lattice)
        # The pair is not claimed by spec 0 (too short), so spec 1 picks it up.
        assert adj[0, 1]

    def test_empty_species(self):
        adj = find_bonds([], np.zeros((0, 3)), [BondSpec(("C", "O"), 1.5)], _cubic_lattice(20.0))
        assert adj.shape == (0, 0)

    def test_empty_specs(self):
        adj = find_bonds(["C"], np.array([[0.0, 0.0, 0.0]]), [], _cubic_lattice(20.0))
        assert adj.shape == (1, 1)
        assert adj.nnz == 0

    def test_precomputed_pair_masks(self):
        """Pre-computed pair masks give the same result as computing inline."""
        species = ["C", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]
        lattice = _cubic_lattice(20.0)

        masks = build_pair_masks(species, specs)
        adj = find_bonds(species, coords, specs, lattice, pair_masks=masks)
        assert adj[0, 1]


# ---------------------------------------------------------------------------
# find_bonds — periodic boundary conditions
# ---------------------------------------------------------------------------


class TestFindBondsPBC:
    """Test bond detection across periodic boundaries."""

    def test_bonded_across_boundary(self):
        """Two atoms near opposite faces of a 10 A cubic cell."""
        species = ["C", "O"]
        # Atom 0 at x=0.5, atom 1 at x=9.5 → MIC distance = 1.0 A.
        coords = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]
        lattice = _cubic_lattice(10.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj[0, 1]

    def test_not_bonded_direct_but_close_in_mic(self):
        """Atoms far apart in direct space but close under MIC."""
        species = ["C", "O"]
        coords = np.array([[0.2, 5.0, 5.0], [9.9, 5.0, 5.0]])
        # Direct distance = 9.7, MIC distance = 0.3.
        specs = [BondSpec(species=("C", "O"), max_length=0.5)]
        lattice = _cubic_lattice(10.0)

        adj = find_bonds(species, coords, specs, lattice)
        assert adj[0, 1]

    def test_mic_violation_raises(self):
        """Bond spec longer than inscribed sphere radius raises."""
        species = ["C", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # Cell is 4 A cubic → inscribed radius = 2.0 A.
        specs = [BondSpec(species=("C", "O"), max_length=2.5)]
        lattice = _cubic_lattice(4.0)

        with pytest.raises(ValueError, match="inscribed sphere"):
            find_bonds(species, coords, specs, lattice)


# ---------------------------------------------------------------------------
# find_clusters
# ---------------------------------------------------------------------------


class TestFindClusters:
    """Test connected component identification."""

    def test_two_clusters(self):
        """5 atoms: (0,1,2) bonded together, (3,4) bonded together."""
        row = [0, 1, 0, 2, 3, 4]
        col = [1, 0, 2, 0, 4, 3]
        data = [True] * 6
        adj = sparse.csr_matrix((data, (row, col)), shape=(5, 5))

        n_clusters, labels = find_clusters(adj)
        assert n_clusters == 2
        # Atoms 0,1,2 share a label; atoms 3,4 share a different one.
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4]
        assert labels[0] != labels[3]

    def test_all_isolated(self):
        """No bonds → each atom is its own cluster."""
        adj = sparse.csr_matrix((4, 4), dtype=bool)

        n_clusters, labels = find_clusters(adj)
        assert n_clusters == 4
        assert len(set(labels)) == 4

    def test_single_cluster(self):
        """All atoms bonded into one cluster."""
        # Chain: 0-1-2-3
        row = [0, 1, 1, 2, 2, 3]
        col = [1, 0, 2, 1, 3, 2]
        data = [True] * 6
        adj = sparse.csr_matrix((data, (row, col)), shape=(4, 4))

        n_clusters, labels = find_clusters(adj)
        assert n_clusters == 1
        assert all(labels == 0)

    def test_labels_canonicalised(self):
        """Label 0 contains atom 0, label 1 the next unclaimed, etc."""
        # Two clusters: {0, 2} and {1, 3}.
        row = [0, 2, 1, 3]
        col = [2, 0, 3, 1]
        data = [True] * 4
        adj = sparse.csr_matrix((data, (row, col)), shape=(4, 4))

        n_clusters, labels = find_clusters(adj)
        assert n_clusters == 2
        assert labels[0] == 0  # Cluster 0 contains atom 0.
        assert labels[1] == 1  # Cluster 1 contains atom 1.
        assert labels[2] == 0
        assert labels[3] == 1


# ---------------------------------------------------------------------------
# cluster_composition
# ---------------------------------------------------------------------------


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
        """Without carbon, H is not given priority — all alphabetical."""
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
        assert result == {"CO2": 2}

    def test_mixed_clusters(self):
        species = ["C", "O", "O", "H", "H", "O"]
        labels = np.array([0, 0, 0, 1, 1, 1])

        result = cluster_composition(species, labels)
        assert result == {"CO2": 1, "H2O": 1}

    def test_isolated_atoms(self):
        species = ["C", "O", "Fe"]
        labels = np.array([0, 1, 2])

        result = cluster_composition(species, labels)
        assert result == {"C": 1, "O": 1, "Fe": 1}

    def test_empty(self):
        result = cluster_composition([], np.array([], dtype=int))
        assert result == {}


# ---------------------------------------------------------------------------
# FrameResult.composition
# ---------------------------------------------------------------------------


class TestFrameResultComposition:
    """Test the composition property on FrameResult."""

    def test_composition_property(self):
        """composition returns the same result as cluster_composition."""
        species = ["C", "O", "O", "H", "H", "O"]
        labels = np.array([0, 0, 0, 1, 1, 1])
        adj = sparse.csr_matrix((6, 6), dtype=bool)

        result = FrameResult(
            species=species, adjacency=adj, n_clusters=2, labels=labels,
        )
        assert result.composition == {"CO2": 1, "H2O": 1}
