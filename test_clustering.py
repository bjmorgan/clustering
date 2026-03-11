"""Tests for the atomic cluster analysis module."""

import numpy as np
import pytest
from collections import Counter
from scipy import sparse

from clustering import (
    BondSpec,
    TrajectoryResult,
    _hill_formula,
    analyse_trajectory,
    cluster_composition,
    find_bonds,
    find_clusters,
)

numba = pytest.importorskip("numba")


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


# ---------------------------------------------------------------------------
# find_bonds — periodic boundary conditions
# ---------------------------------------------------------------------------


class TestFindBondsPBC:
    """Test bond detection across periodic boundaries."""

    def test_bonded_across_boundary(self):
        """Two atoms near opposite faces of a 10 A cubic cell."""
        species = ["C", "O"]
        # Atom 0 at x=0.5, atom 1 at x=9.5 -> MIC distance = 1.0 A.
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
        # Cell is 4 A cubic -> inscribed radius = 2.0 A.
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
        """No bonds -> each atom is its own cluster."""
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


# ---------------------------------------------------------------------------
# analyse_trajectory
# ---------------------------------------------------------------------------


class TestAnalyseTrajectory:
    """Test trajectory analysis with auto-selected backend."""

    def test_matches_sequential_find_bonds(self):
        """Batch results match frame-by-frame find_bonds + find_clusters."""
        rng = np.random.default_rng(99)
        n = 30
        n_frames = 5
        species = rng.choice(["C", "O", "H"], size=n).tolist()
        lattice = _cubic_lattice(10.0)
        specs = [
            BondSpec(species=("C", "O"), max_length=1.6),
            BondSpec(species=("O", "H"), max_length=1.2),
            BondSpec(species=("*", "*"), max_length=1.8),
        ]

        all_coords = rng.uniform(0, 8, size=(n_frames, n, 3))
        all_lattices = np.broadcast_to(lattice, (n_frames, 3, 3)).copy()

        # Reference: sequential find_bonds + find_clusters.
        ref_labels = []
        ref_n_clusters = []
        for f in range(n_frames):
            adj = find_bonds(species, all_coords[f], specs, all_lattices[f])
            nc, lab = find_clusters(adj)
            ref_labels.append(lab)
            ref_n_clusters.append(nc)

        # analyse_trajectory.
        result = analyse_trajectory(
            species, all_coords, specs, all_lattices,
        )

        assert isinstance(result, TrajectoryResult)
        for f in range(n_frames):
            assert result.n_clusters[f] == ref_n_clusters[f], (
                f"Frame {f}: n_clusters differ "
                f"({result.n_clusters[f]} vs {ref_n_clusters[f]})"
            )
            assert np.array_equal(result.labels[f], ref_labels[f]), (
                f"Frame {f}: labels differ"
            )

    def test_all_isolated(self):
        """No bonds at all -- each atom is its own cluster."""
        species = ["C", "O", "H"]
        coords = np.array([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]]])
        lattice = np.array([_cubic_lattice(20.0)])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]

        result = analyse_trajectory(species, coords, specs, lattice)

        assert result.n_clusters[0] == 3
        assert len(set(result.labels[0])) == 3

    def test_single_cluster(self):
        """All atoms bonded into one cluster."""
        species = ["C", "O"]
        coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        lattice = np.array([_cubic_lattice(20.0)])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]

        result = analyse_trajectory(species, coords, specs, lattice)

        assert result.n_clusters[0] == 1
        assert result.labels[0, 0] == result.labels[0, 1] == 0

    def test_variable_lattice(self):
        """Different lattice per frame (NPT-like)."""
        species = ["C", "O"]
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]
        coords_f0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords_f1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        all_coords = np.stack([coords_f0, coords_f1])
        all_lattices = np.stack([_cubic_lattice(20.0), _cubic_lattice(15.0)])

        result = analyse_trajectory(species, all_coords, specs, all_lattices)

        assert result.n_clusters[0] == 1
        assert result.n_clusters[1] == 1


# ---------------------------------------------------------------------------
# TrajectoryResult
# ---------------------------------------------------------------------------


class TestTrajectoryResult:
    """Test TrajectoryResult dataclass."""

    def test_composition(self):
        """composition(frame) returns formula counts for that frame."""
        species = ["C", "O", "O", "H", "H", "O"]
        labels = np.array([[0, 0, 0, 1, 1, 1]])
        n_clusters = np.array([2])

        result = TrajectoryResult(species, n_clusters, labels)
        assert result.composition(0) == Counter({"CO2": 1, "H2O": 1})

    def test_composition_returns_counter(self):
        """composition() returns a Counter."""
        species = ["C", "O", "O"]
        labels = np.array([[0, 0, 0]])
        n_clusters = np.array([1])

        result = TrajectoryResult(species, n_clusters, labels)
        assert isinstance(result.composition(0), Counter)
