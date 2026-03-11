"""Tests for trajectory analysis."""

from collections import Counter

import numpy as np
import pytest

from md_clusters import (
    BondSpec,
    TrajectoryResult,
    analyse_trajectory,
    find_bonds,
    find_clusters,
)
from tests.conftest import cubic_lattice

numba = pytest.importorskip("numba")


class TestAnalyseTrajectory:
    """Test trajectory analysis with auto-selected backend."""

    def test_matches_sequential_find_bonds(self):
        """Batch results match frame-by-frame find_bonds + find_clusters."""
        rng = np.random.default_rng(99)
        n = 30
        n_frames = 5
        species = rng.choice(["C", "O", "H"], size=n).tolist()
        lattice = cubic_lattice(10.0)
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
        lattice = np.array([cubic_lattice(20.0)])
        specs = [BondSpec(species=("C", "O"), max_length=1.5)]

        result = analyse_trajectory(species, coords, specs, lattice)

        assert result.n_clusters[0] == 3
        assert len(set(result.labels[0])) == 3

    def test_single_cluster(self):
        """All atoms bonded into one cluster."""
        species = ["C", "O"]
        coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        lattice = np.array([cubic_lattice(20.0)])
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
        all_lattices = np.stack([cubic_lattice(20.0), cubic_lattice(15.0)])

        result = analyse_trajectory(species, all_coords, specs, all_lattices)

        assert result.n_clusters[0] == 1
        assert result.n_clusters[1] == 1


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
