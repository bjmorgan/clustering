"""Trajectory analysis and convenience wrappers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from md_clusters.bonds import _build_pair_masks, find_bonds
from md_clusters.clusters import find_clusters
from md_clusters.composition import cluster_composition
from md_clusters.types import BondSpec


@dataclass
class TrajectoryResult:
    """Result of trajectory cluster analysis.

    Attributes:
        species: Species labels, length ``n_atoms``.
        n_clusters: Cluster count per frame, shape ``(n_frames,)``.
        labels: Cluster label per atom per frame, shape
            ``(n_frames, n_atoms)``.  Labels are canonicalised so that
            cluster 0 contains the atom with the lowest index.
    """

    species: list[str]
    n_clusters: np.ndarray
    labels: np.ndarray

    def composition(self, frame: int) -> Counter[str]:
        """Cluster formula counts for a single frame.

        Args:
            frame: Frame index.

        Returns:
            Counter mapping canonical formula string (hill order) to
            the number of times that formula appears.
        """
        return cluster_composition(self.species, self.labels[frame])


def analyse_trajectory(
    species: list[str],
    coords: np.ndarray,
    bond_specs: list[BondSpec],
    lattices: np.ndarray,
) -> TrajectoryResult:
    """Analyse an MD trajectory for atomic clusters.

    Auto-selects the fastest available backend: numba parallel
    union-find if numba is installed, otherwise numpy with scipy
    clustering.

    Args:
        species: Species labels, length ``n_atoms``.
        coords: Cartesian coordinates, shape ``(n_frames, n_atoms, 3)``.
        bond_specs: Bond detection rules.
        lattices: Lattice matrices, shape ``(n_frames, 3, 3)``.

    Returns:
        A :class:`TrajectoryResult` with per-frame cluster labels.
    """
    coords = np.asarray(coords, dtype=float)
    lattices = np.asarray(lattices, dtype=float)

    from md_clusters._numba import HAS_NUMBA

    if HAS_NUMBA:
        from md_clusters._numba import _find_clusters_batch

        n_clusters, labels = _find_clusters_batch(
            species, coords, bond_specs, lattices,
        )
        return TrajectoryResult(species, n_clusters, labels)

    # Numpy fallback: loop over frames.
    n_frames = coords.shape[0]
    pair_masks = _build_pair_masks(species, bond_specs)
    all_labels = []
    all_n_clusters = []

    for f in range(n_frames):
        adjacency = find_bonds(species, coords[f], bond_specs, lattices[f])
        nc, lab = find_clusters(adjacency)
        all_labels.append(lab)
        all_n_clusters.append(nc)

    return TrajectoryResult(
        species,
        np.array(all_n_clusters),
        np.stack(all_labels),
    )


def analyse_structures(
    structures: list[Any],
    bond_specs: list[BondSpec],
) -> TrajectoryResult:
    """Analyse a trajectory of pymatgen Structure objects.

    Convenience wrapper around :func:`analyse_trajectory` that extracts
    species, coordinates, and lattice matrices from pymatgen structures.

    Args:
        structures: One or more ``pymatgen.core.Structure`` objects.
            A single structure is treated as a one-frame trajectory.
        bond_specs: Bond detection rules.

    Returns:
        A :class:`TrajectoryResult` with per-frame cluster labels.
    """
    # Accept a single structure.
    if not isinstance(structures, (list, tuple)):
        structures = [structures]

    species = [site.specie.symbol for site in structures[0]]
    coords = np.array([s.cart_coords for s in structures])
    lattices = np.array([s.lattice.matrix for s in structures])
    return analyse_trajectory(species, coords, bond_specs, lattices)


def analyse_atoms(
    atoms: Any,
    bond_specs: list[BondSpec],
) -> TrajectoryResult:
    """Analyse a trajectory of ASE Atoms objects.

    Convenience wrapper around :func:`analyse_trajectory` that extracts
    species, coordinates, and lattice matrices from ASE Atoms objects.

    Args:
        atoms: One or more ``ase.Atoms`` objects, or an
            ``ase.io.Trajectory``.  A single ``Atoms`` is treated as a
            one-frame trajectory.
        bond_specs: Bond detection rules.

    Returns:
        A :class:`TrajectoryResult` with per-frame cluster labels.
    """
    # Accept a single Atoms object.
    if hasattr(atoms, "get_chemical_symbols"):
        atoms = [atoms]
    # Accept any iterable (including ase.io.Trajectory) — materialise it.
    if not isinstance(atoms, list):
        atoms = list(atoms)

    species = atoms[0].get_chemical_symbols()
    coords = np.array([a.positions for a in atoms])
    lattices = np.array([a.cell.array for a in atoms])
    return analyse_trajectory(species, coords, bond_specs, lattices)
