"""Numba-accelerated batch bond detection and clustering.

This module is an internal implementation detail.  Use
:func:`md_clusters.trajectory.analyse_trajectory` instead.
"""

from __future__ import annotations

from fnmatch import fnmatch

import numpy as np

from md_clusters.bonds import _inscribed_sphere_radius
from md_clusters.types import BondSpec

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _build_species_masks(
    species: list[str],
    bond_specs: list[BondSpec],
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-atom species masks for the numba kernel.

    Returns:
        ``(match_a, match_b)`` each of shape ``(n_specs, n_atoms)``.
    """
    unique_species = list(set(species))
    n_atoms = len(species)
    n_specs = len(bond_specs)
    match_a = np.zeros((n_specs, n_atoms), dtype=np.bool_)
    match_b = np.zeros((n_specs, n_atoms), dtype=np.bool_)

    for s, spec in enumerate(bond_specs):
        sp_a, sp_b = spec.species
        set_a = {sp for sp in unique_species if fnmatch(sp, sp_a)}
        set_b = {sp for sp in unique_species if fnmatch(sp, sp_b)}
        for i, sp in enumerate(species):
            match_a[s, i] = sp in set_a
            match_b[s, i] = sp in set_b

    return match_a, match_b


def _make_numba_cluster_batch_kernel():
    """Build a parallel kernel that finds bonds and clusters in one pass.

    Uses union-find (disjoint set) per frame so that clustering is done
    inside the ``numba.prange`` loop without needing sparse matrices or
    scipy.
    """

    @numba.njit(parallel=True)
    def _find_clusters_batch_kernel(
        all_coords,    # (n_frames, n, 3) float64
        all_lattices,  # (n_frames, 3, 3) float64
        all_lat_invs,  # (n_frames, 3, 3) float64
        spec_min_sq,   # (n_specs,) float64
        spec_max_sq,   # (n_specs,) float64
        match_a,       # (n_specs, n) bool
        match_b,       # (n_specs, n) bool
        all_labels,    # (n_frames, n) int64 — output
        all_n_clusters,  # (n_frames,) int64 — output
    ):
        n_frames = all_coords.shape[0]
        n = all_coords.shape[1]
        n_specs = spec_min_sq.shape[0]

        for f in numba.prange(n_frames):
            coords = all_coords[f]
            lattice = all_lattices[f]
            lat_inv = all_lat_invs[f]

            # Initialise union-find: parent[i] = i.
            parent = np.empty(n, dtype=np.int64)
            rank = np.zeros(n, dtype=np.int64)
            for i in range(n):
                parent[i] = i

            for i in range(n):
                for j in range(i + 1, n):
                    dx = coords[i, 0] - coords[j, 0]
                    dy = coords[i, 1] - coords[j, 1]
                    dz = coords[i, 2] - coords[j, 2]

                    fx = dx * lat_inv[0, 0] + dy * lat_inv[1, 0] + dz * lat_inv[2, 0]
                    fy = dx * lat_inv[0, 1] + dy * lat_inv[1, 1] + dz * lat_inv[2, 1]
                    fz = dx * lat_inv[0, 2] + dy * lat_inv[1, 2] + dz * lat_inv[2, 2]

                    fx -= round(fx)
                    fy -= round(fy)
                    fz -= round(fz)

                    cx = fx * lattice[0, 0] + fy * lattice[1, 0] + fz * lattice[2, 0]
                    cy = fx * lattice[0, 1] + fy * lattice[1, 1] + fz * lattice[2, 1]
                    cz = fx * lattice[0, 2] + fy * lattice[1, 2] + fz * lattice[2, 2]

                    dist_sq = cx * cx + cy * cy + cz * cz

                    for s in range(n_specs):
                        if dist_sq < spec_min_sq[s] or dist_sq > spec_max_sq[s]:
                            continue
                        if (match_a[s, i] and match_b[s, j]) or (
                            match_a[s, j] and match_b[s, i]
                        ):
                            # Union i and j (with path compression + rank).
                            # Find root of i.
                            ri = i
                            while parent[ri] != ri:
                                parent[ri] = parent[parent[ri]]
                                ri = parent[ri]
                            # Find root of j.
                            rj = j
                            while parent[rj] != rj:
                                parent[rj] = parent[parent[rj]]
                                rj = parent[rj]
                            if ri != rj:
                                if rank[ri] < rank[rj]:
                                    parent[ri] = rj
                                elif rank[ri] > rank[rj]:
                                    parent[rj] = ri
                                else:
                                    parent[rj] = ri
                                    rank[ri] += 1
                            break

            # Final path compression: find root for every atom.
            for i in range(n):
                ri = i
                while parent[ri] != ri:
                    ri = parent[ri]
                parent[i] = ri

            # Canonicalise labels: cluster 0 = lowest atom index.
            label_map = np.full(n, -1, dtype=np.int64)
            next_label = np.int64(0)
            for i in range(n):
                root = parent[i]
                if label_map[root] == -1:
                    label_map[root] = next_label
                    next_label += 1
                all_labels[f, i] = label_map[root]

            all_n_clusters[f] = next_label

    return _find_clusters_batch_kernel


_numba_cluster_batch_kernel = None


def _get_numba_cluster_batch_kernel():
    global _numba_cluster_batch_kernel
    if _numba_cluster_batch_kernel is None:
        _numba_cluster_batch_kernel = _make_numba_cluster_batch_kernel()
    return _numba_cluster_batch_kernel


def _find_clusters_batch(
    species: list[str],
    all_coords: np.ndarray,
    bond_specs: list[BondSpec],
    all_lattices: np.ndarray,
    species_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect bonds and identify clusters across frames in one parallel pass.

    Combines bond detection and connected-component analysis using
    union-find inside a ``numba.prange`` loop.  This avoids building
    sparse adjacency matrices entirely, so is faster than calling
    :func:`find_bonds` followed by :func:`find_clusters` on each frame.

    Args:
        species: Species labels, length ``n_atoms``.
        all_coords: Coordinates, shape ``(n_frames, n_atoms, 3)``.
        bond_specs: Bond detection rules, applied in order.
        all_lattices: Lattice matrices, shape ``(n_frames, 3, 3)``.
        species_masks: Pre-computed per-atom species masks from
            :func:`_build_species_masks`.

    Returns:
        ``(n_clusters, labels)`` where *n_clusters* has shape
        ``(n_frames,)`` and *labels* has shape ``(n_frames, n_atoms)``.
        Labels are canonicalised so that cluster 0 contains the atom
        with the lowest index in each frame.

    Raises:
        ImportError: If numba is not installed.
        ValueError: If MIC assumption is violated for any frame.
    """
    if not HAS_NUMBA:
        raise ImportError("numba is required for _find_clusters_batch")

    all_coords = np.asarray(all_coords, dtype=float)
    all_lattices = np.asarray(all_lattices, dtype=float)
    n_frames = all_coords.shape[0]
    n_atoms = len(species)

    if n_atoms == 0 or len(bond_specs) == 0:
        labels = np.zeros((n_frames, n_atoms), dtype=np.int64)
        n_clusters = np.zeros(n_frames, dtype=np.int64)
        return n_clusters, labels

    # Validate MIC assumption for all frames.
    max_bond = max(s.max_length for s in bond_specs)
    for f in range(n_frames):
        r_ins = _inscribed_sphere_radius(all_lattices[f])
        if max_bond >= r_ins:
            raise ValueError(
                f"Frame {f}: max bond length ({max_bond:.3f}) >= inscribed "
                f"sphere radius ({r_ins:.3f}).  The unit cell is too small "
                f"for the MIC assumption."
            )

    kernel = _get_numba_cluster_batch_kernel()

    all_lat_invs = np.linalg.inv(all_lattices)

    spec_min_sq = np.array([s.min_length ** 2 for s in bond_specs])
    spec_max_sq = np.array([s.max_length ** 2 for s in bond_specs])

    if species_masks is None:
        match_a, match_b = _build_species_masks(species, bond_specs)
    else:
        match_a, match_b = species_masks

    all_labels = np.empty((n_frames, n_atoms), dtype=np.int64)
    all_n_clusters = np.empty(n_frames, dtype=np.int64)

    kernel(
        all_coords, all_lattices, all_lat_invs,
        spec_min_sq, spec_max_sq,
        match_a, match_b,
        all_labels, all_n_clusters,
    )

    return all_n_clusters, all_labels
