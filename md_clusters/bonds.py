"""Single-frame bond detection under the minimum image convention."""

from __future__ import annotations

from fnmatch import fnmatch

import numpy as np
from scipy import sparse

from md_clusters.types import BondSpec


def _inscribed_sphere_radius(lattice: np.ndarray) -> float:
    """Radius of the largest sphere fitting inside the unit cell.

    For lattice vectors **a**, **b**, **c** (rows of *lattice*), this is
    ``min(h_a, h_b, h_c) / 2`` where each ``h_i`` is the perpendicular
    distance between the pair of faces normal to the *i*-th reciprocal
    lattice direction.
    """
    a, b, c = lattice[0], lattice[1], lattice[2]
    volume = abs(np.dot(a, np.cross(b, c)))
    heights = np.array([
        volume / np.linalg.norm(np.cross(b, c)),
        volume / np.linalg.norm(np.cross(a, c)),
        volume / np.linalg.norm(np.cross(a, b)),
    ])
    return float(heights.min() / 2.0)


def _species_pair_mask(
    spec: BondSpec,
    species: list[str],
    unique_species: list[str],
) -> np.ndarray:
    """Build a boolean (n, n) mask for species pairs matching *spec*."""
    sp_a, sp_b = spec.species
    match_a = {s for s in unique_species if fnmatch(s, sp_a)}
    match_b = {s for s in unique_species if fnmatch(s, sp_b)}
    mask_a = np.array([s in match_a for s in species])
    mask_b = np.array([s in match_b for s in species])
    return (
        (mask_a[:, np.newaxis] & mask_b[np.newaxis, :])
        | (mask_b[:, np.newaxis] & mask_a[np.newaxis, :])
    )


def _build_pair_masks(
    species: list[str],
    bond_specs: list[BondSpec],
) -> list[np.ndarray]:
    """Pre-compute species pair masks for reuse across frames.

    Args:
        species: Species labels, length ``n_atoms``.
        bond_specs: Bond detection rules.

    Returns:
        List of boolean arrays, one per spec, each shape ``(n_atoms, n_atoms)``.
    """
    unique_species = list(set(species))
    return [
        _species_pair_mask(spec, species, unique_species)
        for spec in bond_specs
    ]


def find_bonds(
    species: list[str],
    coords: np.ndarray,
    bond_specs: list[BondSpec],
    lattice: np.ndarray,
) -> sparse.csr_matrix:
    """Detect bonded atom pairs under the minimum image convention.

    Builds a symmetric sparse adjacency matrix where entry ``(i, j)`` is
    ``True`` if atoms *i* and *j* are bonded according to at least one
    bond spec.  First-match-wins ordering applies: once a pair is claimed
    by one spec, later specs cannot override it.

    For trajectory analysis, prefer :func:`analyse_trajectory` which
    uses a faster parallel backend when available.

    Args:
        species: Species labels, length ``n_atoms``.
        coords: Cartesian coordinates, shape ``(n_atoms, 3)``.
        bond_specs: Bond detection rules, applied in order.
        lattice: 3x3 matrix of lattice vectors (row vectors).

    Returns:
        Symmetric boolean sparse matrix, shape ``(n_atoms, n_atoms)``.

    Raises:
        ValueError: If the longest bond spec exceeds the inscribed
            sphere radius (MIC assumption violated).
    """
    coords = np.asarray(coords, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    n_atoms = len(species)

    if n_atoms == 0 or len(bond_specs) == 0:
        return sparse.csr_matrix((n_atoms, n_atoms), dtype=bool)

    # Validate MIC assumption.
    max_bond = max(s.max_length for s in bond_specs)
    r_ins = _inscribed_sphere_radius(lattice)
    if max_bond >= r_ins:
        raise ValueError(
            f"Max bond length ({max_bond:.3f}) >= inscribed sphere "
            f"radius ({r_ins:.3f}).  The unit cell is too small for "
            f"the MIC assumption."
        )

    # Pairwise differences in Cartesian, then fractional.
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    lat_inv = np.linalg.inv(lattice)
    diff_frac = diff @ lat_inv

    # Apply minimum image convention.
    images = np.rint(diff_frac)
    mic_frac = diff_frac - images
    mic_cart = mic_frac @ lattice
    dist = np.linalg.norm(mic_cart, axis=2)

    # Only consider upper triangle (avoid double-counting and self-bonds).
    upper = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)
    claimed = np.zeros((n_atoms, n_atoms), dtype=bool)

    unique_species = list(set(species))
    pair_masks = [
        _species_pair_mask(spec, species, unique_species)
        for spec in bond_specs
    ]

    # Accumulate hits across all specs (first-match-wins).
    bonded = np.zeros((n_atoms, n_atoms), dtype=bool)

    for spec, pair_mask in zip(bond_specs, pair_masks):
        dist_ok = (dist >= spec.min_length) & (dist <= spec.max_length)
        hits = upper & pair_mask & dist_ok & ~claimed
        claimed |= hits
        bonded |= hits

    # Build symmetric sparse matrix.
    bonded_symmetric = bonded | bonded.T
    return sparse.csr_matrix(bonded_symmetric)
