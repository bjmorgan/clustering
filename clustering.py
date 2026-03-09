"""Atomic cluster analysis for MD trajectories.

Identifies bonded atom pairs per frame using the minimum image convention
(MIC) and groups them into clusters of contiguously bonded atoms via
connected-component analysis on the sparse adjacency matrix.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fnmatch import fnmatch

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class BondSpec:
    """Declarative rule for bond detection between a species pair.

    The species pair is stored in sorted order so that the rule is
    invariant under exchange of the two labels.  Species names support
    fnmatch-style wildcards (``*``, ``?``).

    Attributes:
        species: Sorted pair of species patterns.
        max_length: Maximum bond length threshold (angstroms).
        min_length: Minimum bond length threshold (angstroms).
    """

    species: tuple[str, str]
    max_length: float
    min_length: float = 0.0

    def __post_init__(self) -> None:
        # Sort the species pair for order-invariance.
        a, b = sorted(self.species)
        object.__setattr__(self, "species", (a, b))

        if self.max_length <= 0:
            raise ValueError(
                f"max_length must be positive, got {self.max_length}"
            )
        if self.min_length < 0:
            raise ValueError(
                f"min_length must be non-negative, got {self.min_length}"
            )
        if self.min_length > self.max_length:
            raise ValueError(
                f"min_length ({self.min_length}) must not exceed "
                f"max_length ({self.max_length})"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------


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

    # Accumulate hits across all specs (first-match-wins).
    bonded = np.zeros((n_atoms, n_atoms), dtype=bool)

    for spec in bond_specs:
        pair_mask = _species_pair_mask(spec, species, unique_species)
        dist_ok = (dist >= spec.min_length) & (dist <= spec.max_length)
        hits = upper & pair_mask & dist_ok & ~claimed
        claimed |= hits
        bonded |= hits

    # Build symmetric sparse matrix.
    bonded_symmetric = bonded | bonded.T
    return sparse.csr_matrix(bonded_symmetric)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def find_clusters(adjacency: sparse.csr_matrix) -> tuple[int, np.ndarray]:
    """Identify connected components in a bond adjacency matrix.

    Labels are canonicalised so that cluster 0 contains the atom with
    the lowest index, cluster 1 the next unclaimed atom, and so on.

    Args:
        adjacency: Symmetric sparse boolean adjacency matrix from
            :func:`find_bonds`.

    Returns:
        ``(n_clusters, labels)`` where *labels* is an integer array of
        shape ``(n_atoms,)`` assigning each atom to a cluster.
    """
    n_clusters, labels = connected_components(
        adjacency, directed=False, return_labels=True,
    )

    # scipy already labels by first-encountered atom index when using
    # the default BFS, but we canonicalise explicitly to be safe.
    first_occurrence = np.full(n_clusters, fill_value=adjacency.shape[0])
    for i, label in enumerate(labels):
        if i < first_occurrence[label]:
            first_occurrence[label] = i
    rank = np.argsort(first_occurrence)
    inverse = np.empty_like(rank)
    inverse[rank] = np.arange(n_clusters)
    labels = inverse[labels]

    return int(n_clusters), labels


# ---------------------------------------------------------------------------
# Composition analysis
# ---------------------------------------------------------------------------


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
) -> dict[str, int]:
    """Count cluster formulae across all clusters in a frame.

    Args:
        species: Species labels, length ``n_atoms``.
        labels: Cluster label per atom from :func:`find_clusters`.

    Returns:
        Mapping from canonical formula string (hill order) to the number
        of times that formula appears.  e.g. ``{"CO2": 3, "H2O": 5}``.
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

    return dict(formula_counts)


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------


@dataclass
class FrameResult:
    """Per-frame analysis result.

    Attributes:
        species: Species labels, length ``n_atoms``.
        adjacency: Sparse boolean adjacency matrix, shape
            ``(n_atoms, n_atoms)``.
        n_clusters: Number of connected components.
        labels: Cluster label per atom, shape ``(n_atoms,)``.
    """

    species: list[str]
    adjacency: sparse.csr_matrix
    n_clusters: int
    labels: np.ndarray

    @property
    def composition(self) -> dict[str, int]:
        """Cluster formula counts for this frame."""
        return cluster_composition(self.species, self.labels)


def analyse_trajectory(
    structures: list,
    bond_specs: list[BondSpec],
) -> list[FrameResult]:
    """Analyse an MD trajectory for atomic clusters.

    Args:
        structures: List of ``pymatgen.core.Structure`` objects, one per
            frame.  Each carries its own lattice (supporting NPT
            trajectories).
        bond_specs: Bond detection rules.

    Returns:
        One :class:`FrameResult` per frame.
    """
    results: list[FrameResult] = []

    for structure in structures:
        species = [str(site.specie) for site in structure]
        coords = structure.cart_coords
        lattice = structure.lattice.matrix

        adjacency = find_bonds(species, coords, bond_specs, lattice)
        n_clusters, labels = find_clusters(adjacency)
        results.append(FrameResult(species, adjacency, n_clusters, labels))

    return results
