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

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


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


def build_pair_masks(
    species: list[str],
    bond_specs: list[BondSpec],
) -> list[np.ndarray]:
    """Pre-compute species pair masks for reuse across frames.

    For a trajectory where species labels are constant, calling this
    once and passing the result to :func:`find_bonds` avoids redundant
    mask computation on every frame.

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


# ---------------------------------------------------------------------------
# Numba-accelerated bond detection
# ---------------------------------------------------------------------------


def _make_numba_kernel():
    """Build the numba-jitted kernel (deferred to avoid import-time cost)."""

    @numba.njit
    def _find_bonds_numba_kernel(
        coords,       # (n, 3) float64
        lattice,      # (3, 3) float64
        lat_inv,      # (3, 3) float64
        spec_min_sq,  # (n_specs,) float64
        spec_max_sq,  # (n_specs,) float64
        match_a,      # (n_specs, n) bool
        match_b,      # (n_specs, n) bool
        row_buf,      # (max_bonds,) int64 — output
        col_buf,      # (max_bonds,) int64 — output
    ):
        n = coords.shape[0]
        n_specs = spec_min_sq.shape[0]
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Cartesian difference.
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dz = coords[i, 2] - coords[j, 2]

                # To fractional coordinates: diff @ lat_inv.
                fx = dx * lat_inv[0, 0] + dy * lat_inv[1, 0] + dz * lat_inv[2, 0]
                fy = dx * lat_inv[0, 1] + dy * lat_inv[1, 1] + dz * lat_inv[2, 1]
                fz = dx * lat_inv[0, 2] + dy * lat_inv[1, 2] + dz * lat_inv[2, 2]

                # Minimum image convention.
                fx -= round(fx)
                fy -= round(fy)
                fz -= round(fz)

                # Back to Cartesian: frac @ lattice.
                cx = fx * lattice[0, 0] + fy * lattice[1, 0] + fz * lattice[2, 0]
                cy = fx * lattice[0, 1] + fy * lattice[1, 1] + fz * lattice[2, 1]
                cz = fx * lattice[0, 2] + fy * lattice[1, 2] + fz * lattice[2, 2]

                dist_sq = cx * cx + cy * cy + cz * cz

                # First-match-wins over specs.
                for s in range(n_specs):
                    if dist_sq < spec_min_sq[s] or dist_sq > spec_max_sq[s]:
                        continue
                    if (match_a[s, i] and match_b[s, j]) or (
                        match_a[s, j] and match_b[s, i]
                    ):
                        row_buf[count] = i
                        col_buf[count] = j
                        count += 1
                        break

        return count

    return _find_bonds_numba_kernel


# Lazy singleton for the compiled kernel.
_numba_kernel = None


def _get_numba_kernel():
    global _numba_kernel
    if _numba_kernel is None:
        _numba_kernel = _make_numba_kernel()
    return _numba_kernel


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


def _find_bonds_numba(
    species: list[str],
    coords: np.ndarray,
    bond_specs: list[BondSpec],
    lattice: np.ndarray,
    species_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> sparse.csr_matrix:
    """Numba-accelerated bond detection."""
    n_atoms = len(species)
    kernel = _get_numba_kernel()

    lat_inv = np.linalg.inv(lattice)

    spec_min_sq = np.array([s.min_length ** 2 for s in bond_specs])
    spec_max_sq = np.array([s.max_length ** 2 for s in bond_specs])

    if species_masks is not None:
        match_a, match_b = species_masks
    else:
        match_a, match_b = _build_species_masks(species, bond_specs)

    max_bonds = n_atoms * (n_atoms - 1) // 2
    row_buf = np.empty(max_bonds, dtype=np.int64)
    col_buf = np.empty(max_bonds, dtype=np.int64)

    count = kernel(
        coords, lattice, lat_inv,
        spec_min_sq, spec_max_sq,
        match_a, match_b,
        row_buf, col_buf,
    )

    row = row_buf[:count]
    col = col_buf[:count]
    data = np.ones(count, dtype=bool)

    # Build symmetric sparse matrix from upper-triangle hits.
    adj = sparse.coo_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([row, col]), np.concatenate([col, row]))),
        shape=(n_atoms, n_atoms),
    )
    return adj.tocsr()


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------


def find_bonds(
    species: list[str],
    coords: np.ndarray,
    bond_specs: list[BondSpec],
    lattice: np.ndarray,
    pair_masks: list[np.ndarray] | None = None,
    backend: str = "numpy",
    species_masks: tuple[np.ndarray, np.ndarray] | None = None,
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
        pair_masks: Pre-computed species pair masks from
            :func:`build_pair_masks`.  When provided, species mask
            computation is skipped.  Must have one entry per spec.
            Only used by the ``"numpy"`` backend.
        backend: ``"numpy"`` (default) or ``"numba"``.
        species_masks: Pre-computed per-atom species masks from
            :func:`_build_species_masks`.  Only used by the ``"numba"``
            backend.

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

    if backend == "numba":
        if not HAS_NUMBA:
            raise ImportError("numba is required for backend='numba'")
        return _find_bonds_numba(
            species, coords, bond_specs, lattice,
            species_masks=species_masks,
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

    if pair_masks is None:
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

    # Pre-compute species pair masks once (species are constant across frames).
    species = [str(site.specie) for site in structures[0]]
    pair_masks = build_pair_masks(species, bond_specs)

    for structure in structures:
        coords = structure.cart_coords
        lattice = structure.lattice.matrix

        adjacency = find_bonds(
            species, coords, bond_specs, lattice, pair_masks=pair_masks,
        )
        n_clusters, labels = find_clusters(adjacency)
        results.append(FrameResult(species, adjacency, n_clusters, labels))

    return results
