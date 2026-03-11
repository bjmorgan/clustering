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


def _build_pair_masks(
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
# Numba-accelerated batch analysis
# ---------------------------------------------------------------------------


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
    :func:`find_bonds_batch` followed by :func:`find_clusters` on each
    frame.

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
) -> Counter[str]:
    """Count cluster formulae across all clusters in a frame.

    Args:
        species: Species labels, length ``n_atoms``.
        labels: Cluster label per atom from :func:`find_clusters`.

    Returns:
        Counter mapping canonical formula string (hill order) to the
        number of times that formula appears.
        e.g. ``Counter({"CO2": 3, "H2O": 5})``.
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

    return formula_counts


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------


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

    if HAS_NUMBA:
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
    structures: list,
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
    atoms,
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
