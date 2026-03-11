# md-clusters

Atomic cluster analysis for molecular dynamics trajectories.

Identifies bonded atom pairs per frame using the minimum image convention
and groups them into clusters of contiguously bonded atoms. Automatically
uses a parallel numba backend when available (~50x faster than numpy).

## Installation

```bash
pip install .              # core (numpy + scipy)
pip install .[numba]       # with numba acceleration
pip install .[ase]         # with ASE convenience wrapper
pip install .[pymatgen]    # with pymatgen convenience wrapper
```

## Quick start

### From ASE

```python
from ase.io import read
from md_clusters import BondSpec, analyse_atoms

frames = read("trajectory.extxyz", index=":")

bond_specs = [
    BondSpec(species=("C", "O"), max_length=1.6),
    BondSpec(species=("C", "H"), max_length=1.2),
    BondSpec(species=("O", "H"), max_length=1.2),
]

result = analyse_atoms(frames, bond_specs)

for formula, count in result.composition(0).most_common():
    print(f"  {formula}: {count}")
```

### From pymatgen

```python
from pymatgen.core import Structure
from md_clusters import BondSpec, analyse_structures

structures = [Structure.from_file(f"POSCAR_{i}") for i in range(100)]

bond_specs = [
    BondSpec(species=("Li", "O"), max_length=2.2),
    BondSpec(species=("P", "O"), max_length=1.7),
]

result = analyse_structures(structures, bond_specs)
print(result.n_clusters)       # cluster count per frame
print(result.composition(0))   # Counter({"LiO4": 2, "PO4": 1, ...})
```

### From raw arrays

```python
import numpy as np
from md_clusters import BondSpec, analyse_trajectory

species = ["C", "O", "O", "H", "H", "O"]
coords = np.random.uniform(0, 10, (50, 6, 3))       # (n_frames, n_atoms, 3)
lattices = np.broadcast_to(np.eye(3) * 10, (50, 3, 3)).copy()

bond_specs = [
    BondSpec(species=("C", "O"), max_length=1.6),
    BondSpec(species=("O", "H"), max_length=1.2),
]

result = analyse_trajectory(species, coords, bond_specs, lattices)
```

## API

| Function | Description |
|---|---|
| `analyse_trajectory(species, coords, bond_specs, lattices)` | Main entry point. Takes raw arrays, auto-selects backend. |
| `analyse_atoms(atoms, bond_specs)` | Convenience wrapper for ASE `Atoms`, `list[Atoms]`, or `Trajectory`. |
| `analyse_structures(structures, bond_specs)` | Convenience wrapper for pymatgen `Structure` or `list[Structure]`. |
| `find_bonds(species, coords, bond_specs, lattice)` | Single-frame bond detection. Returns sparse adjacency matrix. |
| `find_clusters(adjacency)` | Connected components on an adjacency matrix. |
| `cluster_composition(species, labels)` | Cluster formula counts for a single frame. |

| Type | Description |
|---|---|
| `BondSpec(species, max_length, min_length=0.0)` | Bond detection rule. Species pair supports wildcards (`"*"`). |
| `TrajectoryResult` | Result container with `.n_clusters`, `.labels`, and `.composition(frame)`. |

## Bond specs

A `BondSpec` defines which atom pairs can be bonded and at what distance.
Species names support fnmatch wildcards:

```python
BondSpec(species=("*", "*"), max_length=2.0)       # any pair within 2.0 A
BondSpec(species=("C", "O"), max_length=1.6)        # C-O bonds up to 1.6 A
BondSpec(species=("O", "H"), max_length=1.2, min_length=0.5)  # with min cutoff
```

The species pair is order-invariant: `("C", "O")` and `("O", "C")` are equivalent.
When multiple specs match the same pair, first-match-wins.
