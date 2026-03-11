"""Benchmark: compare numpy and numba backends for trajectory analysis."""

import time

import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from clustering import (
    BondSpec, analyse_trajectory, analyse_structures, analyse_atoms,
    find_bonds, find_clusters, cluster_composition,
)


def main() -> None:
    # Load trajectory.
    print("Loading trajectory...")
    traj = read("example_traj.extxyz", index=":")
    adaptor = AseAtomsAdaptor()
    structures = [adaptor.get_structure(atoms) for atoms in traj]
    n_frames = len(structures)
    n_atoms = len(structures[0])
    print(f"{n_frames} frames, {n_atoms} atoms/frame")

    # Bond specs.
    bond_specs = [
        BondSpec(species=("C", "O"), max_length=1.6),
        BondSpec(species=("C", "H"), max_length=1.2),
        BondSpec(species=("O", "H"), max_length=1.2),
        BondSpec(species=("C", "C"), max_length=1.8),
        BondSpec(species=("C", "F"), max_length=1.5),
        BondSpec(species=("P", "F"), max_length=1.8),
        BondSpec(species=("P", "O"), max_length=1.7),
        BondSpec(species=("Li", "O"), max_length=2.2),
        BondSpec(species=("Li", "F"), max_length=2.1),
        BondSpec(species=("Ni", "O"), max_length=2.2),
    ]

    # Extract raw arrays.
    species = [site.specie.symbol for site in structures[0]]
    all_coords = np.array([s.cart_coords for s in structures])
    all_lattices = np.array([s.lattice.matrix for s in structures])

    # --- Numpy: find_bonds + find_clusters per frame ---
    print("\nNumpy (find_bonds + find_clusters per frame):")
    t0 = time.perf_counter()
    ref_labels = []
    ref_n_clusters = []
    for f in range(n_frames):
        adj = find_bonds(species, all_coords[f], bond_specs, all_lattices[f])
        nc, lab = find_clusters(adj)
        ref_labels.append(lab)
        ref_n_clusters.append(nc)
    t1 = time.perf_counter()
    print(f"  {t1 - t0:.3f}s ({(t1 - t0) / n_frames * 1000:.1f} ms/frame)")

    # --- analyse_trajectory (auto-selects numba) ---
    # Warmup JIT.
    print("\nanalyse_trajectory (warming up JIT)...")
    analyse_trajectory(species, all_coords[:1], bond_specs, all_lattices[:1])
    print("  JIT compiled.")

    print("\nanalyse_trajectory (numba batch + union-find):")
    t0 = time.perf_counter()
    result = analyse_trajectory(species, all_coords, bond_specs, all_lattices)
    t1 = time.perf_counter()
    t_at = t1 - t0
    print(f"  {t_at:.3f}s ({t_at / n_frames * 1000:.1f} ms/frame)")

    # Verify results match.
    print("\nVerifying results match numpy reference...")
    for f in range(n_frames):
        assert result.n_clusters[f] == ref_n_clusters[f], (
            f"Frame {f}: n_clusters differ"
        )
        assert np.array_equal(result.labels[f], ref_labels[f]), (
            f"Frame {f}: labels differ"
        )
    print("  All frames identical.")

    # --- Convenience wrappers ---
    print("\nanalyse_structures (pymatgen):")
    analyse_structures(structures[:1], bond_specs)  # Warmup.
    t0 = time.perf_counter()
    result_pmg = analyse_structures(structures, bond_specs)
    t1 = time.perf_counter()
    print(f"  {t1 - t0:.3f}s ({(t1 - t0) / n_frames * 1000:.1f} ms/frame)")

    print("\nanalyse_atoms (ASE):")
    analyse_atoms(traj[:1], bond_specs)  # Warmup.
    t0 = time.perf_counter()
    result_ase = analyse_atoms(traj, bond_specs)
    t1 = time.perf_counter()
    print(f"  {t1 - t0:.3f}s ({(t1 - t0) / n_frames * 1000:.1f} ms/frame)")

    # --- Show sample composition ---
    print("\nSample composition (frame 0):")
    comp = result.composition(0)
    for formula, count in comp.most_common():
        print(f"  {formula}: {count}")


if __name__ == "__main__":
    main()
