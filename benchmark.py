"""Benchmark: compare find_bonds with and without pre-computed pair masks."""

import time

import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from clustering import BondSpec, build_pair_masks, find_bonds, find_clusters, cluster_composition


def main() -> None:
    # Load trajectory.
    print("Loading trajectory...")
    traj = read("example_traj.extxyz", index=":")
    adaptor = AseAtomsAdaptor()
    structures = [adaptor.get_structure(atoms) for atoms in traj]
    n_frames = len(structures)
    n_atoms = len(structures[0])
    print(f"{n_frames} frames, {n_atoms} atoms/frame")

    # Bond specs — reasonable cutoffs for this system.
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

    species = [str(site.specie) for site in structures[0]]

    # --- Without pre-computed masks ---
    print("\nWithout pre-computed masks:")
    t0 = time.perf_counter()
    results_without: list[tuple] = []
    for structure in structures:
        coords = structure.cart_coords
        lattice = structure.lattice.matrix
        adj = find_bonds(species, coords, bond_specs, lattice)
        n_clusters, labels = find_clusters(adj)
        results_without.append((adj, n_clusters, labels))
    t1 = time.perf_counter()
    print(f"  {t1 - t0:.3f}s ({(t1 - t0) / n_frames * 1000:.1f} ms/frame)")

    # --- With pre-computed masks ---
    print("\nWith pre-computed masks:")
    pair_masks = build_pair_masks(species, bond_specs)
    t0 = time.perf_counter()
    results_with: list[tuple] = []
    for structure in structures:
        coords = structure.cart_coords
        lattice = structure.lattice.matrix
        adj = find_bonds(species, coords, bond_specs, lattice, pair_masks=pair_masks)
        n_clusters, labels = find_clusters(adj)
        results_with.append((adj, n_clusters, labels))
    t1 = time.perf_counter()
    print(f"  {t1 - t0:.3f}s ({(t1 - t0) / n_frames * 1000:.1f} ms/frame)")

    # --- Verify identical results ---
    print("\nVerifying identical results...")
    for i in range(n_frames):
        adj_a, nc_a, lab_a = results_without[i]
        adj_b, nc_b, lab_b = results_with[i]
        assert nc_a == nc_b, f"Frame {i}: n_clusters differ ({nc_a} vs {nc_b})"
        assert np.array_equal(lab_a, lab_b), f"Frame {i}: labels differ"
        assert (adj_a != adj_b).nnz == 0, f"Frame {i}: adjacency matrices differ"
    print("  All frames identical.")

    # --- Show sample composition ---
    print("\nSample composition (frame 0):")
    comp = cluster_composition(species, results_with[0][2])
    for formula, count in sorted(comp.items(), key=lambda x: -x[1]):
        print(f"  {formula}: {count}")


if __name__ == "__main__":
    main()
