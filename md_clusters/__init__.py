"""Atomic cluster analysis for MD trajectories.

Identifies bonded atom pairs per frame using the minimum image convention
(MIC) and groups them into clusters of contiguously bonded atoms via
connected-component analysis.
"""

from md_clusters.bonds import find_bonds
from md_clusters.clusters import find_clusters
from md_clusters.composition import _hill_formula, cluster_composition
from md_clusters.trajectory import (
    TrajectoryResult,
    analyse_atoms,
    analyse_structures,
    analyse_trajectory,
)
from md_clusters.types import BondSpec

__all__ = [
    "BondSpec",
    "TrajectoryResult",
    "analyse_atoms",
    "analyse_structures",
    "analyse_trajectory",
    "cluster_composition",
    "find_bonds",
    "find_clusters",
]
