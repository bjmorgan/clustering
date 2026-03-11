"""Connected-component clustering on bond adjacency matrices."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


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
