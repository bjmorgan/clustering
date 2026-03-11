"""Tests for connected-component clustering."""

import numpy as np
from scipy import sparse

from md_clusters import find_clusters


class TestFindClusters:
    """Test connected component identification."""

    def test_two_clusters(self):
        """5 atoms: (0,1,2) bonded together, (3,4) bonded together."""
        row = [0, 1, 0, 2, 3, 4]
        col = [1, 0, 2, 0, 4, 3]
        data = [True] * 6
        adj = sparse.csr_matrix((data, (row, col)), shape=(5, 5))

        n_clusters, labels = find_clusters(adj)
        assert n_clusters == 2
        # Atoms 0,1,2 share a label; atoms 3,4 share a different one.
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4]
        assert labels[0] != labels[3]

    def test_all_isolated(self):
        """No bonds -> each atom is its own cluster."""
        adj = sparse.csr_matrix((4, 4), dtype=bool)

        n_clusters, labels = find_clusters(adj)
        assert n_clusters == 4
        assert len(set(labels)) == 4

    def test_single_cluster(self):
        """All atoms bonded into one cluster."""
        # Chain: 0-1-2-3
        row = [0, 1, 1, 2, 2, 3]
        col = [1, 0, 2, 1, 3, 2]
        data = [True] * 6
        adj = sparse.csr_matrix((data, (row, col)), shape=(4, 4))

        n_clusters, labels = find_clusters(adj)
        assert n_clusters == 1
        assert all(labels == 0)

    def test_labels_canonicalised(self):
        """Label 0 contains atom 0, label 1 the next unclaimed, etc."""
        # Two clusters: {0, 2} and {1, 3}.
        row = [0, 2, 1, 3]
        col = [2, 0, 3, 1]
        data = [True] * 4
        adj = sparse.csr_matrix((data, (row, col)), shape=(4, 4))

        n_clusters, labels = find_clusters(adj)
        assert n_clusters == 2
        assert labels[0] == 0  # Cluster 0 contains atom 0.
        assert labels[1] == 1  # Cluster 1 contains atom 1.
        assert labels[2] == 0
        assert labels[3] == 1
