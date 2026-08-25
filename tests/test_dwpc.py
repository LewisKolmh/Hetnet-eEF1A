import numpy as np
from scipy import sparse

from src.dwpc import compute_dwpc, degree_weight


def test_degree_weight_toy():
    # A: node0 -> node1, node0 -> node2 (out-degree(0)=2), node3 -> node1 (in-degree(1)=2)
    A = sparse.csr_matrix(np.array([
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float))
    w = 0.5
    weighted = degree_weight(A, w=w)
    # entry (0,1): out_deg[0]=2 -> 2^-0.5 ; in_deg[1]=2 -> 2^-0.5
    expected_01 = (2 ** -w) * 1.0 * (2 ** -w)
    assert np.isclose(weighted[0, 1], expected_01)
    # row 1 (no outgoing edges) must stay all zero
    assert weighted[1].nnz == 0


def test_degree_weight_isolated_node_no_div_by_zero():
    A = sparse.csr_matrix(np.zeros((3, 3)))
    w = 0.4
    weighted = degree_weight(A, w=w)
    assert weighted.nnz == 0
    assert not np.isnan(weighted.toarray()).any()


def test_compute_dwpc_two_hop_toy():
    # C -b-> G1 -i-> G2 : one compound binds gene1, gene1 interacts gene2
    # 2 compounds (0,1), 2 genes (0,1) reused as separate matrices of same shape for simplicity (3 nodes total: C0,C1,G0,G1)
    n = 4  # 0,1 = compounds; 2,3 = genes
    CbG = sparse.lil_matrix((n, n))
    CbG[0, 2] = 1  # compound0 binds gene0
    CbG[1, 2] = 1  # compound1 binds gene0
    CbG = CbG.tocsr()

    GiG = sparse.lil_matrix((n, n))
    GiG[2, 3] = 1
    GiG[3, 2] = 1  # symmetric interaction gene0<->gene1
    GiG = GiG.tocsr()

    dwpc = compute_dwpc([CbG, GiG], w=0.4)
    # path compound0 -b-> gene0 -i-> gene1 should have nonzero DWPC at (0,3)
    assert dwpc[0, 3] > 0
    # compound0 has no path to gene0 itself via this 2-hop metapath (no self-loop)
    assert dwpc[0, 2] == 0
    # symmetric: compound0 and compound1 both bind gene0 with out-degree 1 each,
    # so their DWPC to gene1 should be identical
    assert np.isclose(dwpc[0, 3], dwpc[1, 3])


def test_compute_dwpc_requires_matrices():
    import pytest
    with pytest.raises(ValueError):
        compute_dwpc([])
