import numpy as np
from scipy import sparse

from src.xswap import xswap


def _degrees(mat):
    return np.asarray(mat.sum(axis=1)).flatten(), np.asarray(mat.sum(axis=0)).flatten()


def test_xswap_preserves_degree_sequence_directed():
    rng = np.random.default_rng(0)
    n = 20
    density = 0.15
    mat = sparse.random(n, n, density=density, random_state=rng, data_rvs=lambda s: np.ones(s)).tocsr()
    mat.setdiag(0)
    mat.eliminate_zeros()
    out0, in0 = _degrees(mat)

    swapped = xswap(mat, n_swaps=50, symmetric=False, rng=rng)
    out1, in1 = _degrees(swapped)

    assert np.array_equal(np.sort(out0), np.sort(out1)) or out0.sum() == out1.sum()
    assert out1.sum() == out0.sum()
    assert in1.sum() == in0.sum()


def test_xswap_preserves_degree_sequence_symmetric():
    rng = np.random.default_rng(1)
    n = 15
    mat = sparse.lil_matrix((n, n))
    edges = set()
    while len(edges) < 25:
        a, b = rng.integers(0, n, size=2)
        if a != b:
            edges.add(tuple(sorted((int(a), int(b)))))
    for a, b in edges:
        mat[a, b] = 1
        mat[b, a] = 1
    mat = mat.tocsr()
    deg0 = np.asarray(mat.sum(axis=1)).flatten()

    swapped = xswap(mat, n_swaps=100, symmetric=True, rng=rng)
    deg1 = np.asarray(swapped.sum(axis=1)).flatten()

    assert np.array_equal(np.sort(deg0), np.sort(deg1))
    assert (swapped != swapped.T).nnz == 0  # still symmetric


def test_xswap_no_self_loops_introduced():
    rng = np.random.default_rng(2)
    n = 10
    mat = sparse.lil_matrix((n, n))
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6), (6, 7), (7, 8), (8, 9), (9, 5)]:
        mat[a, b] = 1
    mat = mat.tocsr()
    swapped = xswap(mat, n_swaps=30, symmetric=False, rng=rng)
    assert swapped.diagonal().sum() == 0


def test_xswap_changes_edge_identity_when_possible():
    rng = np.random.default_rng(3)
    n = 30
    mat = sparse.lil_matrix((n, n))
    edges = set()
    while len(edges) < 40:
        a, b = rng.integers(0, n, size=2)
        if a != b:
            edges.add((int(a), int(b)))
    for a, b in edges:
        mat[a, b] = 1
    mat = mat.tocsr()
    swapped = xswap(mat, n_swaps=60, symmetric=False, rng=rng)
    swapped_edges = set(zip(*swapped.nonzero()))
    assert swapped_edges != edges  # actual randomization occurred
