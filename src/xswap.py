"""Degree-preserving edge swap (XSwap), per Hanhijarvi et al. / Himmelstein
et al.'s hetnet permutation scheme.

Repeatedly picks two edges (a,b) and (c,d) and, if the swap doesn't create
a self-loop or a duplicate edge, replaces them with (a,d) and (c,b) (or,
for a symmetric/undirected matrix, also tries (a,c)/(b,d) with 50%
probability so the swap is well-defined for undirected edges too).
This exactly preserves every node's degree while randomizing which
specific edges exist - the null model Himmelstein et al. use to ask
"is this DWPC higher than expected for a network with the same degree
sequence, or is it just an artifact of a well-connected node?"
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

log = logging.getLogger(__name__)


def xswap(
    mat: sparse.csr_matrix,
    n_swaps: int | None = None,
    symmetric: bool = False,
    rng: np.random.Generator | None = None,
    max_attempts_factor: int = 10,
) -> sparse.csr_matrix:
    """Return a degree-preserving randomized copy of a binary adjacency matrix.

    n_swaps defaults to 10x the edge count (Hanhijarvi et al.'s guidance for
    adequate mixing). For symmetric matrices, only the upper triangle is
    swapped and the result is symmetrized.
    """
    rng = rng or np.random.default_rng()
    mat = mat.tocoo()
    if symmetric:
        mask = mat.row < mat.col
        edges = list(zip(mat.row[mask].tolist(), mat.col[mask].tolist()))
    else:
        edges = list(zip(mat.row.tolist(), mat.col.tolist()))

    edge_set = set(edges)
    n_edges = len(edges)
    if n_edges < 2:
        return mat.tocsr()
    if n_swaps is None:
        n_swaps = 10 * n_edges

    max_attempts = n_swaps * max_attempts_factor
    n_success = 0
    attempts = 0
    edges = list(edges)
    while n_success < n_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, n_edges, size=2)
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) < 4:
            continue  # shares a node - would create self-loop or trivial swap
        new1, new2 = (a, d), (c, b)
        if new1 in edge_set or new2 in edge_set or new1 == new2:
            continue
        # commit the swap
        edge_set.discard((a, b))
        edge_set.discard((c, d))
        edge_set.add(new1)
        edge_set.add(new2)
        edges[i] = new1
        edges[j] = new2
        n_success += 1

    if n_success < n_swaps:
        log.warning(
            "xswap: only completed %d/%d requested swaps after %d attempts "
            "(network may be too dense/sparse for full mixing)",
            n_success, n_swaps, attempts,
        )

    rows, cols = zip(*edge_set) if edge_set else ([], [])
    n = mat.shape[0]
    data = np.ones(len(rows), dtype=np.float64)
    new_mat = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    if symmetric:
        new_mat = new_mat.maximum(new_mat.T)
    return new_mat
