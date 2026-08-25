"""Degree-weighted path count (DWPC), following Himmelstein et al. 2017/2023.

For a metapath of metaedges (M1, M2, ..., Mk), each adjacency matrix is
degree-weighted before multiplication:

    M_hat_i = D_out_i^(-w) @ M_i @ D_in_i^(-w)

where D_out_i is the diagonal matrix of row sums of M_i (out-degree along
that specific metaedge) and D_in_i the diagonal of column sums (in-degree),
and w is the damping exponent (Himmelstein et al. use w=0.4 as their
default/calibrated value - down-weights paths through high-degree "hub"
nodes, which otherwise dominate raw path counts).

DWPC(path) is then the (source, target) entry of the matrix product
M_hat_1 @ M_hat_2 @ ... @ M_hat_k.

Caveat vs. the published method: this computes counts via plain matrix
multiplication, which permits paths that revisit a node (a walk, not a
strict simple path). Himmelstein's hetmatpy additionally subtracts
duplicated-node walks for short paths. For the path lengths used in this
project (<=4 hops) node revisits are rare given the network's low density,
but this is a documented approximation, not the exact algorithm - noted
again in HETNET_LOGIC.md / the final report.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

DEFAULT_DAMPING = 0.4


def degree_weight(mat: sparse.csr_matrix, w: float = DEFAULT_DAMPING) -> sparse.csr_matrix:
    """Return D_out^-w @ mat @ D_in^-w, with 0^-w treated as 0 (isolated nodes stay isolated)."""
    mat = mat.tocsr().astype(np.float64)
    out_deg = np.asarray(mat.sum(axis=1)).flatten()
    in_deg = np.asarray(mat.sum(axis=0)).flatten()

    with np.errstate(divide="ignore"):
        out_w = np.where(out_deg > 0, out_deg ** (-w), 0.0)
        in_w = np.where(in_deg > 0, in_deg ** (-w), 0.0)

    d_out = sparse.diags(out_w)
    d_in = sparse.diags(in_w)
    return d_out @ mat @ d_in


def compute_dwpc(matrices: list[sparse.csr_matrix], w: float = DEFAULT_DAMPING) -> sparse.csr_matrix:
    """DWPC matrix for a metapath given its ordered list of raw adjacency matrices."""
    if not matrices:
        raise ValueError("compute_dwpc requires at least one matrix")
    weighted = [degree_weight(m, w) for m in matrices]
    result = weighted[0]
    for m in weighted[1:]:
        result = result @ m
    return result.tocsr()
