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

Repeated metanodes
------------------
Plain matrix multiplication counts *walks*, which may revisit a node; DWPC is
defined over *simple paths*, which may not. This matters here far more than a
density argument suggests, because several metapaths in this project are
deliberately constructed with a repeated metanode:

    Compound -binds-> Gene -participates-> Pathway -participates(rev)-> Gene

The final Gene may be the same node as the first. That is not a rare accident
at low density - it is guaranteed for every gene annotated to the intermediate
term, so the walk count contains a term

    DWPC_hat[c, t] += CbG_hat[c, t] * (GpX_hat @ GpX_hat.T)[t, t]

which depends only on the target's own annotation degree and carries no
information about the compound beyond the direct binding edge it already has.
Measured on this project's committed matrices, that term is 100% of the score
for every compound that binds its own reported target gene.

`compute_dwpc` therefore removes the invalid contribution exactly. For a
metapath whose metanode sequence repeats one metanode at positions i and j,

    valid = P @ S @ Q - P @ diag(diag(S)) @ Q

where P is the degree-weighted product of the metaedges before position i, S
the product spanning i..j, and Q the product after j. The subtracted term is
precisely the sum over paths whose node at position i equals its node at
position j, so this is exact rather than an approximation.

Patterns beyond a single repeated pair (a metanode occurring three or more
times, or two independent repeats) need inclusion-exclusion; rather than
silently approximate them, `compute_dwpc` raises `UncorrectedRepeatError`.

`compute_dwpc_walk` retains the original uncorrected behaviour under a name
that says what it computes. It is kept only so this branch can reproduce the
published numbers for the before/after comparison; it should not be used to
score compounds.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

DEFAULT_DAMPING = 0.4


class UncorrectedRepeatError(NotImplementedError):
    """Raised for repeated-metanode patterns this module cannot correct exactly."""


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


def _product(mats: list[sparse.csr_matrix], n: int) -> sparse.csr_matrix:
    """Left-to-right product; the identity when `mats` is empty."""
    if not mats:
        return sparse.identity(n, format="csr", dtype=np.float64)
    result = mats[0]
    for m in mats[1:]:
        result = result @ m
    return result.tocsr()


def compute_dwpc_walk(matrices: list[sparse.csr_matrix], w: float = DEFAULT_DAMPING) -> sparse.csr_matrix:
    """Degree-weighted WALK count: plain matrix product, node revisits permitted.

    This is the original (uncorrected) behaviour of this module, retained only to
    reproduce previously published numbers. Use `compute_dwpc` to score compounds.
    """
    if not matrices:
        raise ValueError("compute_dwpc_walk requires at least one matrix")
    return _product([degree_weight(m, w) for m in matrices], matrices[0].shape[0])


def _repeat_positions(metanodes: list[str]) -> tuple[int, int] | None:
    """Return the (first, second) positions of the single repeated metanode, or None.

    Raises UncorrectedRepeatError if the repeat structure needs inclusion-exclusion.
    """
    counts = Counter(metanodes)
    repeated = [m for m, c in counts.items() if c > 1]
    if not repeated:
        return None
    if len(repeated) > 1 or counts[repeated[0]] > 2:
        raise UncorrectedRepeatError(
            f"metanode sequence {metanodes} repeats {repeated} and needs inclusion-exclusion; "
            "this module only corrects a single repeated pair exactly"
        )
    m = repeated[0]
    return metanodes.index(m), len(metanodes) - 1 - metanodes[::-1].index(m)


def compute_dwpc(
    matrices: list[sparse.csr_matrix],
    metanodes: list[str],
    w: float = DEFAULT_DAMPING,
) -> sparse.csr_matrix:
    """Simple-path DWPC for a metapath, with repeated metanodes corrected exactly.

    Parameters
    ----------
    matrices
        Ordered raw adjacency matrices for the metapath's metaedges.
    metanodes
        The metanode type visited at each position, length ``len(matrices) + 1``
        (e.g. ``["Compound", "Gene", "Pathway", "Gene"]`` for CbGpPWpG). Required:
        without it the repeat structure is unknown and the result would silently
        be a walk count. See module docstring.
    """
    if not matrices:
        raise ValueError("compute_dwpc requires at least one matrix")
    if len(metanodes) != len(matrices) + 1:
        raise ValueError(
            f"metanodes must have length len(matrices)+1 = {len(matrices) + 1}, got {len(metanodes)}"
        )

    n = matrices[0].shape[0]
    weighted = [degree_weight(m, w) for m in matrices]
    full = _product(weighted, n)

    repeat = _repeat_positions(list(metanodes))
    if repeat is None:
        return full.tocsr()

    i, j = repeat
    prefix = _product(weighted[:i], n)
    segment = _product(weighted[i:j], n)
    suffix = _product(weighted[j:], n)

    invalid = prefix @ sparse.diags(segment.diagonal()) @ suffix
    corrected = (full - invalid).tocsr()

    # The subtracted term is a subset of the walk count, so the result cannot be
    # negative except by floating-point cancellation. A materially negative entry
    # would mean the repeat structure was misidentified, so fail loudly.
    if corrected.nnz:
        most_negative = corrected.data.min()
        if most_negative < -1e-9:
            raise AssertionError(
                f"corrected DWPC has a negative entry ({most_negative:.3e}) for metanodes "
                f"{metanodes}; the repeat correction is wrong, not merely imprecise"
            )
        corrected.data[np.abs(corrected.data) < 1e-12] = 0.0
        corrected.eliminate_zeros()
    return corrected
