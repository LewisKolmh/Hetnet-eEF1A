"""Tests for the repeated-metanode correction in src.dwpc.

The oracle here is explicit enumeration of simple paths on a small block-structured
toy graph, rather than a second matrix identity: enumeration is what DWPC is defined
over, so if the correction and the enumeration agree there is nothing left to assume.

`test_self_binding_compound_scores_zero` is the regression test for the defect these
tests exist for - it passes on the corrected code and fails on the uncorrected walk
count, which is asserted explicitly in the same test.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from src.dwpc import (
    DEFAULT_DAMPING,
    UncorrectedRepeatError,
    compute_dwpc,
    compute_dwpc_walk,
    degree_weight,
)

# Block-structured index space, so the only coincidence a walk can have on the
# Compound-Gene-Annotation-Gene metapath is (first gene == last gene) - exactly
# what the correction removes.
N_NODES = 28
COMPOUNDS = range(0, 10)
GENES = range(10, 20)
ANNOTATIONS = range(20, 28)


def _toy_matrices(seed: int = 0) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    rng = np.random.default_rng(seed)
    cbg = sparse.lil_matrix((N_NODES, N_NODES), dtype=np.float64)
    gpx = sparse.lil_matrix((N_NODES, N_NODES), dtype=np.float64)
    for c in COMPOUNDS:
        for g in rng.choice(list(GENES), size=rng.integers(1, 4), replace=False):
            cbg[c, g] = 1.0
    for g in GENES:
        for x in rng.choice(list(ANNOTATIONS), size=rng.integers(1, 4), replace=False):
            gpx[g, x] = 1.0
    return cbg.tocsr(), gpx.tocsr()


def _brute_force_cgxg(cbg: sparse.csr_matrix, gpx: sparse.csr_matrix, w: float) -> np.ndarray:
    """Enumerate degree-weighted SIMPLE paths Compound -> Gene -> Annotation -> Gene."""
    a = degree_weight(cbg, w).toarray()
    b = degree_weight(gpx, w).toarray()
    out = np.zeros((N_NODES, N_NODES))
    for c, t in itertools.product(COMPOUNDS, GENES):
        total = 0.0
        for g in GENES:
            if g == t:  # a simple path may not revisit the gene it started from
                continue
            for x in ANNOTATIONS:
                total += a[c, g] * b[g, x] * b[t, x]
        out[c, t] = total
    return out


def test_repeat_correction_matches_path_enumeration():
    cbg, gpx = _toy_matrices(seed=1)
    metanodes = ["Compound", "Gene", "Annotation", "Gene"]
    got = compute_dwpc([cbg, gpx, gpx.T.tocsr()], metanodes, w=DEFAULT_DAMPING).toarray()
    expected = _brute_force_cgxg(cbg, gpx, DEFAULT_DAMPING)
    np.testing.assert_allclose(got[np.ix_(list(COMPOUNDS), list(GENES))],
                               expected[np.ix_(list(COMPOUNDS), list(GENES))],
                               rtol=0, atol=1e-12)


def test_walk_count_is_an_upper_bound_and_is_not_equal():
    cbg, gpx = _toy_matrices(seed=2)
    mats = [cbg, gpx, gpx.T.tocsr()]
    corrected = compute_dwpc(mats, ["Compound", "Gene", "Annotation", "Gene"]).toarray()
    walk = compute_dwpc_walk(mats).toarray()
    assert (walk >= corrected - 1e-12).all(), "walk count must dominate the simple-path count"
    assert not np.allclose(walk, corrected), "toy graph should exhibit the defect being corrected"


def test_self_binding_compound_scores_zero():
    """A compound whose only binding edge is to the reported target must score 0.

    Its only walk to the target is Compound -> target -> annotation -> target, which
    revisits the target. This is the exact situation that produced 13 of the 22
    Benjamini-Hochberg hits in the published run.
    """
    c, t, other_gene, ann = 0, 10, 11, 20
    cbg = sparse.lil_matrix((N_NODES, N_NODES), dtype=np.float64)
    gpx = sparse.lil_matrix((N_NODES, N_NODES), dtype=np.float64)
    cbg[c, t] = 1.0
    cbg[1, other_gene] = 1.0  # a second compound, so degrees are not degenerate
    gpx[t, ann] = 1.0
    gpx[other_gene, ann] = 1.0
    mats = [cbg.tocsr(), gpx.tocsr(), gpx.tocsr().T.tocsr()]
    metanodes = ["Compound", "Gene", "Annotation", "Gene"]

    corrected = compute_dwpc(mats, metanodes)
    walk = compute_dwpc_walk(mats)
    assert corrected[c, t] == 0.0, "self-return walk must not contribute to the score"
    assert walk[c, t] > 0.0, "the uncorrected walk count scores this compound - the defect"
    # the other compound reaches t by a genuine simple path and must survive
    assert corrected[1, t] > 0.0


def test_two_hop_self_loop_repeat_is_corrected():
    """Compound -> Gene -> Gene: only a GiG self-loop can make the walk invalid."""
    cbg = sparse.lil_matrix((N_NODES, N_NODES), dtype=np.float64)
    gig = sparse.lil_matrix((N_NODES, N_NODES), dtype=np.float64)
    cbg[0, 10] = 1.0
    gig[10, 10] = 1.0  # self-loop
    gig[10, 11] = gig[11, 10] = 1.0
    mats = [cbg.tocsr(), gig.tocsr()]
    corrected = compute_dwpc(mats, ["Compound", "Gene", "Gene"])
    walk = compute_dwpc_walk(mats)
    assert corrected[0, 10] == 0.0
    assert walk[0, 10] > 0.0
    assert corrected[0, 11] == pytest.approx(walk[0, 11])


def test_no_repeat_leaves_result_unchanged():
    cbg, gpx = _toy_matrices(seed=3)
    mats = [cbg, gpx]
    corrected = compute_dwpc(mats, ["Compound", "Gene", "Annotation"]).toarray()
    walk = compute_dwpc_walk(mats).toarray()
    np.testing.assert_allclose(corrected, walk, rtol=0, atol=1e-15)


def test_unsupported_repeat_pattern_raises_rather_than_approximating():
    cbg, gpx = _toy_matrices(seed=4)
    mats = [cbg, gpx, gpx.T.tocsr(), gpx, gpx.T.tocsr()]
    with pytest.raises(UncorrectedRepeatError):
        compute_dwpc(mats, ["Compound", "Gene", "Ann", "Gene", "Ann", "Gene"])


def test_metanodes_length_is_validated():
    cbg, gpx = _toy_matrices(seed=5)
    with pytest.raises(ValueError):
        compute_dwpc([cbg, gpx], ["Compound", "Gene"])


@pytest.mark.parametrize("metapath,stubs", [
    ("CbGpPW", ["CbG", "GpPW"]),
    ("CbGpCC", ["CbG", "GpCC"]),
])
def test_committed_matrices_self_binders_lose_their_score(metapath, stubs):
    """Integration check against the project's own committed matrices."""
    mdir = Path("data/processed/matrices")
    scope = "full-interactome"
    paths = [mdir / f"{s}.{scope}.npz" for s in stubs]
    if not all(p.exists() for p in paths):
        pytest.skip("committed matrices not present")
    cbg = sparse.load_npz(paths[0])
    gpx = sparse.load_npz(paths[1])
    mats = [cbg, gpx, gpx.T.tocsr()]
    metanodes = ["Compound", "Gene", "Annotation", "Gene"]
    corrected = compute_dwpc(mats, metanodes)
    walk = compute_dwpc_walk(mats)

    # Every compound-target pair where the compound binds that very target must
    # lose score. It falls to exactly zero when the target is the compound's only
    # binding partner in the annotation neighbourhood; a compound that also binds
    # a second annotated gene keeps that genuine simple-path contribution, so the
    # assertion is a strict reduction, with at least one pair vanishing entirely.
    binders = cbg.tocoo()
    n_reduced = n_zeroed = 0
    for c, t in zip(binders.row, binders.col):
        if walk[c, t] <= 0:
            continue
        assert corrected[c, t] < walk[c, t] - 1e-12, (
            f"{metapath}: compound node {c} binds target node {t} but its score was "
            "not reduced by the correction"
        )
        assert corrected[c, t] >= -1e-12
        n_reduced += 1
        n_zeroed += corrected[c, t] == pytest.approx(0.0, abs=1e-12)
    assert n_reduced > 0, "expected self-binding pairs in the committed matrices"
    assert n_zeroed > 0, "expected at least one score to vanish entirely"
