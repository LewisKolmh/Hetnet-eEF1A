#!/usr/bin/env python
"""Generate XSwap-permuted networks and compute the null DWPC distribution
for every metapath, grouped by (source_degree, target_degree) as in
Himmelstein et al. - the piece of this pipeline heavy enough to warrant
an HPC / batch-scheduler run.

Resumable/checkpointed: each permutation's per-metapath null DWPC summary
is written to data/processed/null_distribution/perm_<i>.<scope>.parquet
as soon as it's computed. On restart, already-completed permutation
indices are skipped, so an interrupted job (walltime limit, preemption)
resumes exactly where it left off - just resubmit the same command.

Usage (single node, e.g. inside a SLURM sbatch script):
    python src/compute_null_distribution.py --scope full-interactome \
        --n-permutations 200 --swap-factor 10 --seed 0

Runtime for this project's small network (<=726 nodes, <1000 edges per
metaedge): ~0.05-0.15s per permutation single-threaded, so a 200-permutation
run finishes in well under a minute - this step does NOT need the HPC at
this network size. If the network is later expanded to a much larger
interactome/ChEMBL pull where this becomes minutes per permutation, use
--start-index/--end-index to split the range across parallel array-job
tasks writing to the same output directory - see run_on_hpc.slurm for a
template that still applies at that point.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from dwpc import compute_dwpc
from xswap import xswap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NODES_DIR = Path("data/processed/nodes")
MATRICES_DIR = Path("data/processed/matrices")
NULL_DIR = Path("data/processed/null_distribution")

TARGET_GENES = ["EEF1A1", "EEF1A2"]

METAPATHS = {
    "CbG": [("CbG", False)],
    "CbGiG": [("CbG", False), ("GiG", False)],
    "CbGpBP": [("CbG", False), ("GpBP", False), ("GpBP", True)],
    "CbGpMF": [("CbG", False), ("GpMF", False), ("GpMF", True)],
    "CbGpCC": [("CbG", False), ("GpCC", False), ("GpCC", True)],
    "CbGpPW": [("CbG", False), ("GpPW", False), ("GpPW", True)],
}

# which raw metaedges are symmetric (must be XSwapped as undirected)
SYMMETRIC_METAEDGES = {"GiG"}


def load_matrix(stub: str, scope: str) -> sparse.csr_matrix:
    return sparse.load_npz(MATRICES_DIR / f"{stub}.{scope}.npz")


def permute_all_metaedges(
    base_matrices: dict[str, sparse.csr_matrix], swap_factor: int, rng: np.random.Generator
) -> dict[str, sparse.csr_matrix]:
    permuted = {}
    for stub, mat in base_matrices.items():
        symmetric = stub in SYMMETRIC_METAEDGES
        n_edges = mat.nnz // (2 if symmetric else 1)
        permuted[stub] = xswap(mat, n_swaps=swap_factor * n_edges, symmetric=symmetric, rng=rng)
    return permuted


def dwpc_column_for_targets(
    permuted: dict[str, sparse.csr_matrix], edge_spec: list[tuple[str, bool]],
    target_node_ids: dict[str, int], damping: float,
) -> dict[str, np.ndarray]:
    mats = [(permuted[stub].T.tocsr() if transpose else permuted[stub]) for stub, transpose in edge_spec]
    dwpc_mat = compute_dwpc(mats, w=damping)
    return {g: dwpc_mat[:, tid].toarray().flatten() for g, tid in target_node_ids.items()}


def run_one_permutation(
    idx: int, scope: str, base_matrices: dict, target_node_ids: dict[str, int],
    compound_ids: np.ndarray, gene_out_degree: dict[str, np.ndarray],
    swap_factor: int, damping: float, seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + idx)
    permuted = permute_all_metaedges(base_matrices, swap_factor, rng)

    rows = []
    for metapath, edge_spec in METAPATHS.items():
        cols = dwpc_column_for_targets(permuted, edge_spec, target_node_ids, damping)
        for g, col in cols.items():
            nz = np.nonzero(col)[0]
            for node_idx in nz:
                if node_idx >= len(compound_ids) or compound_ids[node_idx] is None:
                    continue
                rows.append(
                    {
                        "permutation": idx,
                        "metapath": metapath,
                        "target_gene": g,
                        "compound_node_id": int(node_idx),
                        "dwpc": float(col[node_idx]),
                    }
                )
    return pd.DataFrame(rows)


def main(
    scope: str, n_permutations: int, start_index: int, end_index: int | None,
    swap_factor: int, damping: float, seed: int,
) -> None:
    NULL_DIR.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(NODES_DIR / f"nodes.{scope}.tsv", sep="\t")
    gene_map = pd.read_csv(NODES_DIR / f"gene_symbol_to_external_id.{scope}.tsv", sep="\t")
    symbol_to_ext = dict(zip(gene_map["protein_id"], gene_map["external_id"]))
    id_by_ext = {(row.metanode_type, row.external_id): row.node_id for row in nodes.itertuples(index=False)}

    target_node_ids = {}
    for g in TARGET_GENES:
        ext = symbol_to_ext.get(g)
        nid = id_by_ext.get(("Gene", ext)) if ext else None
        if nid is not None:
            target_node_ids[g] = nid

    n_nodes = len(nodes)
    node_to_compound = {row.node_id: row.external_id for row in nodes[nodes.metanode_type == "Compound"].itertuples(index=False)}
    compound_ids = np.array([node_to_compound.get(i) for i in range(n_nodes)], dtype=object)

    base_stubs = sorted({stub for spec in METAPATHS.values() for stub, _ in spec})
    base_matrices = {stub: load_matrix(stub, scope) for stub in base_stubs}

    end_index = end_index if end_index is not None else n_permutations
    log.info(
        "Running permutations [%d, %d) of %d total for scope=%s (swap_factor=%d, damping=%.2f, seed=%d)",
        start_index, end_index, n_permutations, scope, swap_factor, damping, seed,
    )

    for idx in range(start_index, end_index):
        out_path = NULL_DIR / f"perm_{idx:04d}.{scope}.parquet"
        if out_path.exists():
            log.info("[%d] already computed, skipping (%s)", idx, out_path)
            continue
        t0 = time.time()
        df = run_one_permutation(idx, scope, base_matrices, target_node_ids, compound_ids, {}, swap_factor, damping, seed)
        df.to_parquet(out_path, index=False)
        log.info("[%d] done in %.1fs, %d nonzero rows -> %s", idx, time.time() - t0, len(df), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    parser.add_argument("--n-permutations", type=int, default=200, help="Total permutations in the full run (Himmelstein et al. use 200)")
    parser.add_argument("--start-index", type=int, default=0, help="First permutation index this job/task computes (for array jobs)")
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive end index this job/task computes; defaults to n_permutations")
    parser.add_argument("--swap-factor", type=int, default=10, help="XSwap attempts = swap_factor x edge_count per metaedge")
    parser.add_argument("--damping", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.scope, args.n_permutations, args.start_index, args.end_index, args.swap_factor, args.damping, args.seed)
