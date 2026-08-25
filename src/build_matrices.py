"""Build scipy.sparse adjacency matrices for every metaedge in the metagraph
that this project's data populates, indexed by the node table's `node_id`.

Metaedges built:
  CbG  Compound--binds--Gene              (from compound_binds_gene.<scope>.tsv)
  GiG  Gene--interacts--Gene              (from gene_interacts_gene.tsv)
  GpBP Gene--participates--BiologicalProcess (from go_annotations.tsv)
  GpMF Gene--performs--MolecularFunction     (from go_annotations.tsv)
  GpCC Gene--localizes--CellularComponent    (from go_annotations.tsv)
  GpPW Gene--participates--Pathway           (from pathway_annotations.tsv)

Each matrix is saved as a .npz (scipy.sparse.save_npz) in
data/processed/matrices/, shape (n_nodes, n_nodes), dtype float64, with 1.0
for a present edge (binary adjacency - matches Himmelstein et al.'s hetnet
convention of unweighted edges for DWPC).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
NODES_DIR = Path("data/processed/nodes")
MATRICES_DIR = Path("data/processed/matrices")


def load_nodes(scope: str) -> tuple[pd.DataFrame, dict, dict]:
    nodes = pd.read_csv(NODES_DIR / f"nodes.{scope}.tsv", sep="\t")
    id_by_ext = {(row.metanode_type, row.external_id): row.node_id for row in nodes.itertuples(index=False)}
    gene_symbol_map = pd.read_csv(NODES_DIR / f"gene_symbol_to_external_id.{scope}.tsv", sep="\t")
    symbol_to_ext = dict(zip(gene_symbol_map["protein_id"], gene_symbol_map["external_id"]))
    return nodes, id_by_ext, symbol_to_ext


def build_matrix(n: int, edges: list[tuple[int, int]], symmetric: bool) -> sparse.csr_matrix:
    if not edges:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    rows, cols = zip(*edges)
    data = np.ones(len(edges), dtype=np.float64)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    if symmetric:
        mat = mat.maximum(mat.T)
    mat.data[:] = 1.0  # collapse any duplicate-sum entries back to binary
    return mat


def main(scope: str) -> None:
    nodes, id_by_ext, symbol_to_ext = load_nodes(scope)
    n = len(nodes)
    log.info("Building matrices for scope=%s, %d nodes", scope, n)
    MATRICES_DIR.mkdir(parents=True, exist_ok=True)

    def gene_id(symbol: str) -> int | None:
        ext = symbol_to_ext.get(symbol)
        return id_by_ext.get(("Gene", ext)) if ext else None

    # CbG: Compound binds Gene
    cbg = pd.read_csv(RAW_DIR / f"compound_binds_gene.{scope}.tsv", sep="\t")
    edges = []
    n_skipped = 0
    for row in cbg.itertuples(index=False):
        cid = id_by_ext.get(("Compound", row.compound_id))
        gid = gene_id(row.protein_id)
        if cid is None or gid is None:
            n_skipped += 1
            continue
        edges.append((cid, gid))
    m_cbg = build_matrix(n, edges, symmetric=False)
    sparse.save_npz(MATRICES_DIR / f"CbG.{scope}.npz", m_cbg)
    log.info("CbG: %d edges (%d skipped - missing node mapping), nnz=%d", len(edges), n_skipped, m_cbg.nnz)

    # GiG: Gene interacts Gene (symmetric, from the STRING network - same for both scopes)
    gig = pd.read_csv(RAW_DIR / "gene_interacts_gene.tsv", sep="\t")
    edges = []
    for row in gig.itertuples(index=False):
        a, b = gene_id(row.gene_a_symbol), gene_id(row.gene_b_symbol)
        if a is not None and b is not None:
            edges.append((a, b))
    m_gig = build_matrix(n, edges, symmetric=True)
    sparse.save_npz(MATRICES_DIR / f"GiG.{scope}.npz", m_gig)
    log.info("GiG: %d edges, nnz=%d", len(edges), m_gig.nnz)

    # Gp{BP,MF,CC}: Gene x GO term, split by aspect
    go = pd.read_csv(RAW_DIR / "go_annotations.tsv", sep="\t")
    aspect_to_metaedge = {
        "BiologicalProcess": "GpBP",
        "MolecularFunction": "GpMF",
        "CellularComponent": "GpCC",
    }
    for aspect, metaedge in aspect_to_metaedge.items():
        sub = go[go["aspect"] == aspect]
        edges = []
        for row in sub.itertuples(index=False):
            gid = gene_id(row.protein_id)
            tid = id_by_ext.get((aspect, row.go_id))
            if gid is not None and tid is not None:
                edges.append((gid, tid))
        mat = build_matrix(n, edges, symmetric=False)
        sparse.save_npz(MATRICES_DIR / f"{metaedge}.{scope}.npz", mat)
        log.info("%s: %d edges, nnz=%d", metaedge, len(edges), mat.nnz)

    # GpPW: Gene x Pathway
    pw = pd.read_csv(RAW_DIR / "pathway_annotations.tsv", sep="\t")
    edges = []
    for row in pw.itertuples(index=False):
        gid = gene_id(row.protein_id)
        tid = id_by_ext.get(("Pathway", row.pathway_id))
        if gid is not None and tid is not None:
            edges.append((gid, tid))
    m_gppw = build_matrix(n, edges, symmetric=False)
    sparse.save_npz(MATRICES_DIR / f"GpPW.{scope}.npz", m_gppw)
    log.info("GpPW: %d edges, nnz=%d", len(edges), m_gppw.nnz)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    args = parser.parse_args()
    main(args.scope)
