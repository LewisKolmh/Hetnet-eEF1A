"""Compute observed DWPC for every compound against EEF1A1 and EEF1A2,
across every metapath defined in METAPATHS below.

Metapaths (abbreviation: sequence of metaedges, matching Himmelstein-style
notation where each metaedge symbol is <SourceAbbrev><edge_abbrev><TargetAbbrev>):
  CbG      Compound -binds-> Gene                                  (direct bind, 1 hop)
  CbGiG    Compound -binds-> Gene -interacts-> Gene                (2 hop, via interactome)
  CbGpBP   Compound -binds-> Gene -participates-> BiologicalProcess -> (reverse) -> Gene
  CbGpMF   ... likewise via MolecularFunction
  CbGpCC   ... likewise via CellularComponent
  CbGpPW   ... likewise via Pathway

For the annotation-mediated paths (BP/MF/CC/PW), the path is
Compound -b-> Gene_x -p-> Annotation -p(reverse)-> Gene_target, i.e. it
measures "compounds that bind a gene sharing a GO term / pathway with
EEF1A1/EEF1A2" - this is the paper's approach of scoring compounds by
their network context around the target, not just direct binding.

Output: data/processed/dwpc/dwpc_observed.<scope>.tsv with columns
  compound_id, target_gene (EEF1A1/EEF1A2), metapath, dwpc
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from scipy import sparse

from dwpc import compute_dwpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NODES_DIR = Path("data/processed/nodes")
MATRICES_DIR = Path("data/processed/matrices")
DWPC_DIR = Path("data/processed/dwpc")

TARGET_GENES = ["EEF1A1", "EEF1A2"]

# metapath_name -> list of (metaedge_file_stub, transpose?) applied in order
METAPATHS = {
    "CbG": [("CbG", False)],
    "CbGiG": [("CbG", False), ("GiG", False)],
    "CbGpBP": [("CbG", False), ("GpBP", False), ("GpBP", True)],
    "CbGpMF": [("CbG", False), ("GpMF", False), ("GpMF", True)],
    "CbGpCC": [("CbG", False), ("GpCC", False), ("GpCC", True)],
    "CbGpPW": [("CbG", False), ("GpPW", False), ("GpPW", True)],
}


def load_matrix(stub: str, scope: str, transpose: bool) -> sparse.csr_matrix:
    mat = sparse.load_npz(MATRICES_DIR / f"{stub}.{scope}.npz")
    return mat.T.tocsr() if transpose else mat


def main(scope: str, damping: float) -> None:
    nodes = pd.read_csv(NODES_DIR / f"nodes.{scope}.tsv", sep="\t")
    gene_map = pd.read_csv(NODES_DIR / f"gene_symbol_to_external_id.{scope}.tsv", sep="\t")
    symbol_to_ext = dict(zip(gene_map["protein_id"], gene_map["external_id"]))
    id_by_ext = {(row.metanode_type, row.external_id): row.node_id for row in nodes.itertuples(index=False)}

    compound_nodes = nodes[nodes.metanode_type == "Compound"][["node_id", "external_id"]]
    target_node_ids = {}
    for g in TARGET_GENES:
        ext = symbol_to_ext.get(g)
        node_id = id_by_ext.get(("Gene", ext)) if ext else None
        if node_id is None:
            log.warning("Target gene %s not found in node table for scope=%s - skipping", g, scope)
        else:
            target_node_ids[g] = node_id

    rows = []
    for metapath, edge_spec in METAPATHS.items():
        mats = [load_matrix(stub, scope, transpose) for stub, transpose in edge_spec]
        dwpc_mat = compute_dwpc(mats, w=damping)
        log.info("Metapath %s: computed DWPC matrix, nnz=%d", metapath, dwpc_mat.nnz)
        for g, target_id in target_node_ids.items():
            col = dwpc_mat[:, target_id].toarray().flatten()
            for cid_row in compound_nodes.itertuples(index=False):
                val = col[cid_row.node_id]
                if val > 0:
                    rows.append(
                        {
                            "compound_id": cid_row.external_id,
                            "target_gene": g,
                            "metapath": metapath,
                            "dwpc": val,
                        }
                    )

    out = pd.DataFrame(rows)
    DWPC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DWPC_DIR / f"dwpc_observed.{scope}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    log.info("Saved %d nonzero observed DWPC rows to %s", len(out), out_path)
    if not out.empty:
        log.info("Nonzero rows per metapath:\n%s", out.groupby("metapath").size().to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    parser.add_argument("--damping", type=float, default=0.4, help="DWPC damping exponent (Himmelstein default 0.4)")
    args = parser.parse_args()
    main(args.scope, args.damping)
