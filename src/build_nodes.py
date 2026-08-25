"""Assign a single, stable integer node ID per (metanode_type, external_id)
pair across every metanode type in the metagraph, and save the node table.

Metanodes used (matching config/metagraph.yaml, restricted to what this
project's data actually populates):
  Compound            - ChEMBL compound_id
  Gene                - NCBI gene ID (fallback to gene symbol if missing)
  BiologicalProcess   - GO id (aspect=P)
  MolecularFunction   - GO id (aspect=F)
  CellularComponent   - GO id (aspect=C)
  Pathway             - Reactome stable id

Output: data/processed/nodes/nodes.tsv with columns
  node_id (int, globally unique), metanode_type, external_id, name
and data/processed/nodes/nodes.<MetanodeType>.tsv per type for convenience.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
NODES_DIR = Path("data/processed/nodes")


def build_nodes(scope: str) -> pd.DataFrame:
    seed = pd.read_csv(RAW_DIR / "seed_proteins.tsv", sep="\t")
    mapping = pd.read_csv(RAW_DIR / "uniprot_mapping.tsv", sep="\t")
    compounds = pd.read_csv(RAW_DIR / f"compounds_chembl.{scope}.tsv", sep="\t")
    go = pd.read_csv(RAW_DIR / "go_annotations.tsv", sep="\t")
    pw = pd.read_csv(RAW_DIR / "pathway_annotations.tsv", sep="\t")

    records = []

    # Gene nodes: prefer NCBI gene ID, fall back to symbol if a mapping is missing
    gene_key_by_symbol = {}
    for row in seed.itertuples(index=False):
        m = mapping[mapping["protein_id"] == row.protein_id]
        ncbi_id = m["ncbi_gene_id"].iloc[0] if len(m) and pd.notna(m["ncbi_gene_id"].iloc[0]) else None
        key = f"ncbi:{int(ncbi_id)}" if ncbi_id else f"symbol:{row.protein_id}"
        gene_key_by_symbol[row.protein_id] = key
        records.append({"metanode_type": "Gene", "external_id": key, "name": row.protein_id})

    # Compound nodes
    for row in compounds.itertuples(index=False):
        records.append(
            {
                "metanode_type": "Compound",
                "external_id": row.compound_id,
                "name": row.pref_name if isinstance(row.pref_name, str) else row.compound_id,
            }
        )

    # GO nodes, split by aspect into the three metanode types
    for aspect, rows in go.groupby("aspect"):
        for gid, name in rows[["go_id", "go_term"]].drop_duplicates().itertuples(index=False):
            records.append({"metanode_type": aspect, "external_id": gid, "name": name})

    # Pathway nodes
    for pid, name in pw[["pathway_id", "pathway_name"]].drop_duplicates().itertuples(index=False):
        records.append({"metanode_type": "Pathway", "external_id": pid, "name": name})

    nodes = pd.DataFrame(records).drop_duplicates(subset=["metanode_type", "external_id"]).reset_index(drop=True)
    nodes.insert(0, "node_id", range(len(nodes)))
    nodes.attrs["gene_key_by_symbol"] = gene_key_by_symbol
    return nodes


def main(scope: str) -> None:
    nodes = build_nodes(scope)
    NODES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NODES_DIR / f"nodes.{scope}.tsv"
    nodes.to_csv(out_path, sep="\t", index=False)
    log.info("Saved %d nodes to %s", len(nodes), out_path)
    for mtype, grp in nodes.groupby("metanode_type"):
        log.info("  %s: %d nodes", mtype, len(grp))

    # also save the gene symbol -> node external_id lookup, needed by build_matrices.py
    gene_map = nodes.attrs["gene_key_by_symbol"]
    pd.Series(gene_map, name="external_id").rename_axis("protein_id").reset_index().to_csv(
        NODES_DIR / f"gene_symbol_to_external_id.{scope}.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    args = parser.parse_args()
    main(args.scope)
