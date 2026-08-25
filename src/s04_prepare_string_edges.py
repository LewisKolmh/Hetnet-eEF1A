"""Step 1.6: Convert the raw RESKO STRING interaction file into
gene-interacts-gene edges for the hetnet, annotated with NCBI gene IDs.

Reads data/raw/eef1a_string_interactions.csv and data/raw/uniprot_mapping.tsv
(for protein_id -> ncbi_gene_id), writes data/raw/gene_interacts_gene.tsv
with columns: gene_a_id, gene_b_id, gene_a_symbol, gene_b_symbol, score.
Self-edges are dropped (they don't represent an interaction with another
gene node) and edges are deduplicated (undirected: A-B == B-A).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
STRING_FILE = RAW_DIR / "eef1a_string_interactions.csv"
MAPPING_FILE = RAW_DIR / "uniprot_mapping.tsv"
OUTPUT_FILE = RAW_DIR / "gene_interacts_gene.tsv"


def main() -> None:
    if not STRING_FILE.exists():
        raise FileNotFoundError(f"{STRING_FILE} missing")
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"{MAPPING_FILE} missing - run s02_uniprot_and_go.py first")

    df = pd.read_csv(STRING_FILE)
    mapping = pd.read_csv(MAPPING_FILE, sep="\t")
    gene_id_by_symbol = dict(zip(mapping["protein_id"], mapping["ncbi_gene_id"]))

    df = df[df["preferredName_A"] != df["preferredName_B"]].copy()
    log.info("Loaded %d STRING rows (after dropping self-edges)", len(df))

    # dedupe undirected pairs, keeping max score if duplicated across query subnetworks
    pairs: dict[tuple[str, str], float] = {}
    for row in df.itertuples(index=False):
        a, b, score = row.preferredName_A, row.preferredName_B, row.score
        key = tuple(sorted((a, b)))
        pairs[key] = max(pairs.get(key, 0.0), score)

    rows = []
    n_missing_id = 0
    for (a, b), score in pairs.items():
        gid_a, gid_b = gene_id_by_symbol.get(a), gene_id_by_symbol.get(b)
        if pd.isna(gid_a) or pd.isna(gid_b) or gid_a is None or gid_b is None:
            n_missing_id += 1
        rows.append(
            {
                "gene_a_symbol": a,
                "gene_b_symbol": b,
                "gene_a_id": gid_a,
                "gene_b_id": gid_b,
                "score": score,
            }
        )

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, sep="\t", index=False)
    log.info(
        "Saved %d deduplicated gene-interacts-gene edges to %s (%d missing an NCBI gene ID mapping)",
        len(out), OUTPUT_FILE, n_missing_id,
    )


if __name__ == "__main__":
    main()
