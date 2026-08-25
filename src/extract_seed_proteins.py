"""Extract seed proteins and their degree from the RESKO STRING interaction file.

Reads data/raw/eef1a_string_interactions.csv (columns: query_protein,
preferredName_A, preferredName_B, score) and produces
data/raw/seed_proteins.tsv with one row per unique protein and its
degree (number of distinct STRING partners across both eEF1A1 and
eEF1A2 sub-networks).
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
INPUT_FILE = RAW_DIR / "eef1a_string_interactions.csv"
OUTPUT_FILE = RAW_DIR / "seed_proteins.tsv"


def load_interactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected STRING interactions file at {path}. "
            "Copy your RESKO results/eef1a_string_interactions.csv there first."
        )
    df = pd.read_csv(path)
    required = {"preferredName_A", "preferredName_B"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{path} contains no rows.")
    return df


def compute_degrees(df: pd.DataFrame) -> pd.DataFrame:
    """Undirected degree: count distinct partners per protein across A/B columns."""
    partners: dict[str, set[str]] = defaultdict(set)
    for row in df.itertuples(index=False):
        a, b = row.preferredName_A, row.preferredName_B
        if a == b:
            # self-edge (e.g. EEF1A1-EEF1A1 in query overlap) - still register the node
            partners[a].add(a)
            continue
        partners[a].add(b)
        partners[b].add(a)

    records = [
        {"protein_id": p, "gene_name": p, "degree": len(neighbors)}
        for p, neighbors in partners.items()
    ]
    out = pd.DataFrame.from_records(records).sort_values("degree", ascending=False)
    out = out.reset_index(drop=True)
    return out


def main() -> None:
    log.info("Loading STRING interactions from %s", INPUT_FILE)
    df = load_interactions(INPUT_FILE)
    log.info("Loaded %d interaction rows", len(df))

    seed = compute_degrees(df)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seed.to_csv(OUTPUT_FILE, sep="\t", index=False)

    n = len(seed)
    if n == 0:
        log.error("No proteins extracted - aborting.")
        sys.exit(1)

    log.info("Total unique proteins: %d", n)
    log.info(
        "Degree stats: min=%d max=%d median=%.1f",
        seed["degree"].min(),
        seed["degree"].max(),
        seed["degree"].median(),
    )
    log.info("Saved to %s", OUTPUT_FILE)
    print(seed.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
