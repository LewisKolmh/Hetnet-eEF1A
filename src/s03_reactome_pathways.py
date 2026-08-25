"""Step 1.5: Map seed proteins to Reactome pathways via their UniProt accession.

Reads data/raw/uniprot_mapping.tsv (produced by s02_uniprot_and_go.py) and
queries the Reactome ContentService mapping endpoint per accession, human
species only. Writes data/raw/pathway_annotations.tsv
(protein_id, uniprot_accession, pathway_id, pathway_name).
Cached in data/interim/reactome_cache.json.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from chembl_client import JsonCache, cached_get_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
MAPPING_FILE = RAW_DIR / "uniprot_mapping.tsv"
OUTPUT_FILE = RAW_DIR / "pathway_annotations.tsv"
CACHE_FILE = INTERIM_DIR / "reactome_cache.json"

REACTOME_BASE = "https://reactome.org/ContentService/data/mapping/UniProt"


def fetch_pathways(accession: str, cache: JsonCache) -> list[dict]:
    url = f"{REACTOME_BASE}/{accession}/pathways?species=9606"
    data = cached_get_json(url, cache, cache_key=f"pathways::{accession}")
    if not data:
        return []
    # Reactome returns a bare list, not a dict - cached_get_json stores whatever .json() gives
    if isinstance(data, list):
        return data
    return []


def main() -> None:
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"{MAPPING_FILE} missing - run s02_uniprot_and_go.py first")
    mapping = pd.read_csv(MAPPING_FILE, sep="\t")
    mapping = mapping.dropna(subset=["uniprot_accession"])

    cache = JsonCache(CACHE_FILE)
    log.info("Cache loaded with %d prior entries", len(cache))

    rows = []
    for i, row in enumerate(mapping.itertuples(index=False), 1):
        pathways = fetch_pathways(row.uniprot_accession, cache)
        for p in pathways:
            rows.append(
                {
                    "protein_id": row.protein_id,
                    "uniprot_accession": row.uniprot_accession,
                    "pathway_id": p.get("stId"),
                    "pathway_name": p.get("displayName"),
                }
            )
        log.info(
            "[%d/%d] %s (%s) -> %d pathways",
            i, len(mapping), row.protein_id, row.uniprot_accession, len(pathways),
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, sep="\t", index=False)
    n_unique = out["pathway_id"].nunique() if not out.empty else 0
    log.info("Saved %d pathway annotation rows (%d unique pathways) to %s", len(out), n_unique, OUTPUT_FILE)


if __name__ == "__main__":
    main()
