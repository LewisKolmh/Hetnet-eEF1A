"""Step 1.3/1.4: Map seed proteins to UniProt accessions, NCBI gene IDs, and
GO annotations (Biological Process / Molecular Function / Cellular Component).

Populates data/raw/uniprot_mapping.tsv (protein_id, uniprot_accession,
ncbi_gene_id) and data/raw/go_annotations.tsv (protein_id, go_id, go_term,
aspect) using the UniProtKB REST search API, restricted to reviewed
(Swiss-Prot) Homo sapiens entries. Cached in data/interim/uniprot_cache.json.
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
SEED_FILE = RAW_DIR / "seed_proteins.tsv"
MAPPING_FILE = RAW_DIR / "uniprot_mapping.tsv"
GO_FILE = RAW_DIR / "go_annotations.tsv"
CACHE_FILE = INTERIM_DIR / "uniprot_cache.json"

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"

ASPECT_MAP = {"C": "CellularComponent", "P": "BiologicalProcess", "F": "MolecularFunction"}


def fetch_entry(gene: str, cache: JsonCache) -> dict | None:
    url = (
        f"{UNIPROT_BASE}?query=gene:{gene}+AND+organism_id:9606+AND+reviewed:true"
        "&fields=accession,gene_names,xref_geneid,go_id,go_p,go_f,go_c"
        "&format=json"
    )
    data = cached_get_json(url, cache, cache_key=f"search::{gene}")
    if not data:
        return None
    results = data.get("results", [])
    if not results:
        log.warning("No reviewed UniProt entry found for gene %s", gene)
        return None
    return results[0]


def parse_ncbi_gene_id(entry: dict) -> str | None:
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "GeneID":
            return xref.get("id")
    return None


def parse_go_terms(entry: dict) -> list[dict]:
    rows = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "GO":
            continue
        go_id = xref.get("id")
        term = None
        aspect_code = None
        for prop in xref.get("properties", []):
            if prop.get("key") == "GoTerm":
                val = prop.get("value", "")
                if ":" in val:
                    aspect_code, term = val.split(":", 1)
                else:
                    term = val
        if go_id and aspect_code in ASPECT_MAP:
            rows.append({"go_id": go_id, "go_term": term, "aspect": ASPECT_MAP[aspect_code]})
    return rows


def main() -> None:
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"{SEED_FILE} missing - run extract_seed_proteins.py first")
    seed = pd.read_csv(SEED_FILE, sep="\t")
    proteins = seed["protein_id"].tolist()

    cache = JsonCache(CACHE_FILE)
    log.info("Cache loaded with %d prior entries", len(cache))

    mapping_rows = []
    go_rows = []
    for i, protein in enumerate(proteins, 1):
        entry = fetch_entry(protein, cache)
        if entry is None:
            mapping_rows.append({"protein_id": protein, "uniprot_accession": None, "ncbi_gene_id": None})
            log.info("[%d/%d] %s -> no UniProt entry", i, len(proteins), protein)
            continue
        acc = entry.get("primaryAccession")
        ncbi_id = parse_ncbi_gene_id(entry)
        mapping_rows.append({"protein_id": protein, "uniprot_accession": acc, "ncbi_gene_id": ncbi_id})

        terms = parse_go_terms(entry)
        for t in terms:
            go_rows.append({"protein_id": protein, **t})
        log.info(
            "[%d/%d] %s -> %s (NCBI gene %s), %d GO terms",
            i, len(proteins), protein, acc, ncbi_id, len(terms),
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapping_rows).to_csv(MAPPING_FILE, sep="\t", index=False)
    go_df = pd.DataFrame(go_rows)
    go_df.to_csv(GO_FILE, sep="\t", index=False)

    n_mapped = sum(1 for r in mapping_rows if r["uniprot_accession"])
    log.info("Mapped %d/%d proteins to UniProt", n_mapped, len(proteins))
    log.info("Saved %d GO annotation rows (%d unique GO terms) to %s",
              len(go_df), go_df["go_id"].nunique() if not go_df.empty else 0, GO_FILE)


if __name__ == "__main__":
    main()
