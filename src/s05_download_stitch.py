"""Step 5: build a Compound-STITCH-Gene edge layer from STITCH v5.0.

What this adds
--------------
The ChEMBL layer only contains compounds that someone chose to assay against a
seed protein. STITCH aggregates compound-protein associations from a wider set of
sources, so it brings in compounds the seed-target assay tables never saw.

Which channels are used, and why
--------------------------------
STITCH scores each association in four channels: `experimental`, `database`,
`textmining`, `prediction`. **Only `experimental` and `database` are used here.**

`textmining` is derived from co-occurrence of a compound and a protein in the
literature. eEF1A's compound literature *is* the literature that motivated this
project, so a textmining-backed edge would score a compound for appearing in the
same abstracts as the hypothesis being tested. `prediction` is transferred from
other organisms and from chemical similarity, which reintroduces the same
circularity one step removed. Including either would make a high rank a measure of
how much has been written about a compound.

The two retained channels are recombined with STRING's standard independent-channel
formula, ``1 - prod(1 - s_i)``, on the raw channel scores. STITCH's published
`combined_score` cannot be reused because it already contains the discarded
channels.

Caveat that survives this step: the `database` channel is itself curated from the
assay literature, so this deepens rather than removes the circularity noted in
hetnet_limitations.md. It widens the compound set; it does not make the search
prospective.

Usage:
    python src/s05_download_stitch.py --scope full-interactome --min-score 400
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
SEED_FILE = RAW_DIR / "seed_proteins.tsv"
ENSP_CACHE = INTERIM_DIR / "seed_string_ids.json"

STITCH_BASE = "http://stitch-db.org/download"
LINKS_URL = f"{STITCH_BASE}/protein_chemical.links.detailed.v5.0/9606.protein_chemical.links.detailed.v5.0.tsv.gz"
CHEMICALS_URL = f"{STITCH_BASE}/chemicals.v5.0.tsv.gz"
SOURCES_URL = f"{STITCH_BASE}/chemical.sources.v5.0.tsv.gz"

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

EEF1A_ONLY = ["EEF1A1", "EEF1A2"]
DEFAULT_MIN_SCORE = 400  # STITCH/STRING "medium confidence"

# Channels deliberately discarded - see module docstring.
USED_CHANNELS = ("experimental", "database")
DISCARDED_CHANNELS = ("textmining", "prediction")


def resolve_string_ids(symbols: list[str]) -> dict[str, str]:
    """Map gene symbol -> STRING protein id (9606.ENSP...) via UniProt cross-references."""
    if ENSP_CACHE.exists():
        cached = json.loads(ENSP_CACHE.read_text())
        if all(s in cached for s in symbols):
            return {s: cached[s] for s in symbols if cached.get(s)}
    else:
        cached = {}

    for sym in symbols:
        if cached.get(sym):
            continue
        params = {
            "query": f"gene_exact:{sym} AND organism_id:9606 AND reviewed:true",
            "fields": "accession,xref_string",
            "format": "json",
            "size": 1,
        }
        r = requests.get(UNIPROT_SEARCH, params=params, timeout=60)
        r.raise_for_status()
        results = r.json().get("results", [])
        ensp = None
        if results:
            for x in results[0].get("uniProtKBCrossReferences", []):
                if x.get("database") == "STRING":
                    ensp = x.get("id")
                    break
        cached[sym] = ensp
        log.info("  %s -> %s", sym, ensp)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    ENSP_CACHE.write_text(json.dumps(cached, indent=2, sort_keys=True))
    unresolved = [s for s in symbols if not cached.get(s)]
    if unresolved:
        log.warning("No STRING id found for %s - these genes get no STITCH edges", unresolved)
    return {s: cached[s] for s in symbols if cached.get(s)}


def _stream_lines(url: str):
    """Yield decoded lines from a remote gzip file without storing it."""
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with gzip.open(resp.raw, mode="rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line


def fetch_links(ensp_by_symbol: dict[str, str], min_score: int) -> pd.DataFrame:
    """Stream the human protein-chemical links, keeping seed proteins only."""
    symbol_by_ensp = {v: k for k, v in ensp_by_symbol.items()}
    rows = []
    n_seen = 0
    for i, line in enumerate(_stream_lines(LINKS_URL)):
        if i == 0:
            header = line.split()
            log.info("STITCH links columns: %s", header)
            idx = {name: header.index(name) for name in
                   ("chemical", "protein", "experimental", "prediction", "database", "textmining")}
            continue
        parts = line.split()
        n_seen += 1
        sym = symbol_by_ensp.get(parts[idx["protein"]])
        if sym is None:
            continue
        exp = int(parts[idx["experimental"]])
        db = int(parts[idx["database"]])
        if exp == 0 and db == 0:
            continue
        # STRING independent-channel combination on the retained channels only.
        channel_score = 1.0
        for ch in USED_CHANNELS:
            channel_score *= 1.0 - int(parts[idx[ch]]) / 1000.0
        channel_score = round((1.0 - channel_score) * 1000)
        if channel_score < min_score:
            continue
        rows.append({
            "stitch_chemical": parts[idx["chemical"]],
            "protein_id": sym,
            "string_protein": parts[idx["protein"]],
            "experimental": exp,
            "database": db,
            "textmining_discarded": int(parts[idx["textmining"]]),
            "prediction_discarded": int(parts[idx["prediction"]]),
            "channel_score": channel_score,
        })
        if len(rows) % 2000 == 0:
            log.info("  ...%d seed-protein edges kept (%d rows scanned)", len(rows), n_seen)
    log.info("Scanned %d association rows; kept %d on seed proteins at channel_score >= %d",
             n_seen, len(rows), min_score)
    return pd.DataFrame(rows)


def map_to_chembl(chemicals: set[str]) -> dict[str, str]:
    """STITCH chemical id -> ChEMBL id, from chemical.sources.v5.0.tsv.gz."""
    out: dict[str, str] = {}
    for line in _stream_lines(SOURCES_URL):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        flat, stereo, source, source_id = parts[0], parts[1], parts[2], parts[3]
        if source != "ChEMBL":
            continue
        for cid in (flat, stereo):
            if cid in chemicals and cid not in out:
                out[cid] = source_id
    log.info("Mapped %d of %d STITCH chemicals to ChEMBL ids", len(out), len(chemicals))
    return out


def fetch_structures(chemicals: set[str]) -> pd.DataFrame:
    """Name / mass / SMILES for the retained STITCH chemicals."""
    rows = []
    for i, line in enumerate(_stream_lines(CHEMICALS_URL)):
        parts = line.rstrip("\n").split("\t")
        if i == 0:
            log.info("STITCH chemicals columns: %s", parts)
            continue
        if parts[0] in chemicals:
            rows.append({
                "stitch_chemical": parts[0],
                "stitch_name": parts[1] if len(parts) > 1 else None,
                "molecular_weight": parts[2] if len(parts) > 2 else None,
                "stitch_smiles": parts[3] if len(parts) > 3 else None,
            })
    return pd.DataFrame(rows)


def subset_from_full(scope: str) -> bool:
    """Derive the eef1a-only files by subsetting the full-interactome ones.

    The eef1a-only scope is a strict subset of the full-interactome seed set, so
    filtering the already-downloaded edge table on protein_id is exactly equivalent
    to re-streaming 15M association rows for two proteins.
    """
    edges_full = RAW_DIR / "compound_stitch_gene.full-interactome.tsv"
    comps_full = RAW_DIR / "compounds_stitch.full-interactome.tsv"
    if scope != "eef1a-only" or not (edges_full.exists() and comps_full.exists()):
        return False
    edges = pd.read_csv(edges_full, sep="\t")
    edges = edges[edges["protein_id"].isin(EEF1A_ONLY)]
    comps = pd.read_csv(comps_full, sep="\t")
    comps = comps[comps["compound_id"].isin(set(edges["compound_id"]))]
    edges.to_csv(RAW_DIR / f"compound_stitch_gene.{scope}.tsv", sep="\t", index=False)
    comps.to_csv(RAW_DIR / f"compounds_stitch.{scope}.tsv", sep="\t", index=False)
    log.info("Derived eef1a-only STITCH layer from the full-interactome files: "
             "%d edges, %d compounds", len(edges), len(comps))
    return True


def main(scope: str, min_score: int) -> None:
    if subset_from_full(scope):
        return
    seed = pd.read_csv(SEED_FILE, sep="\t")
    symbols = seed["protein_id"].tolist()
    if scope == "eef1a-only":
        symbols = [s for s in symbols if s in EEF1A_ONLY]
    log.info("Scope=%s: resolving STRING ids for %d seed protein(s)", scope, len(symbols))
    ensp_by_symbol = resolve_string_ids(symbols)

    links = fetch_links(ensp_by_symbol, min_score)
    if links.empty:
        log.warning("No STITCH edges retained - writing empty files with headers")

    chemicals = set(links["stitch_chemical"]) if not links.empty else set()
    chembl_by_chemical = map_to_chembl(chemicals) if chemicals else {}
    structures = fetch_structures(chemicals) if chemicals else pd.DataFrame(
        columns=["stitch_chemical", "stitch_name", "molecular_weight", "stitch_smiles"])

    def pubchem_cid(stitch_id: str) -> int | None:
        # STITCH ids are CIDm<10 digits> (flat) or CIDs<10 digits> (stereo-specific)
        if stitch_id.startswith(("CIDm", "CIDs")):
            try:
                return int(stitch_id[4:])
            except ValueError:
                return None
        return None

    if not links.empty:
        links["compound_id"] = [
            chembl_by_chemical.get(c) or f"PUBCHEM:{pubchem_cid(c)}"
            for c in links["stitch_chemical"]
        ]
        links["mapped_to_chembl"] = [c in chembl_by_chemical for c in links["stitch_chemical"]]
    else:
        links["compound_id"] = []
        links["mapped_to_chembl"] = []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    edges_path = RAW_DIR / f"compound_stitch_gene.{scope}.tsv"
    links.to_csv(edges_path, sep="\t", index=False)
    log.info("Saved %d STITCH edges (%d compounds x %d proteins) to %s",
             len(links), links["compound_id"].nunique() if not links.empty else 0,
             links["protein_id"].nunique() if not links.empty else 0, edges_path)

    if not structures.empty:
        structures["compound_id"] = [
            chembl_by_chemical.get(c) or f"PUBCHEM:{pubchem_cid(c)}"
            for c in structures["stitch_chemical"]
        ]
        structures["pubchem_cid"] = [pubchem_cid(c) for c in structures["stitch_chemical"]]
    comp_path = RAW_DIR / f"compounds_stitch.{scope}.tsv"
    structures.to_csv(comp_path, sep="\t", index=False)
    log.info("Saved %d STITCH compound records to %s", len(structures), comp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE,
                        help="Minimum recombined experimental+database score (0-1000); 400 = STITCH medium confidence")
    args = parser.parse_args()
    main(args.scope, args.min_score)
