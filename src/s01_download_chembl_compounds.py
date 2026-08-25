"""Step 1.2: Download ChEMBL compounds tested against any seed protein.

For each seed protein (data/raw/seed_proteins.tsv), find its ChEMBL
target(s) by exact synonym match restricted to Homo sapiens, then pull
every compound with a recorded activity against that target.

Resumable: raw API responses are cached in data/interim/chembl_cache.json
keyed by request URL, so a re-run skips anything already fetched.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from chembl_client import JsonCache, cached_get_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
SEED_FILE = RAW_DIR / "seed_proteins.tsv"
def targets_file(scope: str) -> Path:
    return RAW_DIR / f"chembl_targets.{scope}.tsv"


def compounds_file(scope: str) -> Path:
    return RAW_DIR / f"compounds_chembl.{scope}.tsv"
CACHE_FILE = INTERIM_DIR / "chembl_cache.json"

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def find_targets(protein: str, cache: JsonCache) -> list[dict]:
    """Exact-synonym, Homo sapiens, SINGLE PROTEIN targets only.

    Protein-complex / protein-nucleic-acid-complex targets (e.g. "80S
    Ribosome", "Ran/Importin-beta1/Snurportin-1") are excluded: they are
    shared across many unrelated seed proteins and their activity tables
    are dominated by compounds probing the complex generically rather
    than this specific gene product, which would blow up the compound
    set with edges that are not gene-specific.
    """
    url = f"{CHEMBL_BASE}/target?target_synonym__iexact={protein}&organism=Homo+sapiens&format=json"
    data = cached_get_json(url, cache, cache_key=f"target_search::{protein}")
    if not data:
        return []
    targets = data.get("targets", [])
    single = [t for t in targets if t.get("target_type") == "SINGLE PROTEIN"]
    dropped = len(targets) - len(single)
    if dropped:
        log.info(
            "  %s: dropped %d non-single-protein target(s): %s",
            protein, dropped,
            [t.get("pref_name") for t in targets if t.get("target_type") != "SINGLE PROTEIN"],
        )
    return single


def compounds_for_target(
    target_id: str, cache: JsonCache, limit: int = 1000, max_compounds: int = 2000,
    edge_rows: list[dict] | None = None, protein_id: str | None = None,
) -> list[dict]:
    """Distinct compounds with an activity record against this target.

    If `edge_rows` is given, every (compound, target) activity is also
    appended there (one row per activity record, deduped later by caller)
    so the compound-binds-gene edge table retains the actual activity
    type/relation/value rather than just "some activity exists".
    """
    compounds: dict[str, dict] = {}
    offset = 0
    while True:
        url = (
            f"{CHEMBL_BASE}/activity?target_chembl_id={target_id}"
            f"&limit={limit}&offset={offset}&format=json"
        )
        data = cached_get_json(url, cache, cache_key=f"activity_page::{target_id}::{offset}")
        if not data:
            break
        activities = data.get("activities", [])
        for act in activities:
            cid = act.get("molecule_chembl_id")
            if not cid:
                continue
            if cid not in compounds:
                compounds[cid] = {
                    "compound_id": cid,
                    "pref_name": act.get("molecule_pref_name"),
                }
            if edge_rows is not None:
                edge_rows.append(
                    {
                        "compound_id": cid,
                        "target_chembl_id": target_id,
                        "protein_id": protein_id,
                        "standard_type": act.get("standard_type"),
                        "standard_relation": act.get("standard_relation"),
                        "standard_value": act.get("standard_value"),
                        "standard_units": act.get("standard_units"),
                    }
                )
        meta = data.get("page_meta", {})
        if not meta.get("next") or not activities:
            break
        if len(compounds) >= max_compounds:
            log.warning(
                "  target %s: hit max_compounds cap (%d) - stopping pagination early "
                "(this target is very heavily screened; not all activities were fetched)",
                target_id, max_compounds,
            )
            break
        offset += limit
    return list(compounds.values())


def fetch_compound_details(compound_id: str, cache: JsonCache) -> dict:
    url = f"{CHEMBL_BASE}/molecule/{compound_id}.json"
    data = cached_get_json(url, cache, cache_key=f"molecule::{compound_id}")
    if not data:
        return {"compound_id": compound_id, "canonical_smiles": None, "pref_name": None, "max_phase": None}
    structures = data.get("molecule_structures") or {}
    return {
        "compound_id": compound_id,
        "canonical_smiles": structures.get("canonical_smiles"),
        "pref_name": data.get("pref_name"),
        "max_phase": data.get("max_phase"),
    }


EEF1A_ONLY = ["EEF1A1", "EEF1A2"]


def main(dry_run_n: int | None, scope: str) -> None:
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"{SEED_FILE} missing - run extract_seed_proteins.py first")
    seed = pd.read_csv(SEED_FILE, sep="\t")

    if scope == "eef1a-only":
        proteins = [p for p in seed["protein_id"].tolist() if p in EEF1A_ONLY]
        log.info(
            "Scope=eef1a-only: restricting to %s (compounds that bind eEF1A1/eEF1A2 directly). "
            "Use --scope full-interactome for all %d seed proteins (heavier - run on HPC).",
            proteins, len(seed),
        )
    else:
        proteins = seed["protein_id"].tolist()
        log.info("Scope=full-interactome: using all %d seed proteins", len(proteins))

    if dry_run_n:
        proteins = proteins[:dry_run_n]
        log.info("DRY RUN: limiting to first %d proteins", dry_run_n)

    cache = JsonCache(CACHE_FILE)
    log.info("Cache loaded with %d prior entries", len(cache))

    target_rows = []
    all_target_ids: set[str] = set()
    for i, protein in enumerate(proteins, 1):
        targets = find_targets(protein, cache)
        for t in targets:
            tid = t.get("target_chembl_id")
            all_target_ids.add(tid)
            target_rows.append(
                {
                    "protein_id": protein,
                    "target_chembl_id": tid,
                    "pref_name": t.get("pref_name"),
                    "organism": t.get("organism"),
                    "target_type": t.get("target_type"),
                }
            )
        log.info("[%d/%d] %s -> %d ChEMBL target(s)", i, len(proteins), protein, len(targets))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets_df = pd.DataFrame(target_rows)
    targets_df.to_csv(targets_file(scope), sep="\t", index=False)
    log.info("Saved %d target mappings to %s", len(targets_df), targets_file(scope))

    protein_by_target = {r["target_chembl_id"]: r["protein_id"] for r in target_rows}
    all_compounds: dict[str, dict] = {}
    edge_rows: list[dict] = []
    for i, tid in enumerate(sorted(all_target_ids), 1):
        comps = compounds_for_target(tid, cache, edge_rows=edge_rows, protein_id=protein_by_target.get(tid))
        for c in comps:
            all_compounds.setdefault(c["compound_id"], c)
        log.info(
            "[%d/%d] target %s -> %d compounds (running total unique: %d)",
            i, len(all_target_ids), tid, len(comps), len(all_compounds),
        )

    log.info("Fetching molecule details for %d unique compounds", len(all_compounds))
    detail_rows = []
    for i, cid in enumerate(all_compounds, 1):
        detail_rows.append(fetch_compound_details(cid, cache))
        if i % 50 == 0 or i == len(all_compounds):
            log.info("  ...%d/%d molecule records fetched", i, len(all_compounds))

    out = pd.DataFrame(detail_rows)
    if out.empty:
        log.warning("No compounds found for any seed protein - writing empty file with header")
        out = pd.DataFrame(columns=["compound_id", "canonical_smiles", "pref_name", "max_phase"])
    out.to_csv(compounds_file(scope), sep="\t", index=False)
    log.info("Saved %d compounds to %s", len(out), compounds_file(scope))

    edges_df = pd.DataFrame(edge_rows).drop_duplicates(subset=["compound_id", "target_chembl_id"])
    edges_path = RAW_DIR / f"compound_binds_gene.{scope}.tsv"
    edges_df.to_csv(edges_path, sep="\t", index=False)
    log.info(
        "Saved %d compound-binds-gene edges (%d compounds x %d proteins) to %s",
        len(edges_df), edges_df["compound_id"].nunique() if not edges_df.empty else 0,
        edges_df["protein_id"].nunique() if not edges_df.empty else 0, edges_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=int, default=None, help="Limit to first N seed proteins")
    parser.add_argument(
        "--scope",
        choices=["eef1a-only", "full-interactome"],
        default="eef1a-only",
        help=(
            "eef1a-only (default): compounds binding EEF1A1/EEF1A2 directly - fast, run anywhere. "
            "full-interactome: compounds binding any of the 18 seed proteins - much larger "
            "(one target's activity table alone can run into the hundreds), intended for the HPC."
        ),
    )
    args = parser.parse_args()
    main(args.dry_run, args.scope)
