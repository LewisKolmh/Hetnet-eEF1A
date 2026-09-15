"""Step 6: annotate candidate compounds with structure and purchasability.

For every compound node in the graph this resolves:

* `canonical_smiles`  - from ChEMBL for ChEMBL-identified compounds, from PubChem
  for STITCH-only compounds, so every row in the deliverable has a structure that
  can be pasted into a docking run or a vendor search.
* `inchikey`          - from ChEMBL (or PubChem for STITCH-only compounds), for
  unambiguous vendor/catalogue lookup.
* `pubchem_cid`       - resolved from the InChIKey via PubChem, or taken directly
  for compounds that entered through STITCH (whose ids are PubChem-derived).
* `n_vendors`, `example_vendors` - from PubChem's "Chemical Vendors" source
  category, which is the count of suppliers PubChem has a catalogue record from.

What `n_vendors` does and does not mean
---------------------------------------
It is evidence that a catalogue entry exists, not a quote. It does not establish
price, purity, amount available, stock, shipping restrictions, or that the
supplier is one an academic lab can order from. A natural product with one
supplier listing milligrams at custom-synthesis prices and a screening compound
with forty suppliers both appear here as a number. Treat it as a triage column:
`n_vendors == 0` means do not plan an assay around it without checking synthesis,
and anything above zero still needs a real quote before it goes in a budget.

Responses are cached in `data/interim/purchasability_cache.json`, so re-runs are
free and the annotation is reproducible without re-querying.

Usage:
    python src/s06_annotate_purchasable.py --scope full-interactome
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
NODES_DIR = Path("data/processed/nodes")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
CACHE_FILE = INTERIM_DIR / "purchasability_cache.json"

PUBCHEM_INCHIKEY_CIDS = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{key}/cids/JSON"
PUBCHEM_PROPS = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{props}/JSON"
PUBCHEM_CATEGORIES = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/categories/compound/{cid}/JSON"

REQUEST_PAUSE = 0.25  # PubChem asks for <= 5 requests/second


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}
        self._dirty = False

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))
            self._dirty = False


def cid_from_inchikey(inchikey: str, cache: Cache) -> str | None:
    """Resolve an InChIKey to a PubChem CID.

    Structure-key lookup rather than name matching: a compound with three ChEMBL
    synonyms and no PubChem name entry still resolves, and there is no risk of a
    name collision pulling in the wrong compound.
    """
    key = f"cid_from_inchikey::{inchikey}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    cid = None
    try:
        r = requests.get(PUBCHEM_INCHIKEY_CIDS.format(key=inchikey), timeout=30)
        if r.ok:
            cids = ((r.json().get("IdentifierList") or {}).get("CID") or [])
            cid = str(cids[0]) if cids else None
        elif r.status_code != 404:
            log.warning("PubChem CID lookup for %s returned HTTP %d", inchikey, r.status_code)
            return None
    except requests.RequestException as exc:
        log.warning("PubChem CID lookup failed for %s: %s", inchikey, exc)
        return None
    cache.put(key, cid)
    time.sleep(REQUEST_PAUSE)
    return cid


def pubchem_structure(cid: str, cache: Cache) -> dict:
    key = f"pubchem_struct::{cid}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    out = {"smiles": None, "inchikey": None, "molecular_weight": None}
    try:
        r = requests.get(
            PUBCHEM_PROPS.format(cid=cid, props="ConnectivitySMILES,SMILES,InChIKey,MolecularWeight"),
            timeout=60,
        )
        if r.ok:
            props = (r.json().get("PropertyTable") or {}).get("Properties") or []
            if props:
                p = props[0]
                out["smiles"] = p.get("SMILES") or p.get("ConnectivitySMILES")
                out["inchikey"] = p.get("InChIKey")
                out["molecular_weight"] = p.get("MolecularWeight")
    except requests.RequestException as exc:
        log.warning("PubChem property lookup failed for CID %s: %s", cid, exc)
        return out
    cache.put(key, out)
    time.sleep(REQUEST_PAUSE)
    return out


def pubchem_vendors(cid: str, cache: Cache) -> dict:
    key = f"pubchem_vendors::{cid}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    out = {"n_vendors": 0, "example_vendors": None}
    try:
        r = requests.get(PUBCHEM_CATEGORIES.format(cid=cid), timeout=60)
        if r.ok:
            cats = ((r.json().get("SourceCategories") or {}).get("Categories") or [])
            for cat in cats:
                if cat.get("Category") == "Chemical Vendors":
                    names = [s.get("SourceName") for s in cat.get("Sources", []) if s.get("SourceName")]
                    out["n_vendors"] = len(names)
                    out["example_vendors"] = "; ".join(sorted(names)[:5])
                    break
    except requests.RequestException as exc:
        log.warning("PubChem vendor lookup failed for CID %s: %s", cid, exc)
        return out
    cache.put(key, out)
    time.sleep(REQUEST_PAUSE)
    return out


def main(scope: str) -> None:
    nodes = pd.read_csv(NODES_DIR / f"nodes.{scope}.tsv", sep="\t")
    compound_ids = nodes.loc[nodes.metanode_type == "Compound", "external_id"].tolist()
    log.info("Annotating %d compound nodes for scope=%s", len(compound_ids), scope)

    chembl = pd.read_csv(RAW_DIR / f"compounds_chembl.{scope}.tsv", sep="\t")
    smiles_by_chembl = dict(zip(chembl["compound_id"], chembl["canonical_smiles"]))
    name_by_chembl = dict(zip(chembl["compound_id"], chembl["pref_name"]))
    phase_by_chembl = dict(zip(chembl["compound_id"], chembl["max_phase"]))
    inchikey_by_chembl = dict(zip(chembl["compound_id"], chembl.get("inchikey", pd.Series(dtype=object))))
    alogp_by_chembl = dict(zip(chembl["compound_id"], chembl.get("alogp", pd.Series(dtype=float))))
    mw_by_chembl = dict(zip(chembl["compound_id"], chembl.get("mw", pd.Series(dtype=float))))

    stitch_path = RAW_DIR / f"compounds_stitch.{scope}.tsv"
    stitch_name: dict[str, str] = {}
    stitch_smiles: dict[str, str] = {}
    if stitch_path.exists():
        st = pd.read_csv(stitch_path, sep="\t")
        if not st.empty:
            stitch_name = dict(zip(st["compound_id"], st["stitch_name"]))
            stitch_smiles = dict(zip(st["compound_id"], st["stitch_smiles"]))

    cache = Cache(CACHE_FILE)
    rows = []
    for i, cid in enumerate(compound_ids, 1):
        row = {
            "compound_id": cid,
            "pref_name": name_by_chembl.get(cid) or stitch_name.get(cid),
            "max_phase": phase_by_chembl.get(cid),
            "canonical_smiles": smiles_by_chembl.get(cid) or stitch_smiles.get(cid),
            "pubchem_cid": None,
            "inchikey": inchikey_by_chembl.get(cid),
            "alogp": alogp_by_chembl.get(cid),
            "mw": mw_by_chembl.get(cid),
            "n_vendors": 0,
            "example_vendors": None,
            "structure_source": None,
        }
        if str(cid).startswith("CHEMBL"):
            row["structure_source"] = "ChEMBL" if row["canonical_smiles"] else None
            if isinstance(row["inchikey"], str) and row["inchikey"]:
                row["pubchem_cid"] = cid_from_inchikey(row["inchikey"], cache)
        elif str(cid).startswith("PUBCHEM:"):
            row["pubchem_cid"] = str(cid).split(":", 1)[1]

        if row["pubchem_cid"]:
            if not row["canonical_smiles"]:
                ps = pubchem_structure(row["pubchem_cid"], cache)
                row["canonical_smiles"] = ps.get("smiles")
                row["inchikey"] = row["inchikey"] or ps.get("inchikey")
                row["mw"] = row["mw"] or ps.get("molecular_weight")
                row["structure_source"] = "PubChem" if row["canonical_smiles"] else None
            vend = pubchem_vendors(row["pubchem_cid"], cache)
            row["n_vendors"] = vend.get("n_vendors", 0)
            row["example_vendors"] = vend.get("example_vendors")
        rows.append(row)
        if i % 25 == 0 or i == len(compound_ids):
            cache.flush()
            log.info("  ...%d/%d annotated", i, len(compound_ids))
    cache.flush()

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"compound_annotation.{scope}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    log.info("Saved %d annotated compounds to %s", len(out), out_path)
    log.info("With SMILES: %d | with PubChem CID: %d | with >=1 vendor: %d",
             int(out["canonical_smiles"].notna().sum()), int(out["pubchem_cid"].notna().sum()),
             int((out["n_vendors"] > 0).sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    args = parser.parse_args()
    main(args.scope)
