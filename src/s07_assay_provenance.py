"""Step 7: label each compound-binds-gene edge with the kind of experiment behind it.

Why this exists
---------------
The activity filter (src/activity_filter.py) removes non-detects and unmeasured
records, but it cannot tell a targeted binding measurement from a multiplexed
chemoproteomics deposit. Both carry an exact relation, a numeric pchembl, assay
type B, and ChEMBL confidence_score 9 - the conventional quality fields do not
separate them. Inspecting the top-ranked ChEMBL hits on this graph shows why it
matters:

* CHEMBL5653589 and CHEMBL3752910 carry Kd values against 13 and 10 seed
  proteins respectively - EEF1A2, EEF1B2, EEF1D, EEF1G, ETF1, RAN, PAPSS1, RPL3,
  RPL4, RPL7, RPL18A, RPS2, RPS3, RPS3A - every one of them from the same
  document (CHEMBL5649169) and described as "Binding affinity to human <X> ... by
  Kinobead based pull down assay". This is one competition-pulldown experiment
  reporting apparent affinities across a captured sub-proteome, not thirteen
  independent target validations.
* MOLIBRESIB's seed-protein edges come from document CHEMBL5696440, described as
  "assessed as fold change ... by colloidal coomassie staining based LC-MS/MS
  analysis" - an abundance/enrichment readout. Its fold-change rows are already
  dropped (FC is not an affinity type), but IC50 rows from the same experiment
  survive and read as target engagement of eEF1A1.

Neither is fabricated evidence, and neither is deleted here. But a compound whose
only link to eEF1A is an apparent affinity from a lysate pulldown is a different
proposition from one with a dedicated assay, and the ranking must say which it is
rather than letting the reader assume the latter.

Mechanical rule
---------------
For each (compound, document) pair, count the distinct seed proteins it reports.
A targeted paper reports one or two; a chemoproteomic deposit reports many.
Pairs at or above ``MULTIPLEX_MIN_PROTEINS`` seed proteins are labelled
``multiplexed_proteomics``. A keyword scan of the assay description
(``PROTEOMIC_KEYWORDS``) is recorded alongside it as independent corroboration,
not as the criterion, so the label does not depend on how a curator phrased it.

Output: data/processed/assay_provenance.<scope>.tsv, one row per edge, joined
into the ranked table by rank_compounds.py.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CACHE_FILE = RAW_DIR / "assay_cache.json"
CHEMBL_ASSAY = "https://www.ebi.ac.uk/chembl/api/data/assay/{}.json"

# A single document reporting this many distinct seed proteins for one compound is
# a multiplexed experiment. Set to 4: the largest targeted series in this dataset
# reports 2 seeds (an eEF1A1/eEF1A2 paralogue pair), and the chemoproteomic
# deposits report 10-13.
MULTIPLEX_MIN_PROTEINS = 4

# Keywords for a *proteome-scale readout*, i.e. the quantity measured is protein
# abundance or enrichment across many proteins at once. "Pull down" alone is
# deliberately absent: a biotinylated-probe pulldown read out by western blot
# against one named protein is targeted evidence, and three of the strongest
# eEF1A1/eEF1A2 binders in this dataset are measured exactly that way.
PROTEOME_READOUT_KEYWORDS = (
    "kinobead", "lc-ms", "mass spectrometr", "proteome", "proteomic",
    "chemoproteomic", "coomassie",
)

# Explicitly named biophysical binding methods. This is a positive signal used to
# promote an edge, never to demote one: if the curator wrote "surface plasmon
# resonance", a direct binding measurement was made against the named protein.
BIOPHYSICAL_METHOD_KEYWORDS = (
    "surface plasmon resonance", "isothermal titration", "microscale thermophoresis",
    "fluorescence polarization", "fluorescence polarisation", "x-ray",
    "co-crystal", "crystal structure", "nmr", "biolayer interferometry",
    "differential scanning fluorimetry", "thermal shift", "radioligand",
)


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def fetch_assays(assay_ids: list[str]) -> pd.DataFrame:
    cache = _load_cache()
    session = requests.Session()
    for i, aid in enumerate(assay_ids, 1):
        if aid in cache:
            continue
        try:
            r = session.get(CHEMBL_ASSAY.format(aid), timeout=60)
            r.raise_for_status()
            d = r.json()
            cache[aid] = {
                "assay_chembl_id": aid,
                "assay_description": d.get("description") or "",
                "assay_type_chembl": d.get("assay_type"),
                "confidence_score": d.get("confidence_score"),
                "document_chembl_id": d.get("document_chembl_id"),
                "assay_organism": d.get("assay_organism"),
                "cell_chembl_id": d.get("cell_chembl_id"),
            }
        except requests.RequestException as exc:
            log.warning("assay %s failed: %s", aid, exc)
            cache[aid] = {"assay_chembl_id": aid, "assay_description": "",
                          "assay_type_chembl": None, "confidence_score": None,
                          "document_chembl_id": None, "assay_organism": None,
                          "cell_chembl_id": None}
        if i % 25 == 0:
            log.info("fetched %d/%d assays", i, len(assay_ids))
    CACHE_FILE.write_text(json.dumps(cache, indent=0))
    return pd.DataFrame([cache[a] for a in assay_ids])


def label_provenance(activities: pd.DataFrame, assays: pd.DataFrame) -> pd.DataFrame:
    """One row per (compound, protein) with the provenance of its supporting assays."""
    df = activities.merge(assays, on="assay_chembl_id", how="left")

    # Seed proteins per (compound, document) - the multiplexing measure.
    doc_breadth = (
        df.dropna(subset=["document_chembl_id"])
        .groupby(["compound_id", "document_chembl_id"])["protein_id"]
        .nunique()
        .rename("n_seed_proteins_in_document")
        .reset_index()
    )
    df = df.merge(doc_breadth, on=["compound_id", "document_chembl_id"], how="left")
    df["n_seed_proteins_in_document"] = df["n_seed_proteins_in_document"].fillna(1)

    desc = df["assay_description"].fillna("").str.lower()
    df["proteome_scale_readout"] = desc.apply(
        lambda s: any(k in s for k in PROTEOME_READOUT_KEYWORDS)
    )
    df["biophysical_method"] = desc.apply(
        lambda s: any(k in s for k in BIOPHYSICAL_METHOD_KEYWORDS)
    )
    df["multiplexed_proteomics"] = df["n_seed_proteins_in_document"] >= MULTIPLEX_MIN_PROTEINS

    per_edge = df.groupby(["compound_id", "protein_id"], as_index=False).agg(
        n_supporting_assays=("assay_chembl_id", "nunique"),
        n_supporting_documents=("document_chembl_id", "nunique"),
        max_seed_proteins_in_one_document=("n_seed_proteins_in_document", "max"),
        multiplexed_proteomics=("multiplexed_proteomics", "all"),
        proteome_scale_readout=("proteome_scale_readout", "all"),
        biophysical_method=("biophysical_method", "any"),
        min_confidence_score=("confidence_score", "min"),
        assay_types=("assay_type", lambda x: "|".join(sorted(set(x.dropna())))),
        example_assay_description=("assay_description", "first"),
    )
    # Three grades from structured metadata plus one promotion from an explicitly
    # named method. Note what is deliberately NOT attempted: a binding vs
    # cell-based-functional split. ChEMBL cannot support one here - the HSF1
    # "Inhibition of 17AAG-induced HSF1-mediated HSP72 expression ... by ELISA"
    # assays are typed B (binding) and the eEF1A1/eEF1A2 surface-plasmon-resonance
    # assays carry a cell_chembl_id (MDA-MB-231, the source of the protein), so
    # both assay_type and the cell-line field point the wrong way. `assay_reported`
    # therefore means "ChEMBL reports an activity against this protein and the
    # readout cannot be established from structured fields", and the full assay
    # description travels with every row so a reader can check it.
    def _grade(mux: bool, pro: bool, bio: bool) -> str:
        if mux:
            return "multiplexed_pulldown"
        if pro:
            return "proteome_readout"
        if bio:
            return "direct_biophysical"
        return "assay_reported"

    per_edge["evidence_grade"] = [
        _grade(m, p, b) for m, p, b in zip(
            per_edge["multiplexed_proteomics"],
            per_edge["proteome_scale_readout"],
            per_edge["biophysical_method"],
        )
    ]
    return per_edge


def main(scope: str) -> None:
    act_path = RAW_DIR / f"compound_gene_activities.{scope}.tsv"
    edge_path = RAW_DIR / f"compound_binds_gene.{scope}.tsv"
    if not act_path.exists() or not edge_path.exists():
        raise SystemExit(f"missing {act_path} or {edge_path}; run s01 first")

    activities = pd.read_csv(act_path, sep="\t")
    edges = pd.read_csv(edge_path, sep="\t")

    # Only the activity records that actually survived into edges matter here.
    pairs = set(zip(edges["compound_id"], edges["protein_id"]))
    keep = [
        (c, p) in pairs for c, p in zip(activities["compound_id"], activities["protein_id"])
    ]
    activities = activities[keep]
    assay_ids = sorted(activities["assay_chembl_id"].dropna().unique())
    log.info("%d edges backed by %d distinct assays", len(pairs), len(assay_ids))

    assays = fetch_assays(assay_ids)
    per_edge = label_provenance(activities, assays)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"assay_provenance.{scope}.tsv"
    per_edge.to_csv(out_path, sep="\t", index=False)
    n_mux = int(per_edge["multiplexed_proteomics"].sum())
    log.info(
        "Saved %s: %d of %d edges labelled multiplexed_proteomics (>= %d seed "
        "proteins from one document)",
        out_path, n_mux, len(per_edge), MULTIPLEX_MIN_PROTEINS,
    )
    log.info("evidence grades: %s", per_edge["evidence_grade"].value_counts().to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="full-interactome",
                        choices=["full-interactome", "eef1a-only"])
    args = parser.parse_args()
    main(args.scope)
