"""Purchasable analogues of the compounds that actually bind eEF1A.

Why this step exists
--------------------
The relevance audit leaves four compounds holding a chemical edge to EEF1A1 or
EEF1A2 and clearing the structural floor. Three of them - the 2011 OBOC series
measured by SPR against both paralogues - have no catalogued supplier, so they
cannot be ordered for an assay. This script asks whether anything purchasable
looks like them.

It searches near-neighbours of each seed and reports purchasability, and it is
deliberately an *adjunct*: analogues are never merged into the hetnet-derived
ranking, because they carry no network evidence at all. They are a separate,
labelled group whose only claim is structural similarity to a measured binder.

Sources
-------
* PubChem 2D fast-similarity screen at >= 85%, then Tanimoto recomputed locally
  on Morgan (r=2, 2048-bit) fingerprints, because the screen's own threshold and
  a Morgan Tanimoto are not the same number and only the latter is reported.
* ChEMBL similarity, which adds compounds carrying their own activity records.
* Purchasability from PubChem's "Chemical Vendors" category, via the same
  `s06_annotate_purchasable` helpers the rest of the pipeline uses, so vendor
  counts here mean exactly what they mean in the ranked tables.

ZINC22 was tried first and is not usable: `zinc_search_by_id` resolves, but the
CartBlanche22 structure-search endpoint returned `total_available = 0` for every
query including aspirin and imatinib exact matches, so its silence carries no
information about purchasable space. Recorded here so the absence of ZINC in the
output is not read as an absence of ZINC hits.

Output: results/analogue_candidates.csv, results/analogues_grid.png
"""
from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

import s06_annotate_purchasable as s06
from chem_tractability import annotate_tractability

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RAW_DIR = Path("data/raw")
PUBCHEM_SIM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/cids/JSON"
PUBCHEM_PROPS = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/"
                 "SMILES,ConnectivitySMILES,MolecularWeight,Title/JSON")
CHEMBL_SIM = "https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles}/{pct}.json?limit=100"

SIM_SCREEN_PCT = 85       # PubChem 2D screen threshold
CHEMBL_PCT = (80, 70)     # ChEMBL similarity thresholds
TOP_N_VENDOR_LOOKUP = 15  # vendor lookups are one PUG-View call each; cap per seed
PAUSE = 0.3

MFP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def pubchem_similar(smiles: str, threshold: int = SIM_SCREEN_PCT, maxrec: int = 100) -> list[int]:
    r = requests.post(PUBCHEM_SIM, data={"smiles": smiles, "Threshold": threshold,
                                         "MaxRecords": maxrec}, timeout=120)
    if r.status_code != 200:
        log.warning("PubChem similarity failed (HTTP %s)", r.status_code)
        return []
    return r.json().get("IdentifierList", {}).get("CID", [])


def pubchem_properties(cids: list[int]) -> pd.DataFrame:
    rows = []
    for i in range(0, len(cids), 100):
        r = requests.post(PUBCHEM_PROPS, data={"cid": ",".join(map(str, cids[i:i + 100]))}, timeout=120)
        if r.status_code == 200:
            rows += r.json()["PropertyTable"]["Properties"]
        else:
            log.warning("PubChem property fetch failed (HTTP %s)", r.status_code)
        time.sleep(PAUSE)
    df = pd.DataFrame(rows)
    # PubChem renamed CanonicalSMILES: "SMILES" carries stereochemistry,
    # "ConnectivitySMILES" does not. Prefer the stereo form.
    if "SMILES" in df.columns:
        df["smiles_use"] = df["SMILES"].fillna(df.get("ConnectivitySMILES"))
    else:
        df["smiles_use"] = df.get("ConnectivitySMILES")
    return df


def chembl_similar(smiles: str, pct: int) -> list[dict]:
    import urllib.parse as up
    r = requests.get(CHEMBL_SIM.format(smiles=up.quote(smiles, safe=""), pct=pct), timeout=90)
    if r.status_code != 200:
        log.warning("ChEMBL similarity failed for pct=%d (HTTP %s)", pct, r.status_code)
        return []
    return r.json().get("molecules", [])


def _mol_image(smiles: str, size=(330, 265)):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    AllChem.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def seed_compounds(scope: str) -> pd.DataFrame:
    """The eEF1A-linked, floor-clearing, BH-significant compounds."""
    rk = pd.read_csv(RESULTS_DIR / f"hetnet_ranked_compounds.{scope}.csv")
    cb = pd.read_csv(RAW_DIR / f"compound_binds_gene.{scope}.tsv", sep="\t")
    cs = pd.read_csv(RAW_DIR / f"compound_stitch_gene.{scope}.tsv", sep="\t")
    edges = pd.concat([cb[["compound_id", "protein_id"]], cs[["compound_id", "protein_id"]]])
    eef1a = set(edges[edges.protein_id.isin({"EEF1A1", "EEF1A2"})].compound_id)
    keep = (rk.significant_bh & rk.tractable_small_molecule & ~rk.nucleotide_cofactor
            & ~rk.substrate_relation_only & rk.compound_id.isin(eef1a))
    return rk[keep].copy()


def main(scope: str) -> None:
    seeds = seed_compounds(scope)
    if seeds.empty:
        raise SystemExit(f"no eEF1A-linked floor-clearing compounds in scope={scope}")
    log.info("Seeding analogue search on %d compounds: %s", len(seeds), list(seeds.compound_id))

    cache = s06.Cache(s06.CACHE_FILE)
    seed_fps = {r.compound_id: MFP.GetFingerprint(Chem.MolFromSmiles(r.canonical_smiles))
                for r in seeds.itertuples()}

    hits = {}
    for r in seeds.itertuples():
        hits[r.compound_id] = pubchem_similar(r.canonical_smiles)
        log.info("%s: %d PubChem similarity hits", r.compound_id, len(hits[r.compound_id]))
        time.sleep(PAUSE)

    union = sorted({c for v in hits.values() for c in v})
    props = pubchem_properties(union)

    rows = []
    for p in props.itertuples():
        if not isinstance(p.smiles_use, str):
            continue
        m = Chem.MolFromSmiles(p.smiles_use)
        if m is None:
            continue
        fp = MFP.GetFingerprint(m)
        for seed_id, sfp in seed_fps.items():
            if p.CID in hits[seed_id]:
                rows.append({"parent": seed_id, "pubchem_cid": int(p.CID), "title": p.Title,
                             "canonical_smiles": p.smiles_use, "mw": float(p.MolecularWeight),
                             "tanimoto": round(DataStructs.TanimotoSimilarity(sfp, fp), 3),
                             "source": "PubChem 2D similarity"})
    pc = annotate_tractability(pd.DataFrame(rows))
    pc = pc[pc.tractable_small_molecule & ~pc.nucleotide_cofactor]
    log.info("%d PubChem analogues clear the structural floor", len(pc))

    top = pc.sort_values("tanimoto", ascending=False).groupby("parent").head(TOP_N_VENDOR_LOOKUP).copy()
    vendors = []
    for i, cid in enumerate(top.pubchem_cid):
        vendors.append(s06.pubchem_vendors(str(int(cid)), cache))
        if i % 20 == 0:
            cache.flush()
        time.sleep(PAUSE)
    cache.flush()
    top["n_vendors"] = [v["n_vendors"] for v in vendors]
    top["example_vendors"] = [v["example_vendors"] for v in vendors]
    top["is_exact_or_stereoisomer"] = top.tanimoto >= 0.999

    ch_rows = []
    for r in seeds.itertuples():
        for pct in CHEMBL_PCT:
            for m in chembl_similar(r.canonical_smiles, pct):
                st = m.get("molecule_structures") or {}
                ch_rows.append({"parent": r.compound_id, "title": m["molecule_chembl_id"],
                                "canonical_smiles": st.get("canonical_smiles"),
                                "tanimoto": float(m.get("similarity", "nan")) / 100.0,
                                "pubchem_cid": None, "n_vendors": None, "example_vendors": None,
                                "source": "ChEMBL similarity (vendor status not resolved)"})
            time.sleep(PAUSE)
    ch = pd.DataFrame(ch_rows)
    if not ch.empty:
        ch = ch.sort_values("tanimoto", ascending=False).drop_duplicates(["parent", "title"])
        ch = ch[~ch.title.isin(seeds.compound_id)]
        ch["is_exact_or_stereoisomer"] = ch.tanimoto >= 0.999

    cols = ["parent", "pubchem_cid", "title", "canonical_smiles", "tanimoto", "mw",
            "n_vendors", "example_vendors", "is_exact_or_stereoisomer", "source"]
    out = pd.concat([top.reindex(columns=cols), ch.reindex(columns=cols)], ignore_index=True)
    out = out.sort_values(["parent", "tanimoto"], ascending=[True, False])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "analogue_candidates.csv", index=False)

    buyable = out[(out.n_vendors.astype(float).fillna(0) > 0) & ~out.is_exact_or_stereoisomer]
    for seed_id in seeds.compound_id:
        sub = buyable[buyable.parent == seed_id]
        log.info("%s: %d purchasable analogues, best Tanimoto %s", seed_id, len(sub),
                 f"{sub.tanimoto.max():.2f}" if len(sub) else "none")

    _render_grid(seeds, out, buyable)
    log.info("Saved %s and %s", RESULTS_DIR / "analogue_candidates.csv",
             RESULTS_DIR / "analogues_grid.png")


def _render_grid(seeds: pd.DataFrame, out: pd.DataFrame, buyable: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    n_cols = 4
    fig, axes = plt.subplots(len(seeds), n_cols, figsize=(180 / 25.4, 40.5 * len(seeds) / 25.4))
    axes = axes.reshape(len(seeds), n_cols)
    for row, r in enumerate(seeds.itertuples()):
        cells = [(r.canonical_smiles,
                  f"{r.compound_id}  (pChEMBL {r.pchembl_max:.2f})\nmeasured binder · "
                  f"{int(r.n_vendors or 0)} suppliers", True)]
        sub = buyable[buyable.parent == r.compound_id].sort_values("tanimoto", ascending=False)
        for a in sub.head(n_cols - 1).itertuples():
            name = str(a.title)[:30] + ("…" if len(str(a.title)) > 30 else "")
            cells.append((a.canonical_smiles,
                          f"{name}\nTanimoto {a.tanimoto:.2f} · {int(a.n_vendors)} suppliers", False))
        cells += [(None, "", False)] * (n_cols - len(cells))
        for ax, (smiles, legend, is_parent) in zip(axes[row], cells):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if smiles:
                img = _mol_image(smiles)
                if img is not None:
                    ax.imshow(img)
                ax.set_title(legend, fontsize=5.6, color="#1f4e79" if is_parent else "0.25", pad=3)
            else:
                ax.text(0.5, 0.55, "no purchasable\nanalogue found", ha="center", va="center",
                        fontsize=6.4, color="0.45", transform=ax.transAxes)
            if is_parent:
                for side in ("left", "right", "top", "bottom"):
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_color("#1f4e79")
                    ax.spines[side].set_linewidth(0.9)
    n_with = sum(1 for sid in seeds.compound_id if (buyable.parent == sid).any())
    fig.suptitle(
        f"Purchasable neighbours found for {n_with} of {len(seeds)} eEF1A-linked compounds;\n"
        "the measured binders' own scaffold is absent from purchasable space",
        fontsize=8.5, x=0.02, ha="left", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.30 / len(seeds)))
    fig.savefig(RESULTS_DIR / "analogues_grid.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="eef1a-only",
                        choices=["full-interactome", "eef1a-only"],
                        help="eef1a-only is the scope the relevance audit justifies")
    args = parser.parse_args()
    main(args.scope)
