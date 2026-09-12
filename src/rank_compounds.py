"""Final ranked candidate table: one row per compound, with the network evidence,
the chemistry evidence and the purchasability evidence in separate columns.

Deliberately NOT a composite score
----------------------------------
The connectivity statistic and the medicinal-chemistry properties answer different
questions - "is this compound unusually well connected to eEF1A in the network"
versus "is this a compound worth putting in an assay" - and they are not
commensurable. Blending them into one number would let a high DWPC rank
compensate for a 6-log P natural product, which is exactly the trade a human
should be making explicitly. So the table carries:

* `rank_network`   - by the corrected per-compound permutation p-value (`p_minp`).
* `rank_chemistry` - by ChEMBL pchembl_max (potency), among compounds that have a
  measured affinity at all; blank for STITCH-only compounds, because no potency
  was measured against a seed protein.
* evidence columns - which layer the edge came from, on how many seed proteins,
  and the discarded-channel values for STITCH edges.
* purchasability   - `n_vendors`, `example_vendors`, `canonical_smiles`,
  `inchikey`, `pubchem_cid`.

Substrate relations
-------------------
Three of the eighteen seed proteins are enzymes with small-molecule catalytic
sites (PAPSS1, PAPSS2, ST6GALNAC1) and a fourth, EEF1G, carries a GST-like
domain. STITCH records an enzyme's own substrates and products as associations,
so those seeds pull in metabolites and cofactors - PAPS, CMP-sialic acid,
glutathione conjugates - which are purchasable, well-connected, and useless as
allosteric modulators of eEF1A. `data/raw/seed_protein_class.tsv` marks those
seeds (`substrate_binding_domain`); `substrate_relation_only` marks compounds
whose entire seed evidence comes from them. They stay in the full table and are
written to their own file, but are held out of the purchasable shortlist, because
a shortlist is a list someone orders from.

Promiscuity is judged across scopes
-----------------------------------
The evidence columns are always read from the `full-interactome` edge tables,
even when ranking the `eef1a-only` scope. A compound with a measured Kd against
eight of the eighteen seed proteins looks selective inside a two-protein scope,
and the narrow scope would otherwise launder it onto the shortlist.

`n_seed_targets` is a promiscuity flag, not a merit: a compound with edges to a
dozen seed proteins is usually a reactive or assay-interfering chemotype rather
than a selective binder, and it will also score highly on any connectivity
statistic for structural reasons. It is reported so it can be used to exclude.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from chem_tractability import annotate_tractability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PVALUES_DIR = Path("data/processed/pvalues")
PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")


def main(scope: str) -> None:
    pvals = pd.read_csv(PVALUES_DIR / f"pvalues.{scope}.tsv", sep="\t")

    ann_path = PROCESSED_DIR / f"compound_annotation.{scope}.tsv"
    if ann_path.exists():
        ann = pd.read_csv(ann_path, sep="\t")
    else:
        log.warning("No annotation file at %s - run src/s06_annotate_purchasable.py; "
                    "SMILES and vendor columns will be absent", ann_path)
        ann = pd.DataFrame(columns=["compound_id"])

    # ChEMBL evidence: strongest measured affinity and how many seed proteins it
    # was measured against.
    # Evidence tables: always full-interactome, see "Promiscuity" in the docstring.
    cbg_path = RAW_DIR / "compound_binds_gene.full-interactome.tsv"
    cbg = pd.read_csv(cbg_path, sep="\t") if cbg_path.exists() else pd.DataFrame()
    if not cbg.empty:
        chembl_ev = cbg.groupby("compound_id").agg(
            pchembl_max=("pchembl_max", "max"),
            n_chembl_targets=("protein_id", "nunique"),
            chembl_targets=("protein_id", lambda s: "|".join(sorted(set(s)))),
            evidence_types=("evidence_types", lambda s: "|".join(sorted({t for v in s for t in str(v).split("|")}))),
        ).reset_index()
    else:
        chembl_ev = pd.DataFrame(columns=["compound_id", "pchembl_max", "n_chembl_targets",
                                          "chembl_targets", "evidence_types"])

    # STITCH evidence, with the discarded channels kept visible.
    csg_path = RAW_DIR / "compound_stitch_gene.full-interactome.tsv"
    csg = pd.read_csv(csg_path, sep="\t") if csg_path.exists() else pd.DataFrame()
    if not csg.empty:
        stitch_ev = csg.groupby("compound_id").agg(
            stitch_score_max=("channel_score", "max"),
            stitch_experimental_max=("experimental", "max"),
            stitch_database_max=("database", "max"),
            stitch_textmining_max_discarded=("textmining_discarded", "max"),
            n_stitch_targets=("protein_id", "nunique"),
            stitch_targets=("protein_id", lambda s: "|".join(sorted(set(s)))),
        ).reset_index()
    else:
        stitch_ev = pd.DataFrame(columns=["compound_id", "stitch_score_max", "n_stitch_targets"])

    out = (pvals
           .merge(ann, on="compound_id", how="left")
           .merge(chembl_ev, on="compound_id", how="left")
           .merge(stitch_ev, on="compound_id", how="left"))

    out["n_seed_targets"] = (out.get("n_chembl_targets").fillna(0)
                             + out.get("n_stitch_targets").fillna(0)).astype(int)
    out["edge_layers"] = [
        "|".join(x for x in (
            "ChEMBL" if pd.notna(c) and c > 0 else "",
            "STITCH" if pd.notna(s) and s > 0 else "",
        ) if x)
        for c, s in zip(out.get("n_chembl_targets"), out.get("n_stitch_targets"))
    ]

    # Seed-protein classification: which seeds a compound's evidence rests on.
    seed_class = pd.read_csv(RAW_DIR / "seed_protein_class.tsv", sep="\t")
    substrate_seeds = set(seed_class.loc[seed_class["substrate_binding_domain"], "protein_id"])
    eef1a_seeds = set(seed_class.loc[seed_class["seed_class"] == "eef1a_paralogue", "protein_id"])

    def _targets(row) -> set[str]:
        got = set()
        for col in ("chembl_targets", "stitch_targets"):
            v = row.get(col)
            if isinstance(v, str) and v:
                got.update(v.split("|"))
        return got

    tgt_sets = [_targets(r) for _, r in out.iterrows()]
    out["seed_targets"] = ["|".join(sorted(t)) for t in tgt_sets]
    out["binds_eef1a_directly"] = [bool(t & eef1a_seeds) for t in tgt_sets]
    out["substrate_relation_only"] = [
        bool(t) and t.issubset(substrate_seeds) for t in tgt_sets
    ]

    # Assay provenance (src/s07_assay_provenance.py): what kind of experiment
    # backs each ChEMBL edge. STITCH-only compounds have no ChEMBL assay, so they
    # are labelled 'stitch_only' rather than left blank.
    GRADE_ORDER = ["direct_biophysical", "assay_reported", "proteome_readout",
                   "multiplexed_pulldown"]
    prov_path = PROCESSED_DIR / f"assay_provenance.{scope}.tsv"
    if prov_path.exists():
        prov = pd.read_csv(prov_path, sep="\t")
        prov["_rank"] = prov["evidence_grade"].map({g: i for i, g in enumerate(GRADE_ORDER)})
        best = (prov.sort_values("_rank")
                    .groupby("compound_id", as_index=False)
                    .agg(evidence_grade=("evidence_grade", "first"),
                         assay_protein=("protein_id", "first"),
                         n_supporting_assays=("n_supporting_assays", "max"),
                         max_seed_proteins_in_one_document=("max_seed_proteins_in_one_document", "max"),
                         example_assay_description=("example_assay_description", "first")))
        out = out.merge(best, on="compound_id", how="left")
    else:
        log.warning("no %s; run s07_assay_provenance.py to grade ChEMBL evidence", prov_path)
        for c in ("evidence_grade", "assay_protein", "n_supporting_assays",
                  "max_seed_proteins_in_one_document", "example_assay_description"):
            out[c] = pd.NA
    out["evidence_grade"] = out["evidence_grade"].fillna("stitch_only")

    # Structural tractability: ions, buffer components and small metabolites reach
    # the top of the STITCH-driven ranking on connectivity alone. See
    # src/chem_tractability.py for the criteria and why they are flags, not deletions.
    out = annotate_tractability(out)  # attaches structural_exclusion_reason too
    out["rank_network"] = out["p_minp"].rank(method="min").astype(int)
    out["rank_chemistry"] = out["pchembl_max"].rank(method="min", ascending=False)
    out = out.sort_values(["rank_network", "compound_id"])

    cols = ["rank_network", "rank_chemistry", "compound_id", "pref_name",
            "best_metapath", "best_target_gene", "dwpc_at_best",
            "p_cell_min", "p_minp", "q_bh", "p_bonferroni", "p_westfall_young",
            "significant_bh", "significant_bonferroni", "significant_wy",
            "edge_layers", "n_seed_targets", "pchembl_max", "evidence_types",
            "chembl_targets", "stitch_score_max", "stitch_experimental_max",
            "stitch_database_max", "stitch_textmining_max_discarded", "stitch_targets",
            "binds_eef1a_directly", "substrate_relation_only", "seed_targets",
            "evidence_grade", "assay_protein", "n_supporting_assays",
            "max_seed_proteins_in_one_document", "example_assay_description",
            "tractable_small_molecule", "nucleotide_cofactor",
            "structural_exclusion_reason", "n_heavy_atoms", "n_rings",
            "max_phase", "alogp", "mw", "n_vendors", "example_vendors",
            "canonical_smiles", "inchikey", "pubchem_cid", "structure_source"]
    cols = [c for c in cols if c in out.columns]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"hetnet_ranked_compounds.{scope}.csv"
    out[cols].to_csv(out_path, index=False)
    log.info("Saved %d ranked compounds to %s", len(out), out_path)

    # Purchasable shortlist: BH-significant, has a structure, has at least one
    # supplier, and is not a many-target chemotype.
    orderable = (out["significant_bh"]
                 & out["canonical_smiles"].notna()
                 & (out.get("n_vendors", 0).fillna(0) > 0)
                 & (out["n_seed_targets"] <= 3))
    chemically_sensible = out["tractable_small_molecule"] & ~out["nucleotide_cofactor"]
    shortlist = out[orderable & chemically_sensible & ~out["substrate_relation_only"]]
    held_back = out[orderable & ~(chemically_sensible & ~out["substrate_relation_only"])]
    held_path = RESULTS_DIR / f"substrate_relation_hits.{scope}.csv"
    held_back[cols].to_csv(held_path, index=False)
    log.info(
        "Held back %d significant, purchasable hits that are substrate/cofactor "
        "relations or fail the structural floor -> %s", len(held_back), held_path,
    )
    short_path = RESULTS_DIR / f"purchasable_shortlist.{scope}.csv"
    shortlist[cols].to_csv(short_path, index=False)

    # Lead candidates: the purchasable shortlist plus every compound whose link to
    # a seed protein rests on a named biophysical binding method, whether or not a
    # supplier was found. The second group is where the medicinal chemistry is, and
    # dropping it for want of a catalogue entry would hide the best evidence on the
    # graph; `purchasable` is a column so the distinction stays visible.
    # The structural floor applies here too: a 2.3 kDa macrocyclic peptide with an
    # HSF1 fluorescence-polarization Kd is a real measurement but not a candidate
    # small-molecule allosteric inhibitor for this project.
    biophysical = (out["evidence_grade"] == "direct_biophysical") & out["tractable_small_molecule"]
    lead_mask = biophysical | out.index.isin(shortlist.index)
    leads = out[lead_mask].copy()
    leads["selection_reason"] = [
        "; ".join(filter(None, [
            "purchasable shortlist" if i in set(shortlist.index) else "",
            "direct biophysical binding evidence" if b else "",
        ]))
        for i, b in zip(leads.index, leads["evidence_grade"] == "direct_biophysical")
    ]
    leads["purchasable"] = leads["n_vendors"].fillna(0) > 0
    lead_cols = ["selection_reason", "purchasable"] + cols
    lead_path = RESULTS_DIR / f"lead_candidates.{scope}.csv"
    leads.sort_values(["rank_network", "compound_id"])[lead_cols].to_csv(lead_path, index=False)
    log.info("Lead candidates: %d compounds (%d purchasable) -> %s",
             len(leads), int(leads["purchasable"].sum()), lead_path)
    log.info("Purchasable shortlist: %d compounds -> %s", len(shortlist), short_path)
    if not shortlist.empty:
        log.info("\n%s", shortlist[["rank_network", "compound_id", "pref_name", "q_bh",
                                    "n_seed_targets", "n_vendors"]].head(15).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    args = parser.parse_args()
    main(args.scope)
