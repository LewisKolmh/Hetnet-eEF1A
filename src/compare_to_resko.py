"""Compare the Hetnet Connectivity Search ranking against the prior RESKO
(Ken McGarry-style, evidence/potency-scored) ranking for the same eEF1A
target set, and produce a data table + figure documenting where the two
approaches agree, disagree, and why - given they score compounds on
fundamentally different evidence.

RESKO (ranking_corrected.csv, from the earlier session) scores compounds
by DIRECT experimental evidence: literature-curated potency values,
relation-aware censoring (>=  / <=  qualifiers), and per-record evidence
weighting - it only ranks compounds for which an activity record already
exists. It answers: "of the compounds with recorded activity, which has
the strongest, most-trusted direct evidence of binding?"

Hetnet Connectivity Search (this pipeline) scores compounds by NETWORK
CONTEXT: whether a compound's target(s) sit unusually close to EEF1A1/
EEF1A2 in the interactome/GO/pathway graph, relative to a degree-matched
random null. It answers: "does this compound's known target sit in a
part of the network that is significantly enriched for connectivity to
eEF1A, more than chance given how connected that target already is?"
It can flag a compound with NO recorded eEF1A activity purely because its
target is embedded near eEF1A in the network - the entire point of using
a hetnet is to surface candidates RESKO's direct-evidence approach cannot
see by construction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PVALUES_DIR = Path("data/processed/pvalues")
RESULTS_DIR = Path("results")


def load_resko(resko_path: str) -> pd.DataFrame:
    df = pd.read_csv(resko_path)
    df = df.rename(columns={"molecule_chembl_id": "compound_id"})
    return df[["compound_id", "rank", "best_record_score", "total_potency", "max_evidence", "hits_eef1a", "best_protein"]]


def best_hetnet_row_per_compound(pvals: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per compound: its best (lowest) Bonferroni p-value
    across all target genes and metapaths, i.e. its strongest network-context signal."""
    idx = pvals.groupby("compound_id")["p_value_bonferroni"].idxmin()
    best = pvals.loc[idx].reset_index(drop=True)
    best["hetnet_rank"] = best["p_value_bonferroni"].rank(method="min").astype(int)
    return best.rename(
        columns={
            "metapath": "hetnet_best_metapath",
            "target_gene": "hetnet_best_target",
            "p_value_bonferroni": "hetnet_p_value_bonferroni",
            "dwpc": "hetnet_dwpc",
        }
    )[["compound_id", "hetnet_rank", "hetnet_best_metapath", "hetnet_best_target",
       "hetnet_dwpc", "hetnet_p_value_bonferroni", "significant"]]


def main(scope: str, resko_path: str) -> None:
    pvals = pd.read_csv(PVALUES_DIR / f"pvalues.{scope}.tsv", sep="\t")
    hetnet_best = best_hetnet_row_per_compound(pvals)
    resko = load_resko(resko_path)

    merged = hetnet_best.merge(resko, on="compound_id", how="outer", suffixes=("_hetnet", "_resko"))
    merged["in_resko_ranking"] = merged["rank"].notna()
    merged["in_hetnet_ranking"] = merged["hetnet_rank"].notna()
    merged["in_both"] = merged["in_resko_ranking"] & merged["in_hetnet_ranking"]
    merged = merged.sort_values(
        by=["in_both", "hetnet_p_value_bonferroni"], ascending=[False, True], na_position="last"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"hetnet_vs_resko_comparison.{scope}.csv"
    merged.to_csv(out_path, index=False)

    n_resko = merged["in_resko_ranking"].sum()
    n_hetnet = merged["in_hetnet_ranking"].sum()
    n_both = merged["in_both"].sum()
    log.info(
        "RESKO-ranked compounds: %d | Hetnet-ranked (any nonzero path) compounds: %d | overlap: %d",
        n_resko, n_hetnet, n_both,
    )
    log.info("Compounds in BOTH rankings:\n%s", merged[merged["in_both"]][
        ["compound_id", "rank", "best_record_score", "hetnet_rank", "hetnet_best_metapath",
         "hetnet_p_value_bonferroni", "significant"]
    ].to_string(index=False))
    log.info("Saved comparison table to %s", out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    parser.add_argument("--resko-path", required=True, help="Path to the RESKO ranking_corrected.csv")
    args = parser.parse_args()
    main(args.scope, args.resko_path)
