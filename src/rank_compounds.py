"""Final ranked compound table: one row per compound, its best (lowest
Bonferroni p-value) metapath/target combination, sorted by significance.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PVALUES_DIR = Path("data/processed/pvalues")
RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")


def main(scope: str) -> None:
    pvals = pd.read_csv(PVALUES_DIR / f"pvalues.{scope}.tsv", sep="\t")
    compounds = pd.read_csv(RAW_DIR / f"compounds_chembl.{scope}.tsv", sep="\t")

    idx = pvals.groupby("compound_id")["p_value_bonferroni"].idxmin()
    best = pvals.loc[idx].reset_index(drop=True)
    best = best.merge(compounds, on="compound_id", how="left")
    best["rank"] = best["p_value_bonferroni"].rank(method="min").astype(int)
    best = best.sort_values("rank")

    out = best[
        ["rank", "compound_id", "pref_name", "target_gene", "metapath", "dwpc",
         "p_value", "p_value_bonferroni", "significant"]
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"hetnet_ranked_compounds.{scope}.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved %d ranked compounds to %s", len(out), out_path)
    log.info("Top 10:\n%s", out.head(10).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    args = parser.parse_args()
    main(args.scope)
