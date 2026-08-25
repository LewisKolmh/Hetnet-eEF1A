"""Compute right-tail p-values for each observed compound-target DWPC
against its gamma-hurdle null model, with Bonferroni correction across
all compounds tested per (metapath, target_gene) pair - matching
Himmelstein et al.'s per-metanode-pair multiple-testing correction.

P(observed) = pi * P_Gamma(X > observed_dwpc | alpha, beta)   [if observed > 0]
            = 1                                                [if observed == 0]

(A compound with zero DWPC trivially cannot be significant.)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DWPC_DIR = Path("data/processed/dwpc")
GAMMA_DIR = Path("data/processed/gamma_hurdle")
PVALUES_DIR = Path("data/processed/pvalues")


def main(scope: str, alpha_threshold: float) -> None:
    observed = pd.read_csv(DWPC_DIR / f"dwpc_observed.{scope}.tsv", sep="\t")
    fits = pd.read_csv(GAMMA_DIR / f"gamma_hurdle_fits.{scope}.tsv", sep="\t")

    merged = observed.merge(fits, on=["metapath", "target_gene"], how="left", suffixes=("", "_fit"))

    def pvalue_row(row) -> float:
        if row["fit_method"] == "degenerate" or pd.isna(row["alpha"]) or pd.isna(row["beta"]):
            return float("nan")
        tail = stats.gamma.sf(row["dwpc"], a=row["alpha"], scale=1.0 / row["beta"])
        return float(row["pi"] * tail)

    merged["p_value"] = merged.apply(pvalue_row, axis=1)

    # Bonferroni correction: per (metapath, target_gene) group, multiply by the
    # number of compounds actually tested (i.e. rows in `observed`) for that group
    n_tests = merged.groupby(["metapath", "target_gene"])["compound_id"].transform("count")
    merged["p_value_bonferroni"] = (merged["p_value"] * n_tests).clip(upper=1.0)
    merged["significant"] = merged["p_value_bonferroni"] < alpha_threshold

    out = merged[
        ["compound_id", "target_gene", "metapath", "dwpc", "pi", "alpha", "beta",
         "p_value", "p_value_bonferroni", "significant"]
    ].sort_values("p_value_bonferroni")

    PVALUES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PVALUES_DIR / f"pvalues.{scope}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    log.info("Saved %d p-value rows to %s", len(out), out_path)
    log.info("Significant (Bonferroni p < %.3f): %d rows", alpha_threshold, out["significant"].sum())
    log.info("\n%s", out.head(15).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    parser.add_argument("--alpha", type=float, default=0.05, dest="alpha_threshold")
    args = parser.parse_args()
    main(args.scope, args.alpha_threshold)
