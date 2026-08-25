"""Fit a gamma-hurdle null model to the permuted DWPC distribution, per
Himmelstein et al. 2023.

A gamma-hurdle model treats the null DWPC for a given (metapath,
degree-group) as: zero with probability (1-pi), and Gamma(alpha, beta)
distributed with probability pi when nonzero. This matches the fact that
most permuted networks have DWPC=0 for a given compound-target pair (no
path exists), while the nonzero values are continuous and right-skewed.

Degree grouping: the paper groups by (source_degree, target_degree) to
increase the effective permutation sample size per fit. This project's
network is small enough, and the target set is fixed (EEF1A1, EEF1A2 -
so target_degree is constant per metapath), that we group by
(metapath, target_gene) directly - the natural degree-group boundary
here, since every compound in a metapath shares the same target node.
This is documented as a simplification appropriate to this network's
scale (noted again in the final report).

Method-of-moments gamma fit (as in the paper, for speed/robustness over
MLE on possibly-small nonzero samples):
    mean = alpha / beta        =>  beta  = mean / var
    var  = alpha / beta^2      =>  alpha = mean^2 / var
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NULL_DIR = Path("data/processed/null_distribution")
GAMMA_DIR = Path("data/processed/gamma_hurdle")


def fit_gamma_hurdle_group(dwpc_values: np.ndarray, n_permutations: int) -> dict:
    """dwpc_values: the NONZERO null DWPC values observed within one
    (metapath, target_gene) group, pooled across compounds and permutations.
    n_permutations x n_compounds_in_group = total trials for that group.
    """
    n_nonzero = len(dwpc_values)
    n_total = n_permutations  # trials are already pooled per-compound by caller
    pi = n_nonzero / n_total if n_total > 0 else 0.0

    if n_nonzero < 2 or dwpc_values.std(ddof=1) == 0:
        # not enough variation to fit a gamma - fall back to a point mass at the mean
        mean = dwpc_values.mean() if n_nonzero > 0 else 0.0
        return {"pi": pi, "alpha": np.nan, "beta": np.nan, "mean": mean, "n_nonzero": n_nonzero, "fit_method": "degenerate"}

    mean = dwpc_values.mean()
    var = dwpc_values.var(ddof=1)
    alpha = mean ** 2 / var
    beta = mean / var
    return {"pi": pi, "alpha": alpha, "beta": beta, "mean": mean, "n_nonzero": n_nonzero, "fit_method": "method_of_moments"}


def main(scope: str) -> None:
    import glob

    files = sorted(glob.glob(str(NULL_DIR / f"perm_*.{scope}.parquet")))
    if not files:
        raise FileNotFoundError(f"No permutation files found for scope={scope} in {NULL_DIR}")
    n_permutations = len(files)
    log.info("Loading %d permutation files for scope=%s", n_permutations, scope)

    dfs = [pd.read_parquet(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)

    rows = []
    for (metapath, target_gene), grp in all_df.groupby(["metapath", "target_gene"]):
        # total trials for this group = n_permutations x number of DISTINCT compound
        # node ids that ever appeared nonzero anywhere for this (metapath,target) --
        # conservatively we instead treat one trial per (permutation, compound) pair
        # actually observed; pi is therefore relative to the union of compound slots seen.
        n_compound_slots = grp["compound_node_id"].nunique()
        n_trials = n_permutations * n_compound_slots
        fit = fit_gamma_hurdle_group(grp["dwpc"].values, n_trials)
        fit.update({"metapath": metapath, "target_gene": target_gene, "n_compound_slots": n_compound_slots})
        rows.append(fit)

    out = pd.DataFrame(rows)
    GAMMA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GAMMA_DIR / f"gamma_hurdle_fits.{scope}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    log.info("Saved %d gamma-hurdle fits to %s", len(out), out_path)
    log.info("\n%s", out.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    args = parser.parse_args()
    main(args.scope)
