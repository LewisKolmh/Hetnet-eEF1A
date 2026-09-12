"""Per-compound permutation p-values, with the min-over-metapaths selection and
the across-compound multiplicity corrected separately.

What was wrong with the previous version
----------------------------------------
1. The permutations were computed per compound and then thrown away: a single
   gamma-hurdle was fitted per (metapath, target_gene) by pooling every
   compound's null draws, so a hub compound with 40 binding edges and a singleton
   were tested against the same null. The per-compound null is what the
   permutation run already produces, and it is what a DWPC null is supposed to be
   conditional on (source and target degree).

2. Bonferroni was applied *within* (metapath, target_gene), and then
   `rank_compounds.py` took each compound's minimum across metapaths and targets.
   Minimising a statistic over a family and then reporting it uncorrected for that
   minimisation is a selection effect: the resulting number is not a p-value for
   any hypothesis. With four highly correlated annotation metapaths and two
   targets, this is 8 correlated draws per compound.

3. The Bonferroni denominator counted only rows present in the observed DWPC file,
   which excludes zero-score rows - i.e. the multiplicity was estimated from the
   tests that happened to look promising.

What this version computes
--------------------------
Let R be the number of permutations, and index cells by compound c and
(metapath, target) pair k.

* **Cell-level empirical p** - ``p[c,k] = (1 + #{r: null[r,c,k] >= obs[c,k]}) / (1 + R)``.
  No distributional assumption; the per-compound null is used directly. The floor
  is 1/(R+1), so R sets the resolution (R=1000 gives 1e-3).

* **Per-compound min-p, calibrated by permutation** - the statistic actually used
  to rank a compound is ``min_k p[c,k]``. Its null distribution is obtained by
  applying the identical minimisation inside each permutation (cell p-values for
  permutation r computed leave-one-out against the other R-1 permutations), giving

      ``p_minp[c] = (1 + #{r: min_k p[r,c,k] <= min_k p[c,k]}) / (1 + R)``

  This is the standard min-p correction and it absorbs the correlation between
  metapaths rather than assuming independence.

* **Across-compound multiplicity** - Benjamini-Hochberg on ``p_minp`` over all
  compounds in the graph is the primary hit call (`significant_bh`), per the
  project's meeting decision. Two alternatives are reported alongside:
  Bonferroni over compounds (`significant_bonferroni`), and single-step
  Westfall-Young FWER (`p_westfall_young`), which uses the permutation
  distribution of the *global* minimum and so is both valid and less conservative
  than Bonferroni under the heavy correlation present here.

The gamma-hurdle p-value is retained as `p_gamma_pooled` for comparison with the
previous iteration, clearly labelled as pooled and not used for any hit call.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DWPC_DIR = Path("data/processed/dwpc")
GAMMA_DIR = Path("data/processed/gamma_hurdle")
NULL_DIR = Path("data/processed/null_distribution")
NODES_DIR = Path("data/processed/nodes")
PVALUES_DIR = Path("data/processed/pvalues")


def load_null_cube(scope: str, compound_node_ids: np.ndarray, cells: list[tuple[str, str]]):
    """Dense null array of shape (R, n_compounds, n_cells).

    The permutation files store only nonzero rows; absent entries are genuine
    zeros and are densified back in, because a zero draw is informative about the
    null (it is the least extreme possible value) and dropping it would bias every
    tail estimate upward.
    """
    row_of = {cid: i for i, cid in enumerate(compound_node_ids)}
    col_of = {cell: j for j, cell in enumerate(cells)}
    merged = NULL_DIR / f"null_draws.{scope}.parquet"
    if merged.exists():
        df = pd.read_parquet(merged)
        perms = np.sort(df["permutation"].unique())
        perm_of = {p: r for r, p in enumerate(perms)}
        cube = np.zeros((len(perms), len(compound_node_ids), len(cells)), dtype=np.float64)
        ri = df["permutation"].map(perm_of)
        ci = df["compound_node_id"].map(row_of)
        ki = pd.Series(list(zip(df["metapath"], df["target_gene"]))).map(col_of)
        keep = (ri.notna() & ci.notna() & ki.notna()).to_numpy()
        cube[ri[keep].astype(int).to_numpy(), ci[keep].astype(int).to_numpy(),
             ki[keep].astype(int).to_numpy()] = df.loc[keep, "dwpc"].to_numpy()
        log.info("Loaded null cube: %d permutations x %d compounds x %d (metapath,target) cells",
                 *cube.shape)
        return cube

    paths = sorted(NULL_DIR.glob(f"perm_*.{scope}.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no permutation files for scope={scope} in {NULL_DIR}; run "
            "src/compute_null_distribution.py first"
        )
    cube = np.zeros((len(paths), len(compound_node_ids), len(cells)), dtype=np.float64)
    for r, path in enumerate(paths):
        df = pd.read_parquet(path)
        ci = df["compound_node_id"].map(row_of)
        ki = pd.Series(list(zip(df["metapath"], df["target_gene"]))).map(col_of)
        keep = ci.notna() & ki.notna()
        cube[r, ci[keep].astype(int).to_numpy(), ki[keep].astype(int).to_numpy()] = (
            df.loc[keep.to_numpy(), "dwpc"].to_numpy()
        )
    log.info("Loaded null cube: %d permutations x %d compounds x %d (metapath,target) cells",
             *cube.shape)
    return cube


def empirical_tail_p(obs: np.ndarray, cube: np.ndarray) -> np.ndarray:
    """(1 + #{null >= obs}) / (1 + R), elementwise over cells."""
    n_ge = (cube >= obs[None, :, :]).sum(axis=0)
    return (1.0 + n_ge) / (1.0 + cube.shape[0])


def permutation_cell_p(cube: np.ndarray) -> np.ndarray:
    """Leave-one-out cell p-values for each permutation draw, same shape as `cube`.

    For draw r: (1 + #{s != r: null[s] >= null[r]}) / (1 + (R-1)). Ties count as
    >=, so an all-zero cell gets p = 1 in every permutation, as it should.
    """
    n_perm = cube.shape[0]
    out = np.empty_like(cube)
    # One cell column at a time: rankdata allocates several arrays the size of its
    # input, and the full cube can be hundreds of MB once STITCH widens the graph.
    for j in range(cube.shape[2]):
        # rankdata 'min': 1 + #{strictly smaller}; so #{>=} including self = R - rank + 1
        rank_min = stats.rankdata(cube[:, :, j], method="min", axis=0)
        out[:, :, j] = (1.0 + (n_perm - rank_min)) / float(n_perm)
    return out


def main(scope: str, alpha_threshold: float) -> None:
    observed = pd.read_csv(DWPC_DIR / f"dwpc_observed.{scope}.tsv", sep="\t")
    nodes = pd.read_csv(NODES_DIR / f"nodes.{scope}.tsv", sep="\t")
    compounds = nodes[nodes.metanode_type == "Compound"][["node_id", "external_id"]]
    compound_node_ids = compounds["node_id"].to_numpy()
    ext_by_node = dict(zip(compounds["node_id"], compounds["external_id"]))
    node_by_ext = {v: k for k, v in ext_by_node.items()}

    cells = sorted({(m, t) for m, t in zip(observed["metapath"], observed["target_gene"])})
    log.info("Testing %d compounds x %d (metapath,target) cells = %d cell-level tests",
             len(compound_node_ids), len(cells), len(compound_node_ids) * len(cells))

    # Dense observed matrix over the SAME grid as the null: every compound in the
    # graph occupies a slot, whether or not it scored above zero. This is the
    # multiple-testing denominator the previous version was missing.
    obs = np.zeros((len(compound_node_ids), len(cells)), dtype=np.float64)
    row_of = {cid: i for i, cid in enumerate(compound_node_ids)}
    col_of = {cell: j for j, cell in enumerate(cells)}
    n_unmapped = 0
    for cid, tgt, mp, val in zip(observed["compound_id"], observed["target_gene"],
                                 observed["metapath"], observed["dwpc"]):
        i = row_of.get(node_by_ext.get(cid, -1))
        j = col_of.get((mp, tgt))
        if i is None or j is None:
            n_unmapped += 1
            continue
        obs[i, j] = val
    if n_unmapped:
        log.warning("%d observed rows could not be placed on the compound x cell grid", n_unmapped)

    cube = load_null_cube(scope, compound_node_ids, cells)
    n_perm = cube.shape[0]

    p_cell = empirical_tail_p(obs, cube)
    minp_obs = p_cell.min(axis=1)
    best_cell = p_cell.argmin(axis=1)

    p_cell_perm = permutation_cell_p(cube)
    minp_perm = p_cell_perm.min(axis=2)                       # (R, n_compounds)
    p_minp = (1.0 + (minp_perm <= minp_obs[None, :]).sum(axis=0)) / (1.0 + n_perm)

    # Single-step Westfall-Young: null of the GLOBAL minimum across compounds.
    global_min_perm = minp_perm.min(axis=1)                   # (R,)
    p_wy = (1.0 + (global_min_perm[:, None] <= minp_obs[None, :]).sum(axis=0)) / (1.0 + n_perm)
    p_wy = p_wy.ravel()

    n_compounds = len(compound_node_ids)
    q_bh = stats.false_discovery_control(p_minp, method="bh")
    p_bonf = np.clip(p_minp * n_compounds, 0.0, 1.0)

    out = pd.DataFrame({
        "compound_id": [ext_by_node[c] for c in compound_node_ids],
        "best_metapath": [cells[j][0] for j in best_cell],
        "best_target_gene": [cells[j][1] for j in best_cell],
        "dwpc_at_best": obs[np.arange(n_compounds), best_cell],
        "p_cell_min": minp_obs,
        "p_minp": p_minp,
        "q_bh": q_bh,
        "p_bonferroni": p_bonf,
        "p_westfall_young": p_wy,
        "n_permutations": n_perm,
    })
    out["significant_bh"] = out["q_bh"] < alpha_threshold
    out["significant_bonferroni"] = out["p_bonferroni"] < alpha_threshold
    out["significant_wy"] = out["p_westfall_young"] < alpha_threshold

    # Pooled gamma-hurdle p-value, for comparison with the previous iteration only.
    fits_path = GAMMA_DIR / f"gamma_hurdle_fits.{scope}.tsv"
    cell_rows = pd.DataFrame(
        [{"compound_id": ext_by_node[compound_node_ids[i]], "metapath": cells[j][0],
          "target_gene": cells[j][1], "dwpc": obs[i, j], "p_cell": p_cell[i, j]}
         for i in range(n_compounds) for j in range(len(cells))]
    )
    if fits_path.exists():
        fits = pd.read_csv(fits_path, sep="\t")
        cell_rows = cell_rows.merge(fits, on=["metapath", "target_gene"], how="left")
        ok = cell_rows["alpha"].notna() & cell_rows["beta"].notna() & (cell_rows["dwpc"] > 0)
        cell_rows["p_gamma_pooled"] = np.nan
        cell_rows.loc[ok, "p_gamma_pooled"] = (
            cell_rows.loc[ok, "pi"]
            * stats.gamma.sf(cell_rows.loc[ok, "dwpc"], a=cell_rows.loc[ok, "alpha"],
                             scale=1.0 / cell_rows.loc[ok, "beta"])
        )

    PVALUES_DIR.mkdir(parents=True, exist_ok=True)
    cell_path = PVALUES_DIR / f"pvalues_cells.{scope}.tsv"
    cell_rows.to_csv(cell_path, sep="\t", index=False)
    out_path = PVALUES_DIR / f"pvalues.{scope}.tsv"
    out.sort_values("p_minp").to_csv(out_path, sep="\t", index=False)

    log.info("Saved %d cell-level rows to %s", len(cell_rows), cell_path)
    log.info("Saved %d per-compound rows to %s", len(out), out_path)
    log.info("Permutation resolution floor: p >= %.2e", 1.0 / (1.0 + n_perm))
    log.info("Hits at alpha=%.3f - BH: %d | Bonferroni: %d | Westfall-Young: %d",
             alpha_threshold, int(out["significant_bh"].sum()),
             int(out["significant_bonferroni"].sum()), int(out["significant_wy"].sum()))
    log.info("\n%s", out.sort_values("p_minp").head(15).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["eef1a-only", "full-interactome"], default="full-interactome")
    parser.add_argument("--alpha", type=float, default=0.05, dest="alpha_threshold")
    args = parser.parse_args()
    main(args.scope, args.alpha_threshold)
