"""The four-panel summary of what this branch corrects (docs/fig_corrections.png).

The figure existed before this script did, which meant the repository carried a
headline figure nothing could regenerate. Each panel reads the committed tables:

a  edge admission - the activity-filter audit, one bar per rejection criterion
b  network rank against measured affinity, coloured by assay provenance grade
c  p-value calibration - uncalibrated best-cell minimum against the permutation
   -calibrated statistic, both as empirical CDFs against the uniform diagonal
d  the candidate funnel, applied in the order `rank_compounds` applies it

Panel d deliberately does NOT include purchasability. A compound with no
catalogued supplier can be synthesised, and on this graph the only compounds
carrying a targeted eEF1A binding measurement have no supplier at all, so
gating on catalogue presence would drop exactly the molecules worth making.
Supplier counts travel as a column in the tables and as an annotation here.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DOCS_DIR = Path("docs")

MM = 1 / 25.4
FOCAL = "#1f4e79"
GREY = "#9a9a9a"
META = "#7a7a7a"
GRADE_COLOR = {"direct_biophysical": FOCAL, "multiplexed_pulldown": GREY,
               "proteome_readout": "#b9b9b9", "assay_reported": "#cfcfcf",
               "stitch_only": "#e2e2e2"}
GRADE_LABEL = {"direct_biophysical": "named biophysical method (SPR)",
               "multiplexed_pulldown": "multiplexed pulldown",
               "proteome_readout": "proteome-scale readout",
               "assay_reported": "activity reported, readout unclear",
               "stitch_only": "STITCH interaction only"}


def _style() -> None:
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "figure.dpi": 150, "savefig.dpi": 300, "font.family": "sans-serif",
    })


def _panel_letter(ax, letter: str) -> None:
    ax.text(-0.18, 1.02, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=10, va="bottom", ha="left")


def funnel_steps(rk: pd.DataFrame) -> list[tuple[str, int]]:
    """The shortlist criteria in the order rank_compounds applies them."""
    sig = rk.significant_bh
    has_structure = sig & rk.canonical_smiles.notna()
    selective = has_structure & (rk.n_seed_targets <= 3)
    floor = selective & rk.tractable_small_molecule & ~rk.nucleotide_cofactor
    not_substrate = floor & ~rk.substrate_relation_only
    return [("all ranked", len(rk)),
            ("significant, q < 0.05", int(sig.sum())),
            ("+ structure resolved", int(has_structure.sum())),
            ("+ \u2264 3 seed targets", int(selective.sum())),
            ("+ structural floor", int(floor.sum())),
            ("+ not substrate", int(not_substrate.sum()))], not_substrate


def main(scope: str) -> None:
    rk = pd.read_csv(RESULTS_DIR / f"hetnet_ranked_compounds.{scope}.csv")
    audit = pd.read_csv(RAW_DIR / f"activity_filter_audit.{scope}.tsv", sep="\t")
    pv = pd.read_csv(PROCESSED_DIR / "pvalues" / f"pvalues.{scope}.tsv", sep="\t")
    _style()

    fig = plt.figure(figsize=(180 * MM, 138 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42,
                          left=0.075, right=0.975, top=0.91, bottom=0.10)

    # (a) edge admission
    axa = fig.add_subplot(gs[0, 0])
    # Plain-language restatement of each audit criterion, keyed on the exact
    # strings activity_filter writes. An unmapped criterion is a code change, so
    # fail loudly rather than printing an internal column name into a figure.
    criterion_label = {
        "affinity/potency measurement type": "not an affinity or\npotency measurement",
        "exact relation (=)": "censored value\n(> or <)",
        "molar-convertible (pchembl_value present)": "no concentration that\nconverts to molar",
        "binding or functional assay": "not a binding or\nfunctional assay",
        "no ChEMBL validity flag": "flagged invalid\nby ChEMBL",
        "pchembl_value >= 5.0": "weaker than 10 \u00b5M",
    }
    unmapped = set(audit.criterion) - set(criterion_label)
    if unmapped:
        raise SystemExit(f"unmapped audit criteria, add a plain-language label: {sorted(unmapped)}")
    lab = [criterion_label[c] for c in audit.criterion]
    val = list(audit.records_removed)
    admitted = int(audit.records_remaining.iloc[-1])
    y = np.arange(len(val))[::-1]
    axa.barh(y, val, color=META, height=0.62)
    axa.barh([-1], [admitted], color=FOCAL, height=0.62)
    for yy, v in zip(list(y) + [-1], val + [admitted]):
        # A criterion that removed nothing gets a visible stub, so "0" reads as a
        # measured zero rather than a missing bar.
        if v == 0:
            axa.plot([0, max(val) * 0.008], [yy, yy], color=META, lw=3.2,
                     solid_capstyle="butt")
        axa.text(v + max(val) * 0.025, yy, f"{v}", va="center", fontsize=6)
    axa.set_yticks(list(y) + [-1])
    axa.set_yticklabels(lab + ["admitted as\nbinding edges"], fontsize=6)
    axa.set_xlabel("ChEMBL activity records")
    axa.set_xlim(0, max(val) * 1.22)
    axa.set_title("Most activity records behind the baseline edges\nreport no measured affinity", loc="left")
    _panel_letter(axa, "a")

    # (b) network evidence against measured affinity
    axb = fig.add_subplot(gs[0, 1])
    sub = rk[rk.pchembl_max.notna()].copy()
    sub["y"] = -np.log10(sub.q_bh.clip(lower=1e-3))
    order = ["stitch_only", "assay_reported", "proteome_readout", "multiplexed_pulldown",
             "direct_biophysical"]
    for grade in order:
        s = sub[sub.evidence_grade == grade]
        if s.empty:
            continue
        axb.scatter(s.pchembl_max, s.y, s=26 if grade == "direct_biophysical" else 16,
                    facecolor=GRADE_COLOR[grade], edgecolor="white", linewidth=0.4,
                    zorder=4 if grade == "direct_biophysical" else 3,
                    label=GRADE_LABEL[grade])
    axb.axhline(-np.log10(0.05), color=META, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axb.text(sub.pchembl_max.max(), -np.log10(0.05) + 0.05, "q = 0.05",
             fontsize=6, color=META, ha="right")
    axb.set_xlabel("pChEMBL affinity (higher = better)")
    axb.set_ylabel("network evidence  $-\\log_{10}q$")
    axb.margins(x=0.09)
    axb.legend(frameon=False, loc="upper left", handletextpad=0.3, borderpad=0.1,
               labelspacing=0.3)
    axb.set_title("The strongest measured binders are not the\ncompounds the network ranks highest", loc="left")
    _panel_letter(axb, "b")

    # (c) calibration
    axc = fig.add_subplot(gs[1, 0])
    for col, color, label in [("p_cell_min", "#8fb4d9", "best cell, uncalibrated"),
                              ("p_minp", FOCAL, "permutation-calibrated")]:
        v = np.sort(pv[col].dropna().values)
        axc.step(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=1.3,
                 where="post", label=label)
    axc.plot([0, 1], [0, 1], color=META, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axc.text(0.42, 0.30, "uniform", fontsize=6, color=META, rotation=30,
             rotation_mode="anchor", va="top", ha="left")
    axc.set_xlabel("p-value")
    axc.set_ylabel("fraction of compounds")
    axc.set_xlim(0, 1); axc.set_ylim(0, 1.02)
    axc.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08),
               handletextpad=0.4, borderpad=0.1)
    axc.set_title("Taking each compound's best metapath inflates\nsignificance; calibration removes the excess", loc="left")
    _panel_letter(axc, "c")

    # (d) candidate funnel
    axd = fig.add_subplot(gs[1, 1])
    steps, survivor_mask = funnel_steps(rk)
    yy = np.arange(len(steps))[::-1]
    axd.barh(yy, [s[1] for s in steps],
             color=[META] * (len(steps) - 1) + [FOCAL], height=0.6)
    for y_, (_, v) in zip(yy, steps):
        axd.text(v + len(rk) * 0.02, y_, f"{v}", va="center", fontsize=6)
    axd.set_yticks(yy)
    axd.set_yticklabels([s[0] for s in steps], fontsize=6)
    axd.set_xlabel("compounds")
    axd.set_xlim(0, len(rk) * 1.21)
    survivors = rk[survivor_mask]
    n_sup = int(survivors.n_vendors.fillna(0).gt(0).sum())
    axd.set_title(f"{steps[-1][1]} compound survives every criterion;\n"
                  f"purchasability is reported, not required", loc="left")
    axd.text(len(rk) * 1.19, 0.75,
             f"{n_sup} of {len(survivors)} survivor(s)\nhas a catalogued supplier",
             fontsize=6, color=META, ha="right", va="bottom")
    _panel_letter(axd, "d")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "fig_corrections.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s | funnel: %s", out, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="full-interactome",
                        choices=["full-interactome", "eef1a-only"])
    args = parser.parse_args()
    main(args.scope)
