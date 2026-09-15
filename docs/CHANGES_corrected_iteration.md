# Corrected iteration — change log

Branch: `fix/dwpc-repeat-stitch-purchasable`, off `main` at `46807d3`.

This branch addresses four defects found by auditing the committed pipeline against its own data, and
adds two new layers (STITCH compound–protein evidence, and purchasability/SMILES annotation) so the
output is a buyable shortlist rather than an internal score table.

## Baseline

Before any change, `src/compute_all_dwpcs.py --scope full-interactome` was re-run on the committed
matrices and reproduces `data/processed/dwpc/dwpc_observed.full-interactome.tsv` **exactly** (3,057 rows,
max absolute difference 0.0). Every before/after number in this branch is therefore measured against a
verified baseline, not an approximation of it.

Row counts in the committed baseline: `CbG` 23, `CbGiG` 861, `CbGpBP` 217, `CbGpCC` 871, `CbGpMF` 868,
`CbGpPW` 217 — i.e. 2,173 rows on the four annotation metapaths and 3,057 rows in total.

## Corrections in this branch

| # | Defect | Fix | File |
|---|---|---|---|
| 1 | Annotation metapaths `C-b-G-p-X-p-G` score walks that return to their own source gene; for every compound binding its own reported target the self-return walk is 100% of the score | Exact simple-path correction: subtract the invalid `g == t` contribution; refuse to compute (rather than approximate) any repeat pattern not handled exactly | `src/dwpc.py`, `src/compute_all_dwpcs.py` |
| 2 | Every ChEMBL activity record becomes a binary edge regardless of measurement type, relation or units, so non-detects (`IC50 >= 10000 nM`) and unitless ratios create edges | Filter to binding-relevant records with an equality relation and convertible units; carry pKd as an edge weight | `src/s01_download_chembl_compounds.py`, `src/build_matrices.py` |
| 3 | Per-compound nulls are computed over 200 permutations and then discarded in favour of one pooled gamma fit per (metapath, target) | Use the per-compound null already stored in `null_summary.*.parquet` | `src/compute_pvalues.py` |
| 4 | Bonferroni applied within (metapath, target) and then the minimum taken across metapaths — selection on the statistic being tested; denominator counts only nonzero rows; no BH anywhere in the repo | Westfall–Young min-p permutation correction for the min-over-metapaths statistic, BH as the primary hit call, denominator over all compound slots | `src/compute_pvalues.py`, `src/rank_compounds.py` |

## New layers

- **STITCH** (`src/s05_download_stitch.py`): human protein–chemical links, restricted to the
  `experimental` and `database` channels. `textmining` and `prediction` are deliberately discarded —
  eEF1A's interactome is the literature that motivated this project, so those channels would score
  compounds on the co-occurrence that generated the hypothesis.
- **Purchasability and structure** (`src/s06_annotate_purchasable.py`): canonical SMILES from ChEMBL,
  PubChem CID via UniChem, vendor counts from PubChem chemical-vendor annotations.

## What this branch does not fix

Adding STITCH deepens rather than removes the central circularity: even its `database` channel is curated
from the same assay literature, so the ranking remains a re-ranker of an assay-defined candidate set, not
a prospective finder. The graph still encodes no binding site, no direction of effect and no potency at
which engagement occurs, and the 8 interdomain-pocket binders in the compendium remain ineligible until
complex- and family-level ChEMBL targets are admitted. See `hetnet_limitations.md` (§7) for the full list.
