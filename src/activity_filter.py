"""Decide which ChEMBL activity records may become a Compound-binds-Gene edge.

Why this module exists
----------------------
The first iteration turned every activity record into an edge, deduplicating on
(compound, target) and keeping whichever record happened to come back first. The
committed `data/raw/compound_binds_gene.full-interactome.tsv` therefore opens with

    CHEMBL2110732  RAN  Kd  >  30000.0  nM
    CHEMBL3353410  RAN  Kd  >  30000.0  nM

These are *non-detections*: the experiment established that the compound does not
bind RAN at 30 uM. In the graph they are indistinguishable from a 1 nM Kd. Records
with no units at all (ratios, percent inhibition) were admitted on the same footing.

What is admitted
----------------
A record must clear all of:

1. an affinity or potency measurement type (`AFFINITY_TYPES`),
2. an exact relation (``=``) - so ``>``, ``>=``, ``<``, ``<=``, ``~`` are dropped,
3. a non-null `pchembl_value`. This is ChEMBL's own derived quantity and is
   populated only when the record is a molar-convertible affinity/potency with an
   equality relation, so it enforces unit sanity without this module maintaining a
   unit-conversion table,
4. an assay type in `ALLOWED_ASSAY_TYPES` (binding or functional; ADME,
   toxicity and physicochemical assays are not evidence of target engagement),
5. no `data_validity_comment` - ChEMBL flags suspect records here,
6. `pchembl_value >= min_pchembl` (default 5.0, i.e. 10 uM), the conventional
   activity cutoff for ChEMBL-derived interaction networks.

How potency is used
-------------------
The strongest surviving record per (compound, protein) supplies `pchembl_max`,
which is carried through to the ranked output. It is deliberately *not* used as a
matrix weight: DWPC's null here is a degree-preserving XSwap permutation, which is
defined over binary adjacency, so weighting the edges would leave the observed
statistic and its null measuring different things. Potency therefore travels as a
reported column alongside the score, keeping the network ranking and the potency
ranking separate - as the project's design notes require.
"""
from __future__ import annotations

import pandas as pd

# Measurement types that speak to target engagement. Functional readouts (EC50,
# AC50, Potency) are included because several eEF1A-relevant chemotypes are only
# ever reported functionally, but see the caveat in docs/CHANGES_corrected_iteration.md:
# a functional EC50 is engagement of a pathway, not proof of binding this protein.
AFFINITY_TYPES = frozenset({"Kd", "Ki", "IC50", "EC50", "AC50", "Potency"})

EXACT_RELATIONS = frozenset({"="})

# B = binding, F = functional. Excluded: A (ADME), T (toxicity), P (physicochemical), U (unassigned).
ALLOWED_ASSAY_TYPES = frozenset({"B", "F"})

DEFAULT_MIN_PCHEMBL = 5.0

REQUIRED_COLUMNS = (
    "compound_id",
    "protein_id",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_type",
    "data_validity_comment",
)


def filter_activities(
    df: pd.DataFrame, min_pchembl: float = DEFAULT_MIN_PCHEMBL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the admission criteria to raw activity records.

    Returns ``(kept, audit)`` where `audit` has one row per criterion with the
    number of records it removed, so the attrition is reportable rather than
    implicit.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"activity table is missing columns {missing}; re-run "
            "src/s01_download_chembl_compounds.py on this branch to capture them"
        )

    audit_rows = []
    cur = df.copy()
    cur["pchembl_value"] = pd.to_numeric(cur["pchembl_value"], errors="coerce")

    criteria = [
        ("affinity/potency measurement type", cur["standard_type"].isin(AFFINITY_TYPES)),
        ("exact relation (=)", cur["standard_relation"].isin(EXACT_RELATIONS)),
        ("molar-convertible (pchembl_value present)", cur["pchembl_value"].notna()),
        ("binding or functional assay", cur["assay_type"].isin(ALLOWED_ASSAY_TYPES)),
        ("no ChEMBL validity flag", cur["data_validity_comment"].isna()),
        (f"pchembl_value >= {min_pchembl}", cur["pchembl_value"] >= min_pchembl),
    ]
    mask = pd.Series(True, index=cur.index)
    for name, crit in criteria:
        crit = crit.fillna(False)
        removed = int((mask & ~crit).sum())
        mask &= crit
        audit_rows.append({"criterion": name, "records_removed": removed, "records_remaining": int(mask.sum())})

    kept = cur[mask].copy()
    audit = pd.DataFrame(audit_rows)
    audit.attrs["n_input"] = len(df)
    return kept, audit


def collapse_to_edges(kept: pd.DataFrame) -> pd.DataFrame:
    """One row per (compound, protein), carrying the strongest surviving record."""
    if kept.empty:
        return pd.DataFrame(
            columns=["compound_id", "protein_id", "target_chembl_id", "pchembl_max",
                     "n_records", "evidence_types"]
        )
    grouped = kept.sort_values("pchembl_value", ascending=False).groupby(
        ["compound_id", "protein_id"], as_index=False
    )
    edges = grouped.agg(
        target_chembl_id=("target_chembl_id", "first"),
        pchembl_max=("pchembl_value", "max"),
        n_records=("pchembl_value", "size"),
        evidence_types=("standard_type", lambda s: "|".join(sorted(set(s)))),
    )
    return edges
