"""Tests for the structural tractability floor (src/chem_tractability.py).

The floor exists because the STITCH layer contributes ions, buffer components and
endogenous metabolites that score well on any connectivity statistic - chloride
and magnesium bind everything, glycerol is a cryoprotectant, GDP is eEF1A's own
cofactor - and none of them is a candidate allosteric inhibitor. These tests pin
the two properties that matter: real drug-like compounds pass, and the species
that motivated the floor fail.
"""
import pandas as pd
import pytest

from chem_tractability import annotate_tractability, exclusion_reason

# (label, SMILES, should_pass)
CASES = [
    # Drug-like small molecules: must pass.
    # SMILES copied verbatim from data/processed/compound_annotation.*.tsv.
    ("molibresib", "CCNC(=O)C[C@@H]1N=C(c2ccc(Cl)cc2)c2cc(OC)ccc2-n2c(C)nnc21", True),
    ("flavonoid analogue CHEMBL1802815",
     "CC(C)CNc1nc2cc3c(=O)cc(-c4ccc(C(=O)N[C@@H](Cc5ccccc5)C(N)=O)cc4)oc3cc2n1Cc1ccccc1", True),
    # Real drug, but below the lead-likeness floor: the floor is a candidate
    # filter, not a druglikeness test.
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", False),
    # Ions and small inorganics arriving through STITCH: must fail.
    ("chloride", "[Cl-]", False),
    ("magnesium", "[Mg+2]", False),
    ("water", "O", False),
    # Organic but not a candidate scaffold: no ring, too small.
    ("glycerol", "OCC(O)CO", False),
    ("acetate", "CC(=O)[O-]", False),
    # Metal-containing therapeutic: real drug, but not an allosteric small molecule.
    ("arsenic trioxide", "O=[As]O[As]=O", False),
]


@pytest.mark.parametrize("label,smiles,should_pass", CASES)
def test_structural_floor(label, smiles, should_pass):
    df = pd.DataFrame({"compound_id": [label], "canonical_smiles": [smiles]})
    out = annotate_tractability(df)
    assert bool(out["tractable_small_molecule"].iloc[0]) is should_pass, (
        f"{label}: {out['structural_exclusion_reason'].iloc[0]}"
    )


def test_excluded_rows_carry_a_reason_and_passing_rows_do_not():
    df = pd.DataFrame({
        "compound_id": [c[0] for c in CASES],
        "canonical_smiles": [c[1] for c in CASES],
    })
    out = annotate_tractability(df)
    failed = out[~out["tractable_small_molecule"]]
    passed = out[out["tractable_small_molecule"]]
    assert (failed["structural_exclusion_reason"].str.len() > 0).all(), (
        "every excluded compound must say why it was excluded"
    )
    assert (passed["structural_exclusion_reason"] == "").all(), (
        "a compound that passes the floor must carry no exclusion reason"
    )


def test_nucleotide_cofactor_flagged_separately_from_the_floor():
    """GDP is eEF1A's own cofactor. It is large enough to clear the size floor, so
    the floor alone cannot hold it back - it needs its own flag."""
    gdp = "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1"
    out = annotate_tractability(pd.DataFrame(
        {"compound_id": ["GDP"], "canonical_smiles": [gdp]}))
    assert bool(out["nucleotide_cofactor"].iloc[0]), (
        "GDP must be flagged as a nucleotide cofactor"
    )


def test_unparseable_smiles_fails_closed():
    out = annotate_tractability(pd.DataFrame(
        {"compound_id": ["junk"], "canonical_smiles": ["not-a-smiles"]}))
    assert not bool(out["tractable_small_molecule"].iloc[0])
    assert out["structural_exclusion_reason"].iloc[0]


def test_missing_structure_fails_closed():
    out = annotate_tractability(pd.DataFrame(
        {"compound_id": ["no-structure"], "canonical_smiles": [None]}))
    assert not bool(out["tractable_small_molecule"].iloc[0])


def test_annotation_preserves_row_order_on_duplicate_index_labels():
    """rank_compounds passes a frame whose index has duplicate labels after its
    merges; descriptors must align positionally, not by label."""
    df = pd.DataFrame(
        {"compound_id": ["chloride", "molibresib"],
         "canonical_smiles": [
             "[Cl-]",
             "CCNC(=O)C[C@@H]1N=C(c2ccc(Cl)cc2)c2cc(OC)ccc2-n2c(C)nnc21"]},
        index=[0, 0],
    )
    out = annotate_tractability(df)
    assert list(out["compound_id"]) == ["chloride", "molibresib"]
    assert list(out["tractable_small_molecule"]) == [False, True]


def test_exclusion_reason_names_the_failing_criterion():
    row = {"has_carbon": False, "has_metal": False, "mw": 35.0,
           "n_rings": 0, "n_heavy_atoms": 1}
    reason = exclusion_reason(row)
    assert "carbon" in reason.lower()
