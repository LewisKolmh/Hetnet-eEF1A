import pandas as pd

from compute_all_dwpcs import METAPATHS


def test_observed_dwpc_nonempty_and_positive():
    df = pd.read_csv("data/processed/dwpc/dwpc_observed.full-interactome.tsv", sep="\t")
    assert len(df) > 0
    assert (df["dwpc"] > 0).all()
    assert set(df["metapath"]) <= set(METAPATHS)
    assert set(df["target_gene"]) <= {"EEF1A1", "EEF1A2"}


def test_stitch_metapaths_are_present_and_separate_from_binding():
    df = pd.read_csv("data/processed/dwpc/dwpc_observed.full-interactome.tsv", sep="\t")
    assert {"CsG", "CsGiG", "CsGpPW"} & set(df["metapath"]), "STITCH layer produced no paths"
    # the two layers must stay distinct metaedges, never merged into CbG
    assert not df["metapath"].str.startswith("CbGs").any()


def test_only_numerically_measured_affinities_become_direct_binder_edges():
    """Regression test for the non-detect edges in the previous iteration.

    The old suite asserted cycloheximide (CHEMBL123292) appears as a direct
    eEF1A1 binder. Its only ChEMBL records against any seed protein are a
    percent-inhibition row with no value and an 'Activity' row with no value or
    units, so it has no admissible affinity edge - and cycloheximide's actual
    site is the ribosomal E-site, not eEF1A1. Every CbG edge must now carry a
    numeric pchembl.
    """
    edges = pd.read_csv("data/raw/compound_binds_gene.full-interactome.tsv", sep="\t")
    assert edges["pchembl_max"].notna().all()
    assert (edges["pchembl_max"] >= 5.0).all()
    assert "CHEMBL123292" not in set(edges["compound_id"])

    df = pd.read_csv("data/processed/dwpc/dwpc_observed.full-interactome.tsv", sep="\t")
    cbg = set(df.loc[df.metapath == "CbG", "compound_id"])
    assert cbg <= set(edges["compound_id"])
