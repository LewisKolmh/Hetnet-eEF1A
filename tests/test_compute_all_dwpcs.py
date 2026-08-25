import pandas as pd


def test_observed_dwpc_nonempty_and_positive():
    df = pd.read_csv("data/processed/dwpc/dwpc_observed.full-interactome.tsv", sep="\t")
    assert len(df) > 0
    assert (df["dwpc"] > 0).all()
    assert set(df["metapath"]) == {"CbG", "CbGiG", "CbGpBP", "CbGpMF", "CbGpCC", "CbGpPW"}
    assert set(df["target_gene"]) <= {"EEF1A1", "EEF1A2"}


def test_direct_binders_appear_in_cbg_metapath():
    df = pd.read_csv("data/processed/dwpc/dwpc_observed.eef1a-only.tsv", sep="\t")
    cbg = df[df.metapath == "CbG"]
    compounds = pd.read_csv("data/raw/compounds_chembl.eef1a-only.tsv", sep="\t")
    cid = compounds[compounds.pref_name == "CYCLOHEXIMIDE"]["compound_id"].iloc[0]
    assert cid in cbg["compound_id"].values
