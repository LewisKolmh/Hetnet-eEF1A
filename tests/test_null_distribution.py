import pandas as pd


def test_null_summary_exists_and_covers_all_metapaths():
    df = pd.read_parquet("data/processed/null_distribution/null_summary.full-interactome.parquet")
    assert len(df) > 0
    assert set(df["metapath"]) == {"CbG", "CbGiG", "CbGpBP", "CbGpMF", "CbGpCC", "CbGpPW"}
    assert (df["null_n_permutations"] == 200).all()
    assert (df["null_std"] >= 0).all()
    assert (df["null_mean"] >= 0).all()


def test_null_nonzero_count_never_exceeds_permutation_count():
    df = pd.read_parquet("data/processed/null_distribution/null_summary.full-interactome.parquet")
    assert (df["null_n_nonzero"] <= df["null_n_permutations"]).all()
