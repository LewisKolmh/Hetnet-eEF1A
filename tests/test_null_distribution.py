import pandas as pd

from compute_all_dwpcs import METAPATHS

NULL_DRAWS = "data/processed/null_distribution/null_draws.full-interactome.parquet"


def test_null_draws_exist_and_cover_the_computed_metapaths():
    """The consolidated null replaces the old per-permutation null_summary.

    null_summary.<scope>.parquet was a summary of the previous iteration's null
    (200 permutations, 6 metapaths, uncorrected DWPC) and is not written by any
    script on this branch; the p-value code reads the draws themselves.
    """
    df = pd.read_parquet(NULL_DRAWS)
    assert len(df) > 0
    assert set(df["metapath"]) <= set(METAPATHS)
    assert {"CsGiG", "CsGpPW"} & set(df["metapath"]), "STITCH metapaths must be permuted too"
    assert (df["dwpc"] > 0).all(), "only nonzero draws are stored; zeros are densified on load"


def test_permutation_count_matches_the_resolution_the_pvalues_claim():
    draws = pd.read_parquet(NULL_DRAWS, columns=["permutation"])
    n_perm = draws["permutation"].nunique()
    pvals = pd.read_csv("data/processed/pvalues/pvalues.full-interactome.tsv", sep="\t")
    assert (pvals["n_permutations"] == n_perm).all()
    # no p-value may be finer than the permutation resolution floor
    assert pvals["p_minp"].min() >= 1.0 / (n_perm + 1) - 1e-12


def test_no_compound_appears_twice_within_one_permutation_cell():
    df = pd.read_parquet(NULL_DRAWS)
    assert not df.duplicated(["permutation", "compound_node_id", "metapath", "target_gene"]).any()
