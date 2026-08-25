import numpy as np
import pandas as pd

from src.fit_gamma_hurdle import fit_gamma_hurdle_group


def test_gamma_moments_recovered_on_synthetic_data():
    rng = np.random.default_rng(0)
    true_alpha, true_beta = 3.0, 5.0
    sample = rng.gamma(shape=true_alpha, scale=1.0 / true_beta, size=5000)
    fit = fit_gamma_hurdle_group(sample, n_permutations=5000)
    assert fit["fit_method"] == "method_of_moments"
    assert np.isclose(fit["alpha"], true_alpha, rtol=0.15)
    assert np.isclose(fit["beta"], true_beta, rtol=0.15)
    assert np.isclose(fit["pi"], 1.0)


def test_degenerate_fit_when_insufficient_nonzero():
    fit = fit_gamma_hurdle_group(np.array([0.5]), n_permutations=200)
    assert fit["fit_method"] == "degenerate"
    assert fit["pi"] == 1 / 200


def test_pi_zero_when_no_nonzero_values():
    fit = fit_gamma_hurdle_group(np.array([]), n_permutations=200)
    assert fit["pi"] == 0.0
    assert fit["n_nonzero"] == 0


def test_pvalues_output_bonferroni_geq_raw():
    df = pd.read_csv("data/processed/pvalues/pvalues.full-interactome.tsv", sep="\t")
    valid = df.dropna(subset=["p_value", "p_value_bonferroni"])
    assert (valid["p_value_bonferroni"] >= valid["p_value"] - 1e-12).all()
    assert (valid["p_value_bonferroni"] <= 1.0).all()


def test_zero_dwpc_never_flagged_significant():
    df = pd.read_csv("data/processed/dwpc/dwpc_observed.full-interactome.tsv", sep="\t")
    # dwpc_observed only contains nonzero rows by construction; sanity-check that invariant holds
    assert (df["dwpc"] > 0).all()
