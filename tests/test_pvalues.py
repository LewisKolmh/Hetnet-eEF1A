"""Tests for the corrected p-value machinery (src/compute_pvalues.py).

The property that matters is calibration: taking the minimum p-value over the
(metapath, target) cells of a compound is a selection, so the minimum is not
itself a p-value. The corrected code calibrates it by applying the same
minimisation inside every permutation. These tests check that the calibrated
statistic is uniform under the null, that the uncalibrated minimum is not, and
that the empirical tail respects the resolution floor set by the permutation
count.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.compute_pvalues import empirical_tail_p, permutation_cell_p


@pytest.fixture
def null_cube():
    """(n_perm, n_compound, n_cell) draws with correlated cells, no signal."""
    rng = np.random.default_rng(0)
    n_perm, n_comp, n_cell = 400, 30, 6
    # correlated cells: a per-compound-per-permutation common factor plus noise,
    # mimicking the shared prefix of the annotation metapaths
    common = rng.gamma(2.0, 1.0, size=(n_perm, n_comp, 1))
    cube = common * rng.gamma(2.0, 1.0, size=(n_perm, n_comp, n_cell))
    return cube


def test_empirical_tail_respects_resolution_floor(null_cube):
    n_perm = null_cube.shape[0]
    observed = null_cube[0]
    p = empirical_tail_p(observed, null_cube)
    assert p.min() >= 1.0 / (n_perm + 1) - 1e-12
    assert p.max() <= 1.0


def test_empirical_tail_is_monotone_in_the_observed_value(null_cube):
    observed = null_cube.mean(axis=0)
    p_low = empirical_tail_p(observed, null_cube)
    p_high = empirical_tail_p(observed * 10.0, null_cube)
    assert np.all(p_high <= p_low + 1e-12)


def test_zero_observed_scores_are_not_significant(null_cube):
    observed = np.zeros(null_cube.shape[1:])
    p = empirical_tail_p(observed, null_cube)
    assert np.all(p > 0.5), "a zero DWPC is the least extreme value possible"


def test_calibrated_min_p_is_calibrated_under_the_null(null_cube):
    """Held-out permutations' calibrated min-p should sit near U(0,1).

    Mirrors main(): observed cell p-values against the remaining permutations,
    minimised over cells, then compared to the same minimisation applied inside
    every permutation (leave-one-out).
    """
    held, rest = null_cube[:40], null_cube[40:]
    minp_perm = permutation_cell_p(rest).min(axis=2)          # (R, n_compounds)
    n_perm = rest.shape[0]
    calibrated = []
    for r in range(held.shape[0]):
        minp_obs = empirical_tail_p(held[r], rest).min(axis=1)
        calibrated.append((1.0 + (minp_perm <= minp_obs[None, :]).sum(axis=0)) / (1.0 + n_perm))
    calibrated = np.concatenate(calibrated)
    # A calibrated p-value has mean ~0.5; the raw minimum (next test) is ~0.1.
    assert 0.35 < calibrated.mean() < 0.65, f"mean calibrated p = {calibrated.mean():.3f}"
    # and a nominal 5% test should reject at roughly 5%, not 30%
    assert (calibrated <= 0.05).mean() < 0.15


def test_uncalibrated_minimum_is_anticonservative(null_cube):
    """The defect being fixed: min over 6 cells is stochastically below U(0,1)."""
    held_out, rest = null_cube[0], null_cube[1:]
    p_min = empirical_tail_p(held_out, rest).min(axis=1)
    assert p_min.mean() < 0.3, (
        "min over cells should be far below the 0.5 expected of a uniform "
        "p-value; if it is not, the test fixture has no cell multiplicity"
    )
