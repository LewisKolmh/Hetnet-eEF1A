"""Tests for the ChEMBL activity filter (src/activity_filter.py).

The committed baseline admitted `Kd > 30000 nM` records as compound-binds-gene
edges, i.e. non-detections entered the graph as evidence of binding. These tests
pin the filter that removes them.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.activity_filter import filter_activities, collapse_to_edges, AFFINITY_TYPES


def _record(**kw):
    base = dict(
        compound_id="CHEMBL1",
        target_chembl_id="CHEMBL_T1",
        protein_id="EEF1A1",
        standard_type="Kd",
        standard_relation="=",
        standard_value=10.0,
        standard_units="nM",
        pchembl_value=8.0,
        assay_type="B",
        assay_chembl_id="CHEMBL_A1",
        target_organism="Homo sapiens",
        data_validity_comment=None,
    )
    base.update(kw)
    return base


def test_non_detect_relations_are_dropped():
    df = pd.DataFrame([
        _record(standard_relation="=", standard_value=10.0, pchembl_value=8.0),
        _record(standard_relation=">", standard_value=30000.0, pchembl_value=4.5),
        _record(standard_relation=">=", standard_value=30000.0, pchembl_value=4.5),
    ])
    kept, _ = filter_activities(df)
    assert len(kept) == 1
    assert set(kept["standard_relation"]) == {"="}


def test_non_affinity_measurement_types_are_dropped():
    df = pd.DataFrame([
        _record(standard_type="Kd"),
        _record(standard_type="Inhibition"),
        _record(standard_type="Activity"),
    ])
    kept, _ = filter_activities(df)
    assert set(kept["standard_type"]) <= set(AFFINITY_TYPES)
    assert len(kept) == 1


def test_potency_threshold_is_applied():
    df = pd.DataFrame([
        _record(pchembl_value=8.0),
        _record(pchembl_value=4.0),
    ])
    kept, _ = filter_activities(df, min_pchembl=5.0)
    assert list(kept["pchembl_value"]) == [8.0]


def test_missing_pchembl_is_dropped_not_imputed():
    df = pd.DataFrame([_record(pchembl_value=None)])
    kept, _ = filter_activities(df)
    assert kept.empty


def test_audit_accounts_for_every_input_record():
    df = pd.DataFrame([
        _record(),
        _record(standard_relation=">"),
        _record(standard_type="Inhibition"),
        _record(pchembl_value=3.0),
    ])
    kept, audit = filter_activities(df)
    assert audit.attrs["n_input"] == len(df)
    assert audit["records_remaining"].iloc[-1] == len(kept)
    # every input record is either kept or attributed to exactly one criterion
    assert audit["records_removed"].sum() + len(kept) == len(df)


def test_validity_flagged_records_are_dropped():
    df = pd.DataFrame([
        _record(),
        _record(data_validity_comment="Outside typical range"),
    ])
    kept, _ = filter_activities(df)
    assert len(kept) == 1
    assert kept["data_validity_comment"].isna().all()


def test_collapse_keeps_strongest_affinity_per_pair():
    df = pd.DataFrame([
        _record(pchembl_value=6.0, assay_chembl_id="CHEMBL_A1"),
        _record(pchembl_value=9.0, assay_chembl_id="CHEMBL_A2"),
    ])
    kept, _ = filter_activities(df)
    edges = collapse_to_edges(kept)
    assert len(edges) == 1
    assert edges["pchembl_max"].iloc[0] == pytest.approx(9.0)
    assert edges["n_records"].iloc[0] == 2
