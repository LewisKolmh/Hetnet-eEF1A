import pandas as pd
import pytest

from src.extract_seed_proteins import compute_degrees, load_interactions


def test_compute_degrees_toy():
    df = pd.DataFrame(
        {
            "preferredName_A": ["A", "A", "B"],
            "preferredName_B": ["B", "C", "C"],
        }
    )
    out = compute_degrees(df)
    degrees = dict(zip(out["protein_id"], out["degree"]))
    assert degrees == {"A": 2, "B": 2, "C": 2}


def test_compute_degrees_self_edge():
    df = pd.DataFrame({"preferredName_A": ["A"], "preferredName_B": ["A"]})
    out = compute_degrees(df)
    assert list(out["protein_id"]) == ["A"]
    assert out.loc[0, "degree"] == 1


def test_load_interactions_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        load_interactions(p)


def test_load_interactions_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_interactions(tmp_path / "nope.csv")


def test_real_output_no_duplicates_and_min_size():
    out = pd.read_csv("data/raw/seed_proteins.tsv", sep="\t")
    assert len(out) >= 10
    assert out["protein_id"].is_unique
    assert (out["degree"] > 0).all()
