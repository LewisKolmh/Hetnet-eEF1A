"""Sanity checks on the real fetched Phase 1 outputs (no network calls -
these validate the on-disk results already produced by s01-s04)."""
import pandas as pd


def test_seed_proteins_exists_and_valid():
    df = pd.read_csv("data/raw/seed_proteins.tsv", sep="\t")
    assert len(df) == 18
    assert df["protein_id"].is_unique


def test_eef1a_only_compounds_are_subset_of_full_interactome():
    eef1a = pd.read_csv("data/raw/compounds_chembl.eef1a-only.tsv", sep="\t")
    full = pd.read_csv("data/raw/compounds_chembl.full-interactome.tsv", sep="\t")
    assert len(eef1a) > 0
    assert set(eef1a["compound_id"]) <= set(full["compound_id"])
    assert len(full) >= len(eef1a)


def test_known_eef1a_binders_present():
    eef1a = pd.read_csv("data/raw/compounds_chembl.eef1a-only.tsv", sep="\t")
    names = set(eef1a["pref_name"].dropna().str.upper())
    # well-known eEF1A-targeting translation elongation inhibitors
    assert "CYCLOHEXIMIDE" in names
    assert "LACTIMIDOMYCIN" in names


def test_uniprot_mapping_covers_all_seed_proteins():
    seed = pd.read_csv("data/raw/seed_proteins.tsv", sep="\t")
    mapping = pd.read_csv("data/raw/uniprot_mapping.tsv", sep="\t")
    assert set(seed["protein_id"]) == set(mapping["protein_id"])
    n_mapped = mapping["uniprot_accession"].notna().sum()
    assert n_mapped >= 15  # allow a couple of misses, but not systemic failure


def test_go_annotations_nonempty_and_valid_aspect():
    go = pd.read_csv("data/raw/go_annotations.tsv", sep="\t")
    assert len(go) > 0
    assert go["go_id"].str.startswith("GO:").all()
    assert set(go["aspect"]) <= {"BiologicalProcess", "MolecularFunction", "CellularComponent"}


def test_pathway_annotations_nonempty():
    pw = pd.read_csv("data/raw/pathway_annotations.tsv", sep="\t")
    assert len(pw) > 0
    assert pw["pathway_id"].str.startswith("R-HSA-").all()


def test_gene_interacts_gene_no_self_edges_and_mapped():
    edges = pd.read_csv("data/raw/gene_interacts_gene.tsv", sep="\t")
    assert (edges["gene_a_symbol"] != edges["gene_b_symbol"]).all()
    assert edges["gene_a_id"].notna().all()
    assert edges["gene_b_id"].notna().all()
    # undirected dedup: no pair should appear twice in either order
    pairs = set(tuple(sorted((a, b))) for a, b in zip(edges["gene_a_symbol"], edges["gene_b_symbol"]))
    assert len(pairs) == len(edges)
