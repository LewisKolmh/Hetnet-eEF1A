import numpy as np
import pandas as pd
from scipy import sparse


def test_node_ids_unique_and_dense():
    nodes = pd.read_csv("data/processed/nodes/nodes.full-interactome.tsv", sep="\t")
    assert nodes["node_id"].is_unique
    assert set(nodes["node_id"]) == set(range(len(nodes)))


def test_eef1a_only_is_subset_of_full_gene_and_go_nodes():
    full = pd.read_csv("data/processed/nodes/nodes.full-interactome.tsv", sep="\t")
    eef1a = pd.read_csv("data/processed/nodes/nodes.eef1a-only.tsv", sep="\t")
    # Gene/GO/Pathway node sets come from the same seed proteins regardless of scope
    for mtype in ["Gene", "BiologicalProcess", "MolecularFunction", "CellularComponent", "Pathway"]:
        full_ext = set(full[full.metanode_type == mtype]["external_id"])
        eef1a_ext = set(eef1a[eef1a.metanode_type == mtype]["external_id"])
        assert full_ext == eef1a_ext, f"{mtype} node sets differ between scopes"
    # Compound nodes differ: eef1a-only must be a subset of full-interactome
    full_c = set(full[full.metanode_type == "Compound"]["external_id"])
    eef1a_c = set(eef1a[eef1a.metanode_type == "Compound"]["external_id"])
    assert eef1a_c <= full_c


def test_matrix_shapes_match_node_count():
    nodes = pd.read_csv("data/processed/nodes/nodes.full-interactome.tsv", sep="\t")
    n = len(nodes)
    for metaedge in ["CbG", "GiG", "GpBP", "GpMF", "GpCC", "GpPW"]:
        mat = sparse.load_npz(f"data/processed/matrices/{metaedge}.full-interactome.npz")
        assert mat.shape == (n, n)


def test_gig_matrix_is_symmetric():
    mat = sparse.load_npz("data/processed/matrices/GiG.full-interactome.npz")
    assert (mat != mat.T).nnz == 0


def test_matrix_is_binary():
    mat = sparse.load_npz("data/processed/matrices/CbG.full-interactome.npz")
    vals = np.unique(mat.data)
    assert set(vals) <= {1.0}


def test_cbg_nnz_matches_edge_count():
    cbg_edges = pd.read_csv("data/raw/compound_binds_gene.full-interactome.tsv", sep="\t")
    mat = sparse.load_npz("data/processed/matrices/CbG.full-interactome.npz")
    assert mat.nnz == len(cbg_edges.drop_duplicates(subset=["compound_id", "protein_id"]))
