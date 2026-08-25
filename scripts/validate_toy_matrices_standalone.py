"""Build and validate a synthetic Hetnet using adjacency matrices.

This is a standalone validation script. It creates its own toy graph and
checks direct and indirect path counts to EEF1A1.
"""

import numpy as np
from scipy import sparse
from hetnetpy import hetnet
from hetmatpy.matrix import metaedge_to_adjacency_matrix


def build_metagraph():
    """Create the toy network schema."""
    edge_tuples = [
        ("Compound", "Gene", "binds", "forward"),
        ("Gene", "Gene", "interacts", "both"),
    ]
    return hetnet.MetaGraph.from_edge_tuples(edge_tuples)


def build_graph(metagraph):
    """Create the five-node, four-edge toy graph."""
    graph = hetnet.Graph(metagraph)

    graph.add_node("Compound", "C1", name="Compound A")
    graph.add_node("Compound", "C2", name="Compound B")
    graph.add_node("Compound", "C3", name="Compound C")
    graph.add_node("Gene", "EEF1A1", name="EEF1A1")
    graph.add_node("Gene", "PARTNER1", name="Synthetic Partner 1")

    c1 = graph.get_node(("Compound", "C1"))
    c2 = graph.get_node(("Compound", "C2"))
    c3 = graph.get_node(("Compound", "C3"))
    eef1a1 = graph.get_node(("Gene", "EEF1A1"))
    partner1 = graph.get_node(("Gene", "PARTNER1"))

    graph.add_edge(c1, eef1a1, "binds", "forward")
    graph.add_edge(c2, partner1, "binds", "forward")
    graph.add_edge(c3, partner1, "binds", "forward")
    graph.add_edge(partner1, eef1a1, "interacts", "both")

    return graph


def to_integer_array(values):
    """Convert sparse or dense values to an integer NumPy array."""
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.int64)


def print_matrix(title, rows, columns, values):
    """Print a labelled matrix."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(f"Rows:    {list(rows)}")
    print(f"Columns: {list(columns)}")
    print(values)


def main():
    """Run the complete matrix and path-count validation."""
    print("=" * 60)
    print("TOY HETNET MATRIX VALIDATION")
    print("=" * 60)

    metagraph = build_metagraph()
    graph = build_graph(metagraph)
    metaedges = {str(edge): edge for edge in metagraph.get_edges()}

    compound_ids, binding_gene_ids, binds = metaedge_to_adjacency_matrix(
        graph,
        metaedges["Compound > binds > Gene"],
        dense_threshold=1.0,
    )
    interaction_gene_ids, target_gene_ids, interacts = (
        metaedge_to_adjacency_matrix(
            graph,
            metaedges["Gene - interacts - Gene"],
            dense_threshold=1.0,
        )
    )

    binds = to_integer_array(binds)
    interacts = to_integer_array(interacts)

    if list(binding_gene_ids) != list(interaction_gene_ids):
        raise RuntimeError("Gene ordering differs between matrices.")

    indirect_counts = binds @ interacts

    print_matrix(
        "Compound-binds-Gene adjacency matrix",
        compound_ids,
        binding_gene_ids,
        binds,
    )
    print_matrix(
        "Gene-interacts-Gene adjacency matrix",
        interaction_gene_ids,
        target_gene_ids,
        interacts,
    )
    print_matrix(
        "Compound-binds-Gene-interacts-Gene path-count matrix",
        compound_ids,
        target_gene_ids,
        indirect_counts,
    )

    direct_index = list(binding_gene_ids).index("EEF1A1")
    indirect_index = list(target_gene_ids).index("EEF1A1")
    direct_eef1a1 = binds[:, direct_index]
    indirect_eef1a1 = indirect_counts[:, indirect_index]

    np.testing.assert_array_equal(direct_eef1a1, np.array([1, 0, 0]))
    np.testing.assert_array_equal(indirect_eef1a1, np.array([0, 1, 1]))

    print(f"\nDirect EEF1A1 counts: {direct_eef1a1.tolist()}")
    print(f"Indirect EEF1A1 counts: {indirect_eef1a1.tolist()}")
    print("\nMatrix and path-count validation completed successfully.")


if __name__ == "__main__":
    main()
