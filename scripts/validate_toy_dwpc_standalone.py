"""Validate degree-weighted path counts for a synthetic Hetnet.

This standalone script recreates the five-node toy graph, calculates
DWPC matrices with damping w=0.5 using hetmatpy, and compares the
results with manually expected values.
"""

import numpy as np
from scipy import sparse
from hetnetpy import hetnet
from hetmatpy.degree_weight import dwpc


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


def to_array(values):
    """Convert sparse or dense values to a float NumPy array."""
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64)


def print_matrix(title, rows, columns, values):
    """Print a labelled matrix rounded for readability."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(f"Rows:    {list(rows)}")
    print(f"Columns: {list(columns)}")
    print(np.round(values, 8))


def main():
    """Calculate and validate direct and indirect DWPC matrices."""
    damping = 0.5
    expected_weight = 1.0 / np.sqrt(2.0)

    print("=" * 60)
    print("TOY HETNET DWPC VALIDATION")
    print("=" * 60)
    print(f"Damping exponent: {damping}")

    metagraph = build_metagraph()
    graph = build_graph(metagraph)

    direct_metapath = metagraph.get_metapath("Cb>G")
    indirect_metapath = metagraph.get_metapath("Cb>GiG")

    direct_rows, direct_columns, direct_values = dwpc(
        graph,
        direct_metapath,
        damping=damping,
        dense_threshold=1.0,
    )
    indirect_rows, indirect_columns, indirect_values = dwpc(
        graph,
        indirect_metapath,
        damping=damping,
        dense_threshold=1.0,
    )

    direct_values = to_array(direct_values)
    indirect_values = to_array(indirect_values)

    print_matrix(
        "Direct DWPC matrix: Compound-binds-Gene",
        direct_rows,
        direct_columns,
        direct_values,
    )
    print_matrix(
        "Indirect DWPC matrix: Compound-binds-Gene-interacts-Gene",
        indirect_rows,
        indirect_columns,
        indirect_values,
    )

    expected_direct = np.array(
        [
            [1.0, 0.0],
            [0.0, expected_weight],
            [0.0, expected_weight],
        ]
    )
    expected_indirect = np.array(
        [
            [0.0, 1.0],
            [expected_weight, 0.0],
            [expected_weight, 0.0],
        ]
    )

    np.testing.assert_allclose(
        direct_values,
        expected_direct,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        indirect_values,
        expected_indirect,
        rtol=1e-12,
        atol=1e-12,
    )

    eef1a1_index = list(indirect_columns).index("EEF1A1")
    indirect_eef1a1 = indirect_values[:, eef1a1_index]

    print(f"\nExpected hub-weight value: {expected_weight:.12f}")
    print(
        "Indirect EEF1A1 DWPC values: "
        f"{np.round(indirect_eef1a1, 12).tolist()}"
    )
    print("\nDWPC validation completed successfully.")


if __name__ == "__main__":
    main()
