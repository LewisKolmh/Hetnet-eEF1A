"""Build a synthetic heterogeneous network for software validation.

All nodes and relationships in this script are synthetic test data.
They must not be interpreted as biological evidence.
"""

from hetnetpy import hetnet


def build_metagraph() -> hetnet.MetaGraph:
    """Create the schema for the synthetic test network."""

    edge_tuples = [
        ("Compound", "Gene", "binds", "forward"),
        ("Gene", "Gene", "interacts", "both"),
    ]

    return hetnet.MetaGraph.from_edge_tuples(edge_tuples)


def build_graph(metagraph: hetnet.MetaGraph) -> hetnet.Graph:
    """Create the synthetic graph."""

    graph = hetnet.Graph(metagraph)

    graph.add_node("Compound", "C1", name="Compound A")
    graph.add_node("Compound", "C2", name="Compound B")
    graph.add_node("Compound", "C3", name="Compound C")

    graph.add_node("Gene", "EEF1A1", name="EEF1A1")
    graph.add_node(
        "Gene",
        "PARTNER1",
        name="Synthetic Partner 1",
    )

    compound_a = graph.get_node("Compound", "C1")
    compound_b = graph.get_node("Compound", "C2")
    compound_c = graph.get_node("Compound", "C3")
    eef1a1 = graph.get_node("Gene", "EEF1A1")
    partner_1 = graph.get_node("Gene", "PARTNER1")

    graph.add_edge(compound_a, eef1a1, "binds", "forward")
    graph.add_edge(compound_b, partner_1, "binds", "forward")
    graph.add_edge(compound_c, partner_1, "binds", "forward")
    graph.add_edge(partner_1, eef1a1, "interacts", "both")

    return graph


def main() -> None:
    """Build and summarise the synthetic network."""

    metagraph = build_metagraph()
    graph = build_graph(metagraph)

    nodes = list(graph.get_nodes())
    edges = list(graph.get_edges())

    print("=" * 60)
    print("SYNTHETIC HETNET VALIDATION")
    print("=" * 60)

    print(f"\nNode count: {len(nodes)}")
    for node in nodes:
        print(f"  {node}")

    print(f"\nEdge count: {len(edges)}")
    for edge in edges:
        print(f"  {edge}")

    print("\nMetanodes:")
    for metanode in metagraph.get_nodes():
        print(f"  {metanode}")

    print("\nMetaedges:")
    for metaedge in metagraph.get_edges():
        print(f"  {metaedge}")

    print("\nToy graph constructed successfully.")


if __name__ == "__main__":
    main()