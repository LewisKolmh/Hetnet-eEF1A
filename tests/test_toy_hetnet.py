"""Automated tests for the synthetic heterogeneous network."""

from scripts.build_toy_hetnet import build_graph, build_metagraph


def create_toy_graph():
    """Construct and return the synthetic metagraph and graph."""

    metagraph = build_metagraph()
    graph = build_graph(metagraph)

    return metagraph, graph


def test_toy_graph_has_expected_node_count() -> None:
    """The synthetic graph should contain exactly five nodes."""

    _, graph = create_toy_graph()

    assert len(list(graph.get_nodes())) == 5


def test_toy_graph_has_expected_edge_count() -> None:
    """The synthetic graph should contain exactly four edges."""

    _, graph = create_toy_graph()

    assert len(list(graph.get_edges())) == 4


def test_expected_nodes_exist() -> None:
    """All expected synthetic nodes should exist."""

    _, graph = create_toy_graph()

    expected_node_ids = [
        ("Compound", "C1"),
        ("Compound", "C2"),
        ("Compound", "C3"),
        ("Gene", "EEF1A1"),
        ("Gene", "PARTNER1"),
    ]

    for node_id in expected_node_ids:
        assert graph.get_node(node_id) is not None


def test_metagraph_has_expected_metanodes() -> None:
    """The metagraph should contain Compound and Gene node types."""

    metagraph, _ = create_toy_graph()

    metanode_names = {
        str(metanode)
        for metanode in metagraph.get_nodes()
    }

    assert metanode_names == {"Compound", "Gene"}


def test_metagraph_has_expected_metaedges() -> None:
    """The metagraph should contain two relationship types."""

    metagraph, _ = create_toy_graph()

    metaedges = list(metagraph.get_edges())

    assert len(metaedges) == 2

    metaedge_text = {
        str(metaedge)
        for metaedge in metaedges
    }

    assert "Compound > binds > Gene" in metaedge_text
    assert "Gene - interacts - Gene" in metaedge_text


def test_eef1a1_node_exists() -> None:
    """The synthetic target EEF1A1 should be retrievable."""

    _, graph = create_toy_graph()

    eef1a1 = graph.get_node(("Gene", "EEF1A1"))

    assert eef1a1 is not None
    assert str(eef1a1) == "Gene::EEF1A1"


def test_partner_node_exists() -> None:
    """The synthetic interaction partner should be retrievable."""

    _, graph = create_toy_graph()

    partner = graph.get_node(("Gene", "PARTNER1"))

    assert partner is not None
    assert str(partner) == "Gene::PARTNER1"
