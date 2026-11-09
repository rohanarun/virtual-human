"""Graph-layer tests: the core data structure, the algorithms, and the two
headline verifications (blood conservation on the body graph; the force-field
exclusion topology reproduced from the bond graph)."""

import numpy as np
import pytest

from vh.graph import Graph, Schema
from vh.graph import algorithms as A
from vh.graph.core import EdgeRule


def diamond():
    g = Graph("diamond")
    for n in "sabt":
        g.add_node(n, "x")
    g.add_edge("s", "a", "e", flow=3.0)
    g.add_edge("s", "b", "e", flow=5.0)
    g.add_edge("a", "t", "e", flow=3.0)
    g.add_edge("b", "t", "e", flow=5.0)
    return g


def test_reachability_and_paths():
    g = diamond()
    assert A.reachable(g, "s") == {"a", "b", "t"}
    paths = A.all_paths(g, "s", "t")
    assert sorted(paths) == [["s", "a", "t"], ["s", "b", "t"]]
    assert A.bfs_distances(g, "s")["t"] == 2


def test_cycle_and_scc_detection():
    g = Graph("loop")
    for n in "abc":
        g.add_node(n, "x")
    g.add_edge("a", "b", "e")
    g.add_edge("b", "c", "e")
    g.add_edge("c", "a", "e")
    assert A.topological_sort(g) is None                    # has a cycle
    loops = A.recirculation_loops(g, ("e",))
    assert len(loops) == 1 and set(loops[0]) == {"a", "b", "c"}
    g2 = diamond()
    assert A.topological_sort(g2) is not None               # a DAG


def test_widest_path_picks_the_high_capacity_route():
    g = diamond()
    path, cap = A.widest_path(g, "s", "t", "flow")
    assert path == ["s", "b", "t"] and cap == 5.0


def test_flow_balance_detects_imbalance():
    g = diamond()
    bal = A.flow_balance(g, "flow", ("e",), sources=("s",), sinks=("t",))
    assert bal["balanced"]                                  # a and b are balanced (in==out)
    g.add_edge("s", "a", "e", flow=1.0)                     # break balance at a
    bal2 = A.flow_balance(g, "flow", ("e",), sources=("s",), sinks=("t",))
    assert not bal2["balanced"]


def test_connected_components_and_subgraph():
    g = Graph("two")
    for n in "abcd":
        g.add_node(n, "x")
    g.add_edge("a", "b", "e")
    g.add_edge("c", "d", "e")
    comps = A.connected_components(g, ("e",))
    assert sorted(sorted(c) for c in comps) == [["a", "b"], ["c", "d"]]
    sub = g.subgraph(["a", "b"])
    assert set(sub.nodes) == {"a", "b"} and len(sub.edges) == 1


def test_subgraph_order_is_reproducible_across_hash_seeds():
    # String-set iteration order changes with PYTHONHASHSEED; a subgraph must
    # follow the parent graph's order so its JSON and Mermaid are byte-stable.
    import os
    import subprocess
    import sys

    g = Graph("ordered")
    for n in "fedcba":
        g.add_node(n, "x")
    g.add_edge("f", "a", "e")
    assert list(g.subgraph(["a", "c", "f"]).nodes) == ["f", "c", "a"]
    assert list(g.subgraph(reversed(["a", "c", "f"])).nodes) == ["f", "c", "a"]
    with pytest.raises(KeyError, match="unknown nodes"):
        g.subgraph(["a", "missing"])

    code = ("from vh.body import PBPKModel, MIDAZOLAM\n"
            "from vh.graph.physiology_graph import build_pbpk_graph\n"
            "from vh.graph.render import to_mermaid\n"
            "g, _ = build_pbpk_graph(PBPKModel(MIDAZOLAM))\n"
            "ids = ['arterial', 'liver', 'kidney', 'venous', 'gut', 'spleen', 'pancreas', 'eliminated']\n"
            "print(to_mermaid(g.subgraph(set(ids))))\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs = set()
    for seed in ("0", "1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed,
               "PYTHONPATH": os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))}
        run = subprocess.run([sys.executable, "-c", code], env=env, cwd=root,
                             capture_output=True, text=True, check=True, timeout=120)
        outputs.add(run.stdout)
    assert len(outputs) == 1


def test_schema_validation_catches_bad_edges():
    schema = Schema(node_types={"blood": set(), "organ": {"volume"}},
                    edge_rules=[EdgeRule("flow", {"blood", "organ"}, {"blood", "organ"}, {"flow"})])
    g = Graph("x")
    g.add_node("art", "blood")
    g.add_node("liver", "organ", volume=1.8)
    g.add_edge("art", "liver", "flow", flow=25.0)
    assert g.validate(schema) == []
    g.add_node("bad", "organ")                              # missing required 'volume'
    g.add_edge("art", "liver", "flow")                     # missing required 'flow'
    probs = g.validate(schema)
    assert any("volume" in p for p in probs) and any("flow" in p for p in probs)


# --- the verifications on real engines --------------------------------------
@pytest.mark.parametrize("sex", ["male", "female"])
def test_pbpk_graph_conserves_blood_flow(sex):
    pytest.importorskip("openmm")  # body model has no openmm dep, but keep CI uniform
    from vh.body import MIDAZOLAM, PBPKModel, reference_individual
    from vh.graph.physiology_graph import build_pbpk_graph
    g, schema = build_pbpk_graph(PBPKModel(MIDAZOLAM, reference_individual(sex)))
    assert g.validate(schema) == []                         # includes the conservation invariant
    bal = A.flow_balance(g, "flow", ("flow",))
    assert bal["max_rel_residual"] < 1e-9


def test_pbpk_graph_structural_pharmacology():
    from vh.body import MIDAZOLAM, PBPKModel, reference_individual
    from vh.graph.physiology_graph import build_pbpk_graph
    g, _ = build_pbpk_graph(PBPKModel(MIDAZOLAM, reference_individual("male")))
    # oral drug reaches the liver before the systemic circulation (first-pass)
    assert ["gut_lumen", "gut", "liver"] in A.all_paths(g, "gut_lumen", "liver", ("absorbs", "flow"), 4)
    # systemic circulation is one big recirculation loop
    loops = A.recirculation_loops(g.edge_view("flow"), ("flow",))
    assert max(len(c) for c in loops) >= 13


def test_bond_graph_reproduces_forcefield_exclusions():
    pytest.importorskip("openmm")
    from pathlib import Path
    from vh.molecular.prep import load_protein
    from vh.graph.physiology_graph import build_molecular_graph, exclusions_from_bond_graph
    data = Path(__file__).resolve().parents[1] / "data/structures/1uao_chignolin.pdb"
    system, pos, top, omm = load_protein(data)
    g, _ = build_molecular_graph(system)
    graph_excl = exclusions_from_bond_graph(system, g)
    engine_excl = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in np.asarray(system.exceptions)}
    assert graph_excl == engine_excl                        # pure graph BFS == force-field topology
    # one connected molecule per chain; chignolin is a single chain
    assert len(A.connected_components(g, ("bond",))) == 1
