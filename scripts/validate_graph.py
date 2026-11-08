"""Validate the graph formalization and show what graph algorithms answer.

Proves, on the real validated engines:
  1. the whole-body graph obeys blood conservation (Kirchhoff) -- a physical law
     checked by a graph algorithm;
  2. structural pharmacology falls out of topology (IV reach, oral first pass,
     recirculation, hubs);
  3. clearance reference consistency, with a separate exact IV dose/AUC check;
  4. a pure graph BFS on the bond graph reproduces the force-field exclusion list
     exactly -- the graph verifying the physics engine;
  5. renders the body graph and a subgraph to diagrams.

    python scripts/validate_graph.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vh.body import MIDAZOLAM, PBPKModel, reference_individual
from vh.graph import algorithms as A
from vh.graph.physiology_graph import build_molecular_graph, build_pbpk_graph, exclusions_from_bond_graph
from vh.graph.render import render
from vh.molecular.prep import load_protein

ROOT = Path(__file__).resolve().parents[1]
DOCS, IMG = ROOT / "docs", ROOT / "docs" / "img"
report: dict = {}


def whole_body():
    m = PBPKModel(MIDAZOLAM, reference_individual("male"))
    g, schema = build_pbpk_graph(m)
    print(g.summary())
    problems = g.validate(schema)
    print("schema + invariants:", "all pass" if not problems else problems)
    if problems:
        raise ValueError("invalid PBPK graph: " + "; ".join(problems))

    bal = A.flow_balance(g, "flow", ("flow",))
    print(f"[1 blood conservation] worst residual {bal['max_abs_residual']:.2e} L/h "
          f"(relative {bal['max_rel_residual']:.1e}) -> balanced = {bal['balanced']}")

    iv = A.reachable(g, "venous", ("flow",))
    oral = A.all_paths(g, "gut_lumen", "liver", ("absorbs", "flow"), max_length=4)
    loops = A.recirculation_loops(g.edge_view("flow"), ("flow",))
    bc = A.betweenness_centrality(g.edge_view("flow"), ("flow",))
    hubs = sorted(bc.items(), key=lambda x: -x[1])[:3]
    print(f"[2 structure] IV reaches {len(iv)} compartments (brain: {'brain' in iv}); "
          f"oral first-pass path {oral[0] if oral else None}; "
          f"{len(loops)} recirculation loop (size {max(len(c) for c in loops)}); "
          f"hubs {[h[0] for h in hubs]}")

    # elimination split from the clearance edges
    cl = {e.attrs['organ']: e.attrs['cl'] for e in g.edges_of_type('clears')}
    print(f"[2 elimination] hepatic {cl['liver']:.2f} L/h vs renal {cl['kidney']:.2f} L/h "
          f"-> hepatic share {100*cl['liver']/sum(cl.values()):.0f}% "
          "(reported plasma-referenced parameters)")

    report["whole_body"] = {
        "summary": g.summary(), "schema_ok": not problems,
        "blood_conservation_rel_residual": bal["max_rel_residual"],
        "iv_reaches": len(iv), "brain_reachable": "brain" in iv,
        "oral_first_pass_path": oral[0] if oral else None,
        "recirculation_loop_size": max(len(c) for c in loops),
        "hubs": [h[0] for h in hubs],
        "hepatic_reported_parameter_fraction": cl["liver"] / sum(cl.values()),
        "clearance_units": "L/h", "clearance_reference_basis": "plasma",
    }
    g.save_json(DOCS / "pbpk_graph.json")
    render(g, str(IMG / "pbpk_graph.mmd"), direction="LR", title="Midazolam whole-body graph")

    # 3. Export consistency is not an independent check of the ODE. The renal
    # reporting parameter is nominal filtration clearance, before finite flow.
    sub = g.subgraph(["arterial", "liver", "kidney", "venous", "gut", "spleen", "pancreas", "eliminated"],
                     name="elimination-subgraph")
    sub_cl = sum(e.attrs["cl"] for e in sub.edges_of_type("clears"))
    reported_cl = m.p.cl_total_plasma_l_h
    # Independent analytic integrated mass balance: M_active * integral(x dt)
    # = -x(0) for a 1 mg IV bolus, with all circulating drug gone at infinity.
    active = [i for i, n in enumerate(m.names) if n in m.V]
    initial = np.zeros(len(active))
    initial[active.index(m.ix["venous"])] = 1.0
    integrated = -np.linalg.solve(m.M[np.ix_(active, active)], initial)
    auc = integrated[active.index(m.ix["venous"])] / m.V["venous"] / m.p.bp
    dose_over_auc = 1.0 / auc
    # Exact well-stirred organ balance, evaluated from inputs, not edge values.
    q_h, q_r = m.ind.liver_flow_total * m.p.bp, m.ind.flows["kidney"] * m.p.bp
    intrinsic_h = m.p.fu * m.p.clint_hep_l_h
    filtration = m.compound.renal_filtration * m.p.fu * m.ind.gfr
    analytic_cl = q_h * intrinsic_h / (q_h + intrinsic_h) + q_r * filtration / (q_r + filtration)
    print(f"[3 clearance reporting] subgraph {sub_cl:.6f} L/h = reported parameter "
          f"sum {reported_cl:.6f} L/h (plasma); exact IV dose/AUC {dose_over_auc:.6f} L/h")
    print(f"[3 independent balance] dose/AUC vs analytic organ balance: "
          f"{abs(dose_over_auc - analytic_cl):.2e} L/h error")
    render(sub, str(IMG / "pbpk_elimination_subgraph.mmd"), direction="LR",
           title="Elimination subgraph (plasma-referenced reporting parameters)")
    report["clearance_reporting"] = {
        "units": "L/h", "reference_basis": "plasma",
        "subgraph_parameter_sum": sub_cl, "model_reported_parameter_sum": reported_cl,
        "export_matches": abs(sub_cl - reported_cl) < 1e-9,
        "iv_dose_over_venous_plasma_auc": dose_over_auc,
        "analytic_well_stirred_clearance": analytic_cl,
        "analytic_absolute_error": abs(dose_over_auc - analytic_cl),
        "reported_sum_minus_dose_over_auc": sub_cl - dose_over_auc,
        "limitation": "Reported renal filtration parameter omits finite-flow extraction; "
                      "export agreement is not independent model validation.",
    }


def molecular():
    system, pos, top, omm = load_protein(ROOT / "data/structures/1l2y_trpcage.pdb")
    g, schema = build_molecular_graph(system)
    print("\n" + g.summary())
    comps = A.connected_components(g, ("bond",))
    cyc = [c for c in A.simple_cycles(g.edge_view("bond"), ("bond",), max_cycles=300) if len(c) - 1 >= 3]
    rings = sorted({len(c) - 1 for c in cyc})
    graph_excl = exclusions_from_bond_graph(system, g)
    engine_excl = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in np.asarray(system.exceptions)}
    ok = graph_excl == engine_excl
    print(f"[4 graph verifies the engine] bond-graph components {len(comps)}; ring sizes {rings}; "
          f"exclusion pairs: graph {len(graph_excl)} == engine {len(engine_excl)} -> {ok} "
          f"(symmetric difference {len(graph_excl ^ engine_excl)})")
    report["molecular"] = {"n_atoms": system.n_atoms, "components": len(comps), "ring_sizes": rings,
                           "graph_exclusions": len(graph_excl), "engine_exclusions": len(engine_excl),
                           "exclusions_match": ok}


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    whole_body()
    molecular()
    def _clean(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    (DOCS / "validation_graph.json").write_text(json.dumps(report, indent=1, default=_clean))
    print("\nwrote docs/validation_graph.json, docs/pbpk_graph.json, docs/img/pbpk_graph.*")


if __name__ == "__main__":
    main()
