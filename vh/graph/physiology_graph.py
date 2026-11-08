"""Formalize the validated engines as typed graphs.

Two builders turn real, already-validated models into graph instances:

  build_pbpk_graph(model)      the whole-body model as a flow network: compartments
                               (organs + blood pools) joined by blood-flow edges,
                               with partition, clearance and absorption edges, every
                               number pulled from the validated PBPK model.

  build_molecular_graph(sys)   a molecule as its atoms (nodes) and bonds (edges),
                               the natural graph on which force-field topology lives.

Both return a `Graph` plus a `Schema` whose invariants state what makes the
formalization physically consistent (blood conservation; the force-field exclusion
topology). Those invariants are checked by graph algorithms, not by trust.
"""

from __future__ import annotations

import numpy as np

from .algorithms import bfs_distances, flow_balance
from .core import EdgeRule, Graph, Schema


# ---------------------------------------------------------------------------
# whole body (PBPK) as a flow network
# ---------------------------------------------------------------------------
def build_pbpk_graph(model, drug_name: str | None = None) -> tuple[Graph, Schema]:
    """The PBPK model as a graph. Blood-flow edges carry L/h; partition edges carry
    the Rodgers-Rowland Kp; clearance edges carry explicitly plasma-referenced
    L/h. These are reporting parameters, not the tissue elimination coefficients
    used by the ODE. The oral route carries the absorption path."""
    from vh.body.physiology import PORTAL, TISSUES

    ind, p = model.ind, model.p
    g = Graph(f"pbpk:{model.compound.name}")
    drug = drug_name or model.compound.name

    # blood pools + external sink for elimination
    g.add_node("arterial", "blood", "organism", label="arterial blood", volume=ind.v_arterial)
    g.add_node("venous", "blood", "organism", label="venous blood", volume=ind.v_venous)
    g.add_node("eliminated", "sink", "organism", label="eliminated")
    g.add_node("gut_lumen", "lumen", "organism", label="gut lumen (oral dose)")

    for t in TISSUES:
        g.add_node(t, "compartment", "organ", label=t,
                   volume=float(ind.volumes[t]), kp=float(p.kp[t]), flow=float(ind.flows[t]))

    co = ind.cardiac_output
    # venous -> lung -> arterial (the whole cardiac output)
    g.add_edge("venous", "lung", "flow", flow=co, model="pbpk.flow")
    g.add_edge("lung", "arterial", "flow", flow=co, model="pbpk.flow")
    # arterial supply to each organ; venous (or portal) drainage
    for t in TISSUES:
        if t == "lung":
            continue
        g.add_edge("arterial", t, "flow", flow=float(ind.flows[t]), model="pbpk.flow")
        if t in PORTAL:
            g.add_edge(t, "liver", "flow", flow=float(ind.flows[t]), model="pbpk.flow", route="portal")
        elif t != "liver":
            g.add_edge(t, "venous", "flow", flow=float(ind.flows[t]), model="pbpk.flow")
    g.add_edge("liver", "venous", "flow", flow=float(ind.liver_flow_total), model="pbpk.flow")

    # the partition coefficient (how strongly the drug concentrates in each tissue,
    # from Rodgers-Rowland) is already on each compartment node as `kp`.

    # Reporting clearances: CL_blood * C_blood = CL_plasma * C_plasma,
    # hence CL_plasma = CL_blood * bp. Never sum blood- and plasma-based CL.
    # The kidney value is nominal filtration clearance, not the finite-flow
    # effective clearance; their sum reproduces p.cl_total_plasma_l_h only.
    g.add_edge("liver", "eliminated", "clears", cl=float(p.cl_hep_blood_l_h * p.bp),
               units="L/h", reference_basis="plasma", clearance_kind="well_stirred_hepatic",
               source="PBPKModel.p.cl_hep_blood_l_h * PBPKModel.p.bp",
               organ="liver", model="pbpk.hepatic")
    g.add_edge("kidney", "eliminated", "clears", cl=float(p.cl_renal_l_h),
               units="L/h", reference_basis="plasma", clearance_kind="nominal_renal_filtration",
               source="PBPKModel.p.cl_renal_l_h", organ="kidney", model="pbpk.renal")
    # oral absorption route
    g.add_edge("gut_lumen", "gut", "absorbs", fa=float(p.fa), fg=float(p.fg), ka=float(p.ka_h), model="pbpk.absorption")

    # target engagement edge (molecular -> body bridge), if the drug has a target
    tgt = model.compound.target
    if tgt is not None:
        g.add_node("target", "target", "molecule", label=tgt.name, kd_nm=float(tgt.kd_nm), tissue=tgt.tissue)
        g.add_edge(tgt.tissue, "target", "binds", kd_nm=float(tgt.kd_nm), model="bridge.occupancy")

    schema = _pbpk_schema(co)
    return g, schema


def _pbpk_schema(cardiac_output: float) -> Schema:
    schema = Schema(
        node_types={"blood": set(), "compartment": {"volume", "kp", "flow"}, "sink": set(),
                    "lumen": set(), "target": {"kd_nm"}},
        edge_rules=[
            EdgeRule("flow", {"blood", "compartment"}, {"blood", "compartment"}, {"flow"}),
            EdgeRule("clears", {"compartment"}, {"sink"},
                     {"cl", "units", "reference_basis", "clearance_kind", "source"}),
            EdgeRule("absorbs", {"lumen"}, {"compartment"}, {"fa"}),
            EdgeRule("binds", {"compartment"}, {"target"}, {"kd_nm"}),
        ],
    )

    @schema.invariant("clearance-reference")
    def _clearance_reference(g: Graph) -> list[str]:
        out = []
        for e in g.edges_of_type("clears"):
            if e.attrs.get("units") != "L/h" or e.attrs.get("reference_basis") != "plasma":
                out.append(f"edge {e.key}: clearance must use L/h referenced to plasma")
            cl = e.attrs.get("cl")
            try:
                valid = (not isinstance(cl, (bool, np.bool_))
                         and isinstance(cl, (int, float, np.number)) and np.isrealobj(cl)
                         and np.isfinite(float(cl)) and cl >= 0)
            except (OverflowError, TypeError, ValueError):
                valid = False
            if not valid:
                out.append(f"edge {e.key}: clearance must be a finite nonnegative number")
        return out

    @schema.invariant("blood-conservation")
    def _conserve(g: Graph) -> list[str]:
        # every compartment and the blood pools must balance blood flow in = out
        bal = flow_balance(g, "flow", ("flow",), sources=(), sinks=())
        out = []
        for n, r in bal["residual"].items():
            if g.node(n).type in ("blood", "compartment") and abs(r) > 1e-6:
                out.append(f"compartment {n} has net blood flow {r:+.4f} L/h (should be 0)")
        return out

    @schema.invariant("single-blood-loop")
    def _loop(g: Graph) -> list[str]:
        from .algorithms import recirculation_loops
        loops = recirculation_loops(g.edge_view("flow"), ("flow",))
        # the systemic circulation is one big strongly-connected loop; flag if absent
        biggest = max((len(c) for c in loops), default=0)
        return [] if biggest >= 3 else ["no systemic circulation loop found in the flow graph"]

    @schema.invariant("elimination-reachable")
    def _elim(g: Graph) -> list[str]:
        from .algorithms import reachable
        r = reachable(g, "venous", ("flow", "clears"))
        return [] if "eliminated" in r else ["drug in blood cannot reach elimination"]

    return schema


# ---------------------------------------------------------------------------
# a molecule as its bond graph
# ---------------------------------------------------------------------------
def build_molecular_graph(system, max_atoms: int | None = None) -> tuple[Graph, Schema]:
    """Atoms as nodes, bonds as (undirected) edges. The graph on which force-field
    connectivity lives: molecules are connected components, rings are cycles, and
    1-2/1-3/1-4 neighbor relations are graph distances 1/2/3."""
    g = Graph(f"mol:{system.name}")
    n = system.n_atoms if max_atoms is None else min(max_atoms, system.n_atoms)
    elements = system.elements or ["X"] * system.n_atoms
    names = system.atom_names or [f"a{i}" for i in range(system.n_atoms)]
    resn = system.residue_names or ["UNK"] * system.n_atoms
    for i in range(n):
        g.add_node(f"a{i}", "atom", "atom", label=f"{names[i]}",
                   element=elements[i], residue=resn[i],
                   charge=float(system.charges[i]) if system.charges is not None else 0.0)
    for a, b in np.asarray(system.bonds).reshape(-1, 2):
        if a < n and b < n:
            g.add_edge(f"a{int(a)}", f"a{int(b)}", "bond", directed=False, model="ff.bond")
    return g, _molecular_schema()


def _molecular_schema() -> Schema:
    schema = Schema(node_types={"atom": {"element"}}, edge_rules=[EdgeRule("bond", {"atom"}, {"atom"})])

    @schema.invariant("no-isolated-heavy-atoms")
    def _iso(g: Graph) -> list[str]:
        out = []
        for nd in g.nodes.values():
            if nd.attrs.get("element") not in ("H", None) and g.degree(nd.id) == 0:
                out.append(f"heavy atom {nd.id} ({nd.attrs.get('element')}) has no bonds")
        return out

    return schema


def exclusions_from_bond_graph(system, g: Graph) -> set:
    """Pairs within graph distance <= 3 on the bond graph (the 1-2, 1-3, 1-4
    neighbors). These are exactly the pairs a force field excludes/scales -- so
    this pure graph computation should reproduce the engine's exclusion list."""
    pairs = set()
    ids = list(g.nodes)
    idx = {f"a{i}": i for i in range(system.n_atoms)}
    for s in ids:
        d = bfs_distances(g, s, ("bond",))
        for t, dist in d.items():
            if 1 <= dist <= 3:
                i, j = idx[s], idx[t]
                pairs.add((min(i, j), max(i, j)))
    return pairs
