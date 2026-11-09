"""Clearance export checked against exact organ mass balance, not copied CLs."""

from fractions import Fraction

import numpy as np
import pytest

from vh.body import MIDAZOLAM, PBPKModel, reference_individual
from vh.graph import Graph
from vh.graph.physiology_graph import build_pbpk_graph
from vh.graph.render import to_mermaid


@pytest.mark.parametrize("bp", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("sex", ["male", "female"])
def test_export_obeys_hepatic_mass_balance_for_nonunity_blood_plasma_ratio(bp, sex):
    # Invented fu and CLint test the algebra; they are not fitted biological data.
    ind = reference_individual(sex)
    model = PBPKModel(MIDAZOLAM, ind, bp=bp, fu=0.2, clint_hepatic_l_h=40.0)
    graph, schema = build_pbpk_graph(model)
    edges = {e.attrs["organ"]: e for e in graph.edges_of_type("clears")}

    # Exact rational steady-state organ balance at input plasma concentration
    # 2 mg/L: Q*(C_b,in-C_b,out) = CLint*fu/bp*C_b,out.
    # This oracle uses original inputs, never DerivedParams clearance properties.
    q = Fraction(str(ind.liver_flow_total))
    ratio = Fraction(str(bp))
    intrinsic = Fraction(40) * Fraction("0.2") / ratio
    blood_in = ratio * 2
    blood_out = q * blood_in / (q + intrinsic)
    hepatic_rate_mg_h = q * (blood_in - blood_out)
    renal_rate_mg_h = Fraction(str(ind.gfr)) * Fraction("0.2") * 2
    assert edges["liver"].attrs["cl"] * 2 == pytest.approx(float(hepatic_rate_mg_h), rel=0, abs=1e-12)
    assert edges["kidney"].attrs["cl"] * 2 == pytest.approx(float(renal_rate_mg_h), rel=0, abs=1e-12)
    assert all(e.attrs["units"] == "L/h" and e.attrs["reference_basis"] == "plasma"
               for e in edges.values())
    assert graph.validate(schema) == []
    # Serialization and partial graphs must retain the explicit reference basis.
    restored = Graph.from_dict(graph.to_dict()).subgraph(["liver", "kidney", "eliminated"])
    assert [e.attrs for e in restored.edges] == [e.attrs for e in edges.values()]
    assert "(plasma)" in to_mermaid(restored)


@pytest.mark.parametrize("attr,value", [
    ("units", "mL/min"), ("reference_basis", "blood"),
    ("cl", -1.0), ("cl", float("inf")), ("cl", float("nan")),
    ("cl", True), ("cl", "1"), ("cl", 1 + 1j), ("cl", None), ("cl", 10**1000),
])
def test_clearance_invariant_rejects_unsafe_aggregation(attr, value):
    graph, schema = build_pbpk_graph(PBPKModel(MIDAZOLAM))
    graph.edges_of_type("clears")[0].attrs[attr] = value
    assert any("[clearance-reference]" in error for error in graph.validate(schema))


@pytest.mark.parametrize("attr", ["cl", "units", "reference_basis", "clearance_kind", "source"])
def test_clearance_schema_requires_semantics(attr):
    graph, schema = build_pbpk_graph(PBPKModel(MIDAZOLAM))
    del graph.edges_of_type("clears")[0].attrs[attr]
    assert any(f"missing attribute '{attr}'" in error for error in graph.validate(schema))


@pytest.mark.parametrize("bp", [0.5, 1.0, 2.0])
def test_reporting_sum_is_not_misidentified_as_exact_systemic_clearance(bp):
    ind = reference_individual("male")
    model = PBPKModel(MIDAZOLAM, ind, bp=bp, fu=0.2, clint_hepatic_l_h=40.0)
    graph, _ = build_pbpk_graph(model)
    reporting_sum = sum(e.attrs["cl"] for e in graph.edges_of_type("clears"))
    assert reporting_sum == pytest.approx(model.p.cl_total_plasma_l_h, rel=0, abs=1e-12)

    # Integral of the linear ODE: M * integral(x dt) = -x0 (1 mg IV bolus).
    active = [i for i, name in enumerate(model.names) if name in model.V]
    x0 = np.zeros(len(active))
    venous = active.index(model.ix["venous"])
    x0[venous] = 1.0
    integrated = -np.linalg.solve(model.M[np.ix_(active, active)], x0)
    dose_over_auc = model.V["venous"] * bp / integrated[venous]

    # Independent exact rational organ balances. The renal filtration parameter
    # must be flow-limited before adding it to effective hepatic clearance.
    ratio = Fraction(str(bp))
    q_h = Fraction(str(ind.liver_flow_total)) * ratio
    q_r = Fraction(str(ind.flows["kidney"])) * ratio
    intrinsic_h = Fraction(40) * Fraction("0.2")
    filtration = Fraction(str(ind.gfr)) * Fraction("0.2")
    effective_h = q_h * intrinsic_h / (q_h + intrinsic_h)
    effective_r = q_r * filtration / (q_r + filtration)
    assert dose_over_auc == pytest.approx(float(effective_h + effective_r), rel=0, abs=1e-10)
    # Preserve the distinction in exported quantities, without changing the ODE.
    assert reporting_sum - dose_over_auc == pytest.approx(float(filtration - effective_r), rel=0, abs=1e-10)
    assert graph.edges_of_type("clears")[1].attrs["clearance_kind"] == "nominal_renal_filtration"
