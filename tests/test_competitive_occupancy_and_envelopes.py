"""Tests for competitive two-ligand groove occupancy, Motulsky-Mahan kinetics,
shared-core non-identifiability, and deterministic evidence corner envelopes.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vh.bridge import (
    competitive_equilibrium_occupancy,
    competitive_kinetic_occupancy,
    equilibrium_occupancy,
    kinetic_occupancy,
    shared_core_identifiability_locus,
)
from vh.graph import (
    ContractError,
    Edge,
    EvidenceInput,
    ModelRegistry,
    ModelSpec,
    PortSpec,
    Quantity,
    Uncertainty,
    execute_evidence_envelope,
)


def test_competitive_equilibrium_matches_exact_rationals_and_single_ligand_limit():
    res = competitive_equilibrium_occupancy(30.0, 10.0, 20.0, 20.0)
    assert res["occupancy_a"] == pytest.approx(float(Fraction(3, 5)), abs=1e-15)
    assert res["occupancy_b"] == pytest.approx(float(Fraction(1, 5)), abs=1e-15)
    assert res["occupancy_total"] == pytest.approx(float(Fraction(4, 5)), abs=1e-15)
    assert res["free_receptor_fraction"] == pytest.approx(float(Fraction(1, 5)), abs=1e-15)

    # Zero competitor B reduces identically to single-ligand Hill-Langmuir occupancy
    c_grid = np.array([0.0, 2.5, 10.0, 40.0])
    comp0 = competitive_equilibrium_occupancy(c_grid, 10.0, np.zeros_like(c_grid), 25.0)
    single = equilibrium_occupancy(c_grid, 10.0)
    np.testing.assert_allclose(comp0["occupancy_a"], single, rtol=0, atol=1e-15)
    np.testing.assert_allclose(comp0["occupancy_b"], 0.0, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "ca,kda,cb,kdb",
    [
        (-1.0, 10.0, 5.0, 10.0),
        (1.0, 0.0, 5.0, 10.0),
        (1.0, 10.0, -0.1, 10.0),
        (1.0, 10.0, 5.0, -2.0),
        (float("nan"), 10.0, 5.0, 10.0),
        (1.0, float("inf"), 5.0, 10.0),
    ],
)
def test_competitive_equilibrium_rejects_invalid_inputs(ca, kda, cb, kdb):
    with pytest.raises(ValueError):
        competitive_equilibrium_occupancy(ca, kda, cb, kdb)


def test_shared_core_identifiability_locus_preserves_total_occupancy():
    target = 0.75
    kda, kdb = 10.0, 25.0
    for cu_a in (0.0, 5.0, 12.5, 20.0, 30.0):
        cu_b = shared_core_identifiability_locus(target, kda, kdb, cu_a)
        res = competitive_equilibrium_occupancy(cu_a, kda, cu_b, kdb)
        assert res["occupancy_total"] == pytest.approx(target, abs=1e-15)

    # Exceeding target_total_occupancy with ligand A alone must fail
    with pytest.raises(ValueError, match="exceeds"):
        shared_core_identifiability_locus(0.50, 10.0, 25.0, 15.0)
    with pytest.raises(ValueError):
        shared_core_identifiability_locus(1.0, 10.0, 25.0, 0.0)


def test_competitive_kinetics_matches_dop853_and_captures_transient_overshoot():
    t = np.linspace(0.0, 25.0, 251)
    cu_a, kd_a, koff_a = 30.0, 10.0, 0.2
    cu_b, kd_b, koff_b = 40.0, 40.0, 4.0
    mm = competitive_kinetic_occupancy(t, cu_a, kd_a, koff_a, cu_b, kd_b, koff_b, oa0=0.0, ob0=0.0)

    kon_a, kon_b = koff_a / kd_a, koff_b / kd_b
    sol = solve_ivp(
        lambda _t, y: [
            kon_a * cu_a * (1.0 - y[0] - y[1]) - koff_a * y[0],
            kon_b * cu_b * (1.0 - y[0] - y[1]) - koff_b * y[1],
        ],
        (0.0, 25.0),
        [0.0, 0.0],
        t_eval=t,
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
        max_step=0.02,
    )
    assert sol.success
    np.testing.assert_allclose(mm["occupancy_a"], sol.y[0], rtol=0, atol=1e-13)
    np.testing.assert_allclose(mm["occupancy_b"], sol.y[1], rtol=0, atol=1e-13)

    # Fast competitor B transiently overshoots >2x its equilibrium occupancy (0.20)
    eq = competitive_equilibrium_occupancy(cu_a, kd_a, cu_b, kd_b)
    assert np.max(mm["occupancy_b"]) > 2.0 * eq["occupancy_b"]

    # Zero competitor B matches single-ligand kinetic_occupancy exactly, including equal-koff degenerate branch
    single = kinetic_occupancy(t, np.full_like(t, cu_a), kd_a, koff_a, o0=0.1)
    comp_zero = competitive_kinetic_occupancy(t, cu_a, kd_a, koff_a, 0.0, kd_a, koff_a, oa0=0.1, ob0=0.0)
    np.testing.assert_allclose(comp_zero["occupancy_a"], single, rtol=0, atol=1e-14)
    np.testing.assert_allclose(comp_zero["occupancy_b"], 0.0, rtol=0, atol=1e-15)


def test_execute_evidence_envelope_computes_exact_corners_and_threshold_classifications():
    ctx = {"system": "competitive_groove"}
    in_specs = {
        "cu_a": PortSpec("concentration", "nM", "unbound", ctx, minimum=0.0),
        "kd_a": PortSpec("dissociation_constant", "nM", "unbound", ctx, minimum=0.1),
        "cu_b": PortSpec("concentration", "nM", "unbound", ctx, minimum=0.0),
        "kd_b": PortSpec("dissociation_constant", "nM", "unbound", ctx, minimum=0.1),
    }
    out_specs = {
        "occ_a": PortSpec("occupancy_fraction", "1", "bound", ctx, minimum=0.0, maximum=1.0),
        "occ_total": PortSpec("occupancy_fraction", "1", "bound", ctx, minimum=0.0, maximum=1.0),
    }
    calls = []

    def model(inputs):
        calls.append(dict(inputs))
        res = competitive_equilibrium_occupancy(
            inputs["cu_a"].value, inputs["kd_a"].value, inputs["cu_b"].value, inputs["kd_b"].value
        )
        return {
            "occ_a": Quantity(res["occupancy_a"], "occupancy_fraction", "1", "bound", ctx, ("m",)),
            "occ_total": Quantity(res["occupancy_total"], "occupancy_fraction", "1", "bound", ctx, ("m",)),
        }

    reg = ModelRegistry()
    reg.register(ModelSpec("gaddum", in_specs, out_specs, model, "domain", "val", ("src",)), model)
    edge = Edge("pool", "groove", "bind", model="gaddum")

    ev = {
        "cu_a": EvidenceInput(
            "R_CUA", "recorded", Quantity(30.0, "concentration", "nM", "unbound", ctx, ("p",)),
            "S", "L", "lim", Uncertainty("bounds", "range", 15.0, 60.0),
        ),
        "kd_a": EvidenceInput(
            "R_KDA", "recorded", Quantity(10.0, "dissociation_constant", "nM", "unbound", ctx, ("p",)),
            "S", "L", "lim", Uncertainty("bounds", "range", 5.0, 20.0),
        ),
        "cu_b": EvidenceInput(
            "R_CUB", "recorded", Quantity(20.0, "concentration", "nM", "unbound", ctx, ("p",)),
            "S", "L", "lim", Uncertainty("bounds", "range", 10.0, 40.0),
        ),
        "kd_b": EvidenceInput(
            "R_KDB", "recorded", Quantity(20.0, "dissociation_constant", "nM", "unbound", ctx, ("p",)),
            "S", "L", "lim", Uncertainty("bounds", "point", 20.0, 20.0),
        ),
    }

    env = execute_evidence_envelope(
        reg, edge, ev, in_specs, thresholds={"occ_a": 0.50, "occ_total": 0.99}
    )
    # 1 nominal call + 8 unique corner calls = 9 model invocations
    assert len(calls) == 9
    assert env["occ_a"].corner_count == 8
    assert env["occ_a"].nominal_within_corners is True
    assert env["occ_a"].nominal.value == pytest.approx(0.60, abs=1e-15)
    assert env["occ_a"].envelope_low.value == pytest.approx(0.20, abs=1e-15)
    assert env["occ_a"].envelope_high.value == pytest.approx(float(Fraction(8, 9)), abs=1e-15)
    assert env["occ_a"].threshold_status == "straddles_threshold"
    assert env["occ_total"].threshold_status == "below_across_envelope"


def test_execute_evidence_envelope_fails_closed_before_calling_model():
    ctx = {"system": "in_vitro"}
    spec = PortSpec("rate_constant", "1/h", "unbound_plasma", ctx, minimum=0.0, maximum=10.0)
    calls = []
    reg = ModelRegistry()
    reg.register(
        ModelSpec("pass", {"in": spec}, {"out": spec}, lambda x: calls.append(x) or {"out": x["in"]}, "d", "v", ("s",)),
        lambda x: calls.append(x) or {"out": x["in"]},
    )
    edge = Edge("a", "b", "r", model="pass")
    q = Quantity(2.0, "rate_constant", "1/h", "unbound_plasma", ctx, ("src",))

    bad_records = [
        # Missing uncertainty endpoints
        EvidenceInput("E1", "recorded", q, "S", "L", "lim", Uncertainty("unquantified", "none")),
        # Estimate outside [low, high]
        EvidenceInput("E2", "recorded", q, "S", "L", "lim", Uncertainty("CI95", "shifted", 2.5, 4.0)),
        # Low endpoint below PortSpec.minimum
        EvidenceInput("E3", "recorded", q, "S", "L", "lim", Uncertainty("CI95", "negative low", -0.5, 4.0)),
        # High endpoint above PortSpec.maximum
        EvidenceInput("E4", "recorded", q, "S", "L", "lim", Uncertainty("CI95", "excess high", 1.0, 12.0)),
        # Lossy Decimal endpoint
        EvidenceInput("E5", "recorded", q, "S", "L", "lim", Uncertainty("CI95", "decimal", Decimal("1.1"), 3.0)),
    ]
    for bad in bad_records:
        with pytest.raises(ContractError):
            execute_evidence_envelope(reg, edge, {"in": bad}, {"in": spec})
    assert calls == []

    # Unknown threshold output port rejected
    good = EvidenceInput("E_OK", "recorded", q, "S", "L", "lim", Uncertainty("CI95", "ok", 1.0, 3.0))
    with pytest.raises(ContractError, match="unknown output port"):
        execute_evidence_envelope(reg, edge, {"in": good}, {"in": spec}, thresholds={"missing_port": 1.5})
