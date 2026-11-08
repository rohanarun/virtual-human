"""Boundary checks and exact synthetic arithmetic; no biological validation."""
from dataclasses import replace
from fractions import Fraction
import json
import runpy
from pathlib import Path

import pytest

from vh.graph import ContractError, Edge, ModelRegistry, ModelSpec, PortSpec, Quantity
from vh.graph.evidence import EvidenceInput, Uncertainty, execute_evidence

SPEC = PortSpec("concentration", "mol/L", "free", {"system": "synthetic"}, minimum=0)
EDGE = Edge("a", "b", "converts", model="identity")


def record(status="recorded"):
    return EvidenceInput("fixture-1", status,
        Quantity(2.0, SPEC.quantity, SPEC.unit, SPEC.basis, SPEC.context, ("fixture",)),
        "artificial source", "table 1, row 1", "No biological validity",
        Uncertainty("CI95", "Artificial confidence interval, not a distribution", 1.0, 3.0))


def registry():
    calls = []
    def compute(values):
        calls.append(values)
        return {"out": values["in"]}
    result = ModelRegistry()
    result.register(ModelSpec("identity", {"in": SPEC}, {"out": SPEC}, compute,
                              "synthetic", "identity", ("synthetic identity",)))
    return result, calls


def test_complete_audit_survives_execution_and_original_context_mutation():
    source_context = {"system": "synthetic"}
    evidence = replace(record(), estimate=replace(record().estimate, context=source_context))
    source_context["system"] = "changed"
    models, calls = registry()
    result = execute_evidence(models, EDGE, {"in": evidence}, {"in": SPEC})["out"]
    audit = json.loads(next(s[len("evidence:"):] for s in result.provenance if s.startswith("evidence:")))
    assert audit == evidence.audit()
    assert audit["uncertainty"]["kind"] == "CI95"
    assert audit["estimate"]["context"] == {"system": "synthetic"}
    assert result.value == 2.0
    assert len(calls) == 1


@pytest.mark.parametrize("status", ["conjecture", "synthetic", "missing", "conflicting"])
def test_unsupported_status_never_calls_model(status):
    item = record() if status not in ("missing", "conflicting") else replace(record(), status=status, estimate=None)
    item = replace(item, status=status)
    models, calls = registry()
    with pytest.raises(ContractError, match="cannot bind"):
        execute_evidence(models, EDGE, {"in": item}, {"in": SPEC})
    assert calls == []


def test_synthetic_permission_is_explicit_and_does_not_allow_conjecture():
    assert record("synthetic").bind(SPEC, allow_synthetic=True).value == 2
    with pytest.raises(ContractError):
        record("conjecture").bind(SPEC, allow_synthetic=True)
    with pytest.raises(ContractError):
        record("synthetic").bind(SPEC, allow_synthetic="yes")


@pytest.mark.parametrize("field,value", [("quantity", "hazard_ratio"), ("unit", "mg/L"),
    ("basis", "total"), ("context", {"system": "human"}), ("value", -1)])
def test_wrong_semantics_never_calls_model(field, value):
    evidence = replace(record(), estimate=replace(record().estimate, **{field: value}))
    models, calls = registry()
    with pytest.raises(ContractError):
        execute_evidence(models, EDGE, {"in": evidence}, {"in": SPEC})
    assert calls == []


def test_registered_specs_cannot_be_overridden_by_binding_specs():
    other = replace(SPEC, basis="total")
    evidence = replace(record(), estimate=replace(record().estimate, basis="total"))
    models, calls = registry()
    with pytest.raises(ContractError, match="basis mismatch"):
        execute_evidence(models, EDGE, {"in": evidence}, {"in": other})
    assert calls == []


@pytest.mark.parametrize("change", [{"status": "unknown"}, {"estimate": None}, {"source": ""},
    {"locator": ""}, {"limits": ""}, {"uncertainty": None}, {"estimate": 2},
    {"status": "missing"}, {"status": "conflicting"}])
def test_malformed_evidence_rejected(change):
    with pytest.raises(ContractError):
        replace(record(), **change)


@pytest.mark.parametrize("low,high", [(1, None), (None, 2), (3, 1), (True, 2), (1, float("inf")),
                                       (float("nan"), 3)])
def test_malformed_uncertainty_rejected(low, high):
    with pytest.raises(ContractError):
        Uncertainty("CI95", "reported interval", low, high)


def test_uncertainty_is_retained_not_used_as_input_bounds():
    # A CI is not a hard range: execution checks the nominal value only.
    assert replace(record(), uncertainty=Uncertainty("CI95", "source CI", -1, 3)).bind(SPEC).value == 2


def test_missing_extra_and_untyped_ports_rejected_before_dispatch():
    models, calls = registry()
    for data in ({}, {"in": record(), "extra": record()}, {"in": 2}):
        with pytest.raises(ContractError):
            execute_evidence(models, EDGE, data, {"in": SPEC})
    assert calls == []


def test_synthetic_end_to_end_against_exact_rational_identity():
    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/validate_evidence_bindings.py"))
    result = script["synthetic_check"]()
    assert result["cases"] == 3
    assert result["maximum_absolute_error_L_per_h"] == 0
    assert result["passed"]


def test_actual_ledger_cannot_supply_kinetic_inputs():
    root = Path(__file__).parents[1]
    script = runpy.run_path(str(root / "scripts/validate_evidence_bindings.py"))
    ledger = json.loads((root / "data/clinical/ovarian_hgsoc_parameter_ledger.json").read_text())
    result = script["readiness"](ledger)
    assert not result["ready"]
    assert len(result["requirements"]) == 4
    assert all(r["reason"] == "missing estimate" for r in result["requirements"])
    # A newly filled row still needs explicit semantic binding; merely changing
    # its label/value cannot authorize a simulation.
    row = next(r for r in ledger["entries"] if r["id"] == "P015")
    row.update(label="KNOWN", value=10)
    updated = script["readiness"](ledger)
    assert updated["requirements"][0]["reason"] == "explicit reviewed quantity/unit/basis/context binding required"
    assert not updated["ready"]


def test_finite_real_scalars_have_serializable_audits():
    item = replace(record(), estimate=replace(record().estimate, value=Fraction(1, 2)),
                   uncertainty=Uncertainty("bounds", "artificial", Fraction(1, 4), Fraction(3, 4)))
    bound = item.bind(SPEC)
    assert bound.value == Fraction(1, 2)
    assert item.audit()["estimate"]["value"] == {"representation": "rational", "numerator": 1, "denominator": 2}
    assert item.audit()["uncertainty"]["low"] == {"representation": "rational", "numerator": 1, "denominator": 4}


def test_audit_retains_large_integer_estimate_and_bounds_exactly():
    value = 2**53 + 1
    item = replace(record(), estimate=replace(record().estimate, value=value),
                   uncertainty=Uncertainty("bounds", "artificial exact integers", value, value + 2))
    models, calls = registry()
    result = execute_evidence(models, EDGE, {"in": item}, {"in": SPEC})["out"]
    audit = json.loads(next(s[len("evidence:"):] for s in result.provenance if s.startswith("evidence:")))
    assert audit["estimate"]["value"] == value == result.value
    assert audit["uncertainty"]["low"] == value
    assert audit["uncertainty"]["high"] == value + 2
    assert isinstance(audit["estimate"]["value"], int)
    assert len(calls) == 1
