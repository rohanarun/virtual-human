"""Interface failures must stop execution; exact mass balance checks conversion.

All numbers and contexts below are artificial test fixtures, not patient or
clinical observations. They validate dispatch and arithmetic, not physiology.
"""

from dataclasses import replace
from fractions import Fraction

import pytest

from vh.graph import ContractError, Edge, Graph, ModelRegistry, ModelSpec, PortSpec, Quantity


CONTEXT = {"system": "synthetic", "compound": "test-solute"}
SOURCE = ("synthetic fixture; no experimental measurement",)
BLOOD = PortSpec("clearance", "L/h", "blood", CONTEXT, minimum=0)
PLASMA = PortSpec("clearance", "L/h", "plasma", CONTEXT, minimum=0)
RATIO = PortSpec("concentration_ratio", "1", "blood/plasma", CONTEXT, minimum=0.1, maximum=2)
EDGE = Edge("blood", "plasma", "converts", model="synthetic.clearance_to_plasma")


def quantity(value, spec=BLOOD, provenance=SOURCE):
    return Quantity(value, spec.quantity, spec.unit, spec.basis, spec.context, provenance)


def conversion_registry(compute=None):
    calls = []

    def convert(inputs):
        calls.append(inputs)
        return {"plasma": quantity(inputs["blood"].value * inputs["ratio"].value, PLASMA)}

    registry = ModelRegistry()
    registry.register(ModelSpec(
        key=EDGE.model,
        inputs={"blood": BLOOD, "ratio": RATIO},
        outputs={"plasma": PLASMA},
        compute=compute or convert,
        domain="Artificial scalar mass balance; ratio in [0.1, 2]. No physiological claim.",
        validation="Independent elimination-rate / plasma-concentration identity in test_model_contracts.py",
        provenance=("synthetic conversion definition: CLplasma = CLblood * Cblood/Cplasma",),
    ))
    return registry, calls


def inputs():
    return {"blood": quantity(10), "ratio": quantity(0.6, RATIO)}


@pytest.mark.parametrize("rate,blood_c,plasma_c", [
    (Fraction(3), Fraction(3, 10), Fraction(1, 2)),
    (Fraction(5), Fraction(1, 4), Fraction(1, 2)),
    (Fraction(7), Fraction(1, 2), Fraction(1, 4)),
    (Fraction(9), Fraction(3, 4), Fraction(1, 2)),
])
def test_conversion_against_independent_rate_over_concentration(rate, blood_c, plasma_c):
    # Independent authority: rate [mg/h] / Cp [mg/L] defines clearance [L/h].
    # Dispatch instead receives blood-referenced clearance and a concentration
    # ratio. Exact rational fixtures were specified without fitting parameters.
    registry, calls = conversion_registry()
    result = registry.execute(EDGE, {
        "blood": quantity(float(rate / blood_c)),
        "ratio": quantity(float(blood_c / plasma_c), RATIO),
    })["plasma"]
    assert result.value == pytest.approx(float(rate / plasma_c), abs=1e-12, rel=0)
    assert result.basis == "plasma" and result.unit == "L/h"
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [
    ("unit", "mL/min"),
    ("basis", "plasma"),
    ("quantity", "flow"),
    ("context", {"system": "synthetic", "compound": "another-solute"}),
    ("context", {"system": "synthetic"}),
    ("context", {**CONTEXT, "species": "human"}),
])
def test_semantic_mismatch_never_calls_model(field, value):
    registry, calls = conversion_registry()
    bad = inputs()
    bad["blood"] = replace(bad["blood"], **{field: value})
    with pytest.raises(ContractError, match=f"{field} mismatch"):
        registry.execute(EDGE, bad)
    assert calls == []


@pytest.mark.parametrize("bad", [
    {},
    {"blood": quantity(10)},
    {**inputs(), "unrequested": quantity(1)},
    {**inputs(), 12: quantity(1)},
    {**inputs(), "ratio": 0.6},
    None,
])
def test_missing_extra_or_untyped_inputs_never_call_model(bad):
    registry, calls = conversion_registry()
    with pytest.raises(ContractError):
        registry.execute(EDGE, bad)
    assert calls == []


@pytest.mark.parametrize("key", [None, "missing", "os.system('never execute strings')"])
def test_only_explicitly_registered_models_execute(key):
    registry, calls = conversion_registry()
    with pytest.raises(ContractError, match="explicitly registered"):
        registry.execute(replace(EDGE, model=key), inputs())
    assert calls == []


@pytest.mark.parametrize("value", [0.09, 2.01])
def test_outside_declared_domain_never_calls_model(value):
    registry, calls = conversion_registry()
    bad = {**inputs(), "ratio": quantity(value, RATIO)}
    with pytest.raises(ContractError, match="minimum|maximum"):
        registry.execute(EDGE, bad)
    assert calls == []


@pytest.mark.parametrize("value", [0.1, 2])
def test_domain_bounds_are_inclusive(value):
    registry, calls = conversion_registry()
    result = registry.execute(EDGE, {**inputs(), "ratio": quantity(value, RATIO)})
    assert result["plasma"].value == 10 * value
    assert len(calls) == 1


@pytest.mark.parametrize("output", [
    {},
    {"plasma": quantity(6)},  # wrong reference basis despite a plausible number
    {"plasma": quantity(-1, PLASMA)},
    {"plasma": 6},
    {"plasma": quantity(6, PLASMA), "extra": quantity(1)},
    None,
])
def test_invalid_outputs_are_not_returned(output):
    registry, _ = conversion_registry(lambda _: output)
    with pytest.raises(ContractError):
        registry.execute(EDGE, inputs())


def test_nonfinite_computed_output_is_rejected():
    registry, _ = conversion_registry(lambda _: {"plasma": quantity(float("inf"), PLASMA)})
    with pytest.raises(ContractError, match="finite"):
        registry.execute(EDGE, inputs())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "10", None, 1j])
def test_quantity_does_not_convert_missing_or_invalid_measurements_to_numbers(value):
    with pytest.raises(ContractError, match="finite real"):
        quantity(value)


@pytest.mark.parametrize("field,value", [
    ("quantity", ""), ("unit", " "), ("basis", None), ("context", {}),
    ("context", {"species": ""}), ("provenance", ()), ("provenance", "not-a-sequence"),
])
def test_missing_metadata_is_rejected(field, value):
    with pytest.raises(ContractError):
        replace(quantity(1), **{field: value})


def test_output_keeps_model_and_all_input_provenance_without_duplicates():
    registry, _ = conversion_registry()
    result = registry.execute(EDGE, {
        "blood": quantity(10, provenance=("synthetic clearance record A",)),
        "ratio": quantity(0.6, RATIO, ("synthetic ratio record B",)),
    })["plasma"]
    assert "synthetic clearance record A" in result.provenance
    assert "synthetic ratio record B" in result.provenance
    assert "model:synthetic.clearance_to_plasma" in result.provenance
    assert len(result.provenance) == len(set(result.provenance))


def test_specs_and_inputs_snapshot_metadata_and_do_not_allow_callback_mutation():
    context = dict(CONTEXT)
    spec = replace(BLOOD, context=context)
    value = quantity(1, spec)
    context["compound"] = "changed-after-construction"
    assert spec.context["compound"] == value.context["compound"] == "test-solute"
    with pytest.raises(TypeError):
        spec.context["compound"] = "mutated"

    def mutate(inputs):
        inputs["blood"] = quantity(999)

    registry, _ = conversion_registry(mutate)
    with pytest.raises(TypeError):
        registry.execute(EDGE, inputs())


def test_duplicate_registration_is_not_silent_replacement():
    model = ModelSpec("test", {"x": BLOOD}, {"y": BLOOD}, lambda x: {"y": x["x"]},
                      "synthetic identity", "identity law", SOURCE)
    registry = ModelRegistry()
    registry.register(model)
    with pytest.raises(ContractError, match="already registered"):
        registry.register(model)


def test_loading_graph_preserves_key_without_executing_or_serializing_callable():
    registry, calls = conversion_registry()
    graph = Graph()
    graph.add_node("blood", "reference")
    graph.add_node("plasma", "reference")
    graph.add_edge("blood", "plasma", "converts", model=EDGE.model)
    loaded = Graph.from_dict(graph.to_dict())
    assert calls == []
    assert registry.execute(loaded.edges[0], inputs())["plasma"].value == 6


def test_invalid_numeric_bounds_are_rejected():
    with pytest.raises(ContractError, match="minimum must not exceed"):
        replace(RATIO, minimum=3)
    with pytest.raises(ContractError, match="finite"):
        replace(RATIO, maximum=float("inf"))


def test_finite_integer_outside_float_range_raises_contract_error():
    with pytest.raises(ContractError, match="outside the supported finite scalar range"):
        quantity(10 ** 1000)
    with pytest.raises(ContractError, match="outside the supported finite scalar range"):
        replace(RATIO, maximum=10 ** 1000)
