"""Synthetic model-dispatch benchmark; no clinical or fitted quantities.

Run with PYTHONPATH=. python scripts/validate_model_contracts.py.
"""

import json
from fractions import Fraction

from vh.graph import Edge, ModelRegistry, ModelSpec, PortSpec, Quantity


def validate():
    context = {"system": "synthetic", "compound": "test-solute"}
    source = ("artificial mass-balance fixture; no experimental measurement",)
    blood = PortSpec("clearance", "L/h", "blood", context, minimum=0)
    plasma = PortSpec("clearance", "L/h", "plasma", context, minimum=0)
    ratio = PortSpec("concentration_ratio", "1", "blood/plasma", context, minimum=0.1, maximum=2)

    def q(value, spec):
        return Quantity(value, spec.quantity, spec.unit, spec.basis, spec.context, source)

    registry = ModelRegistry()
    registry.register(ModelSpec(
        "synthetic.clearance_to_plasma", {"blood": blood, "ratio": ratio}, {"plasma": plasma},
        lambda values: {"plasma": q(values["blood"].value * values["ratio"].value, plasma)},
        domain="Artificial scalar mass balance; ratio in [0.1, 2]. No physiological claim.",
        validation="Check against exact elimination rate divided by plasma concentration.",
        provenance=("definition of clearance and concentration ratio",),
    ))
    edge = Edge("blood", "plasma", "converts", model="synthetic.clearance_to_plasma")
    fixtures = [
        (Fraction(3), Fraction(3, 10), Fraction(1, 2)),
        (Fraction(5), Fraction(1, 4), Fraction(1, 2)),
        (Fraction(7), Fraction(1, 2), Fraction(1, 4)),
        (Fraction(9), Fraction(3, 4), Fraction(1, 2)),
    ]
    rows = []
    for rate, blood_c, plasma_c in fixtures:
        actual = registry.execute(edge, {
            "blood": q(float(rate / blood_c), blood),
            "ratio": q(float(blood_c / plasma_c), ratio),
        })["plasma"].value
        expected = float(rate / plasma_c)  # independent exact mass-balance authority
        rows.append({"rate_mg_h": str(rate), "blood_concentration_mg_l": str(blood_c),
                     "plasma_concentration_mg_l": str(plasma_c),
                     "computed_clearance_l_h": actual, "analytic_clearance_l_h": expected,
                     "absolute_error_l_h": abs(actual - expected)})
    maximum_error = max(row["absolute_error_l_h"] for row in rows)
    if maximum_error > 1e-12:
        raise AssertionError(f"mass-balance error {maximum_error} L/h exceeds tolerance")
    return {"fixture_type": "synthetic; no experimental uncertainty or parameter fitting",
            "authority": "exact elimination rate / plasma concentration; rational input fixtures",
            "tolerance_l_h": 1e-12, "max_absolute_error_l_h": maximum_error, "cases": rows}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
