"""Offline nominal synthetic mass balance and ovarian kinetic-input readiness.

No ledger value is converted, imported into a model, or scientifically validated.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from fractions import Fraction

from vh.graph import Edge, ModelRegistry, ModelSpec, PortSpec, Quantity
from vh.graph.evidence import EvidenceInput, Uncertainty, execute_evidence

ROOT = Path(__file__).resolve().parents[1]


def synthetic_check():
    context = {"system": "artificial mass-balance fixture"}
    specs = {"rate": PortSpec("elimination_rate", "mg/h", "eliminated", context, minimum=0),
             "concentration": PortSpec("concentration", "mg/L", "plasma", context, minimum=0.01)}
    output = PortSpec("clearance", "L/h", "plasma", context, minimum=0)
    def quantity(value, spec):
        return Quantity(value, spec.quantity, spec.unit, spec.basis, spec.context, ("artificial fixture",))
    registry = ModelRegistry()
    registry.register(ModelSpec("synthetic.clearance", specs, {"clearance": output},
        lambda values: {"clearance": quantity(values["rate"].value / values["concentration"].value, output)},
        "Artificial scalar mass balance only", "Exact rational rate/concentration identity",
        ("synthetic definition",)))
    errors = []
    for rate, concentration in ((3, Fraction(1, 2)), (7, Fraction(1, 4)), (9, Fraction(3, 4))):
        records = {name: EvidenceInput(name, "synthetic", quantity(float(value), specs[name]),
            "synthetic fixture", "validate_evidence_bindings.py:synthetic_check", "Not a biological measurement",
            Uncertainty("not_quantified", "Fixed artificial input; no sampling interpretation"))
            for name, value in (("rate", rate), ("concentration", concentration))}
        result = execute_evidence(registry, Edge("a", "b", "converts", model="synthetic.clearance"),
                                  records, specs, allow_synthetic=True)["clearance"]
        errors.append(abs(result.value - float(Fraction(rate) / concentration)))
    return {"cases": len(errors), "maximum_absolute_error_L_per_h": max(errors),
            "tolerance_L_per_h": 1e-12, "passed": max(errors) <= 1e-12,
            "interpretation": "Nominal synthetic arithmetic only; no output uncertainty or biological validation"}


def readiness(ledger):
    entries = ledger["entries"]
    by_id = {row["id"]: row for row in entries}
    if len(by_id) != len(entries):
        raise ValueError("duplicate ledger IDs")
    required = {"P015": "tumor_growth", "P016": "surgical_retention", "P019": "carboplatin_clearance",
                "P020": "paclitaxel_volume_and_clearance"}
    rows = []
    for identifier, role in required.items():
        row = by_id.get(identifier)
        rows.append({"record_id": identifier, "required_role": role, "ready": False,
            "reason": "record absent" if row is None else (
                "missing estimate" if row["label"] == "MISSING" or row["value"] is None else
                "explicit reviewed quantity/unit/basis/context binding required"),
            "ledger_record": row})
    return {"ready": False, "requirements": rows,
            "interpretation": "Readiness for these four necessary inputs only, not a complete model specification. "
                              "KNOWN describes source access, not applicability; prevalence and hazard ratios are not kinetic rates."}


def main():
    path = ROOT / "data/clinical/ovarian_hgsoc_parameter_ledger.json"
    content = path.read_bytes()
    result = {"schema_version": 1, "ledger_sha256": hashlib.sha256(content).hexdigest(),
              "synthetic_validation": synthetic_check(), "ovarian_readiness": readiness(json.loads(content))}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["synthetic_validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
