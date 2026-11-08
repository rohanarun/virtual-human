# Executing an edge with explicit model contracts

`Edge.model` is a declarative key. A `ModelRegistry` now lets a caller explicitly
register a callable for that key and execute one edge after checking its inputs.
Loading a graph does not register, import or run any code. Existing physiology
and molecular graph builders do not automatically register their engine models.

Each `Quantity` must carry a finite real value, semantic quantity name, unit,
reference basis, nonempty context and provenance. Each `PortSpec` requires exact
matches for those semantic fields, including the complete context mapping. For
example, a blood clearance cannot enter a plasma-clearance port even though both
use L/h. A different compound, species, specimen or assay also fails when encoded
in context. Missing observations remain absent; zero is a measured or modeled
number, not a substitute for an unknown value.

Every input port must be supplied, and extra ports are rejected. Inclusive
minimum/maximum bounds provide simple numerical domain checks. No callable runs
until all inputs pass. Returned quantities must satisfy the output contracts;
outputs inherit source identifiers from the model and all inputs as an audit
trail. That lineage does not assert that a cited source supports the result.

```python
from vh.graph import Edge, ModelRegistry, ModelSpec, PortSpec, Quantity

context = {"system": "synthetic", "compound": "test-solute"}
blood = PortSpec("clearance", "L/h", "blood", context, minimum=0)
ratio = PortSpec("concentration_ratio", "1", "blood/plasma", context,
                 minimum=0.1, maximum=2)
plasma = PortSpec("clearance", "L/h", "plasma", context, minimum=0)

def quantity(value, spec):
    return Quantity(value, spec.quantity, spec.unit, spec.basis, spec.context,
                    ("artificial example; not a measurement",))

registry = ModelRegistry()
registry.register(ModelSpec(
    key="synthetic.clearance_to_plasma",
    inputs={"blood": blood, "ratio": ratio},
    outputs={"plasma": plasma},
    compute=lambda x: {"plasma": quantity(x["blood"].value * x["ratio"].value, plasma)},
    domain="Artificial example; ratio in [0.1, 2]. No physiological claim.",
    validation="Exact mass balance; scripts/validate_model_contracts.py",
    provenance=("CLplasma = CLblood * Cblood/Cplasma by definition",),
))
edge = Edge("blood", "plasma", "converts", model="synthetic.clearance_to_plasma")
result = registry.execute(edge, {"blood": quantity(10, blood), "ratio": quantity(0.6, ratio)})
assert result["plasma"].value == 6  # L/h, plasma reference
```

The conversion is explicit: the registry does not infer units, search for a
conversion path, or silently change reference bases. Four synthetic mass-balance
cases are checked against the exact identity `CLplasma = elimination_rate / Cp`.
They validate dispatch and arithmetic only. Run:

```bash
PYTHONPATH=. python scripts/validate_model_contracts.py
PYTHONPATH=. python -m pytest -q tests/test_model_contracts.py
```

`domain` and `validation` on `ModelSpec` are required documentation. Their truth
cannot be checked from strings; reviewers must assess the sources, context keys
and numerical bounds. A caller who omits a relevant biological condition can
still build an invalid scientific model. Exact string equality is intentionally
conservative: even equivalent unit spellings need a deliberate conversion or
normalization outside the registry.

This API does not infer causation from graph reachability, execute a whole graph,
solve cycles, propagate uncertainty, certify provenance, or validate clinical
predictions. Edge endpoint types and other structural invariants remain the job
of `Schema`; the registry dispatches the explicitly named model only. Callbacks
should be pure functions: rejecting an invalid output cannot undo callback side
effects. The registry is not a sandbox for untrusted Python. Contract and callable
definitions remain outside serialized graph data.

## Optional evidence bindings

For recorded source locators, uncertainty metadata, missing/conflicting input refusal,
and explicitly enabled synthetic scenarios, see [evidence bindings](evidence_bindings.md).
This opt-in layer preserves input audit records; it does not certify scientific
applicability or infer output uncertainty.
