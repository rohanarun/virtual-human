# Explicit model contracts for graph edges

- **Track**: E — incremental graph interface work; not the complete SBML milestone
- **Author**: OpenAI Codex, under the direction of [@ahrazzle](https://github.com/ahrazzle)
- **Date**: 2026-11-05
- **Branch / PR**: `auto/E-model-contracts`
- **Code added**: `vh/graph/contracts.py`, `scripts/validate_model_contracts.py`,
  `tests/test_model_contracts.py`; API guide: `docs/model_contracts.md`

## Question

`Edge.model` names a computational model, but the existing graph had no execution
registry for that key. Can an explicit dispatcher detect incompatible quantity
semantics, units, reference bases and contexts before a model consumes them?

## Method

Added a standard-library-only `ModelRegistry` and immutable `Quantity`, `PortSpec`
and `ModelSpec` declarations. Callers register trusted Python callables explicitly.
All input names, semantic quantity names, units, bases and complete context
mappings must match the registered ports. Scalars must be finite and within any
declared inclusive bounds. Missing or incompatible inputs fail before the
callable runs; incompatible outputs are not returned. Duplicate model keys fail.
Graph deserialization continues to load only a string key and never imports code.

Model domain and validation statements are required documentation, not proof.
Inputs and models require source identifiers; outputs preserve all their input
and model identifiers as audit lineage. Copies of mappings prevent accidental
mutation after declaration. There is no automatic unit conversion, dynamic import,
graph scheduler or pre-registered adapter to an existing engine.

The illustrative conversion is deliberately synthetic:
`CLplasma = CLblood * Cblood/Cplasma`. All fixture values are artificial rational
numbers specified in the validation script; no experimental observations,
patient files or fitted parameters are involved. It is an interface example, not
a new physiological model. Existing engines and their calculations are unchanged.

## Validation — the part that matters

**Independent authority:** the exact mass-balance definition
`CLplasma = elimination_rate / plasma_concentration`. The dispatcher receives
blood-referenced clearance and a concentration ratio; the comparison instead uses
elimination rate divided directly by plasma concentration, evaluated as exact
rational arithmetic before float conversion. No fitted parameters are used.

| Quantity | This work | Authority | Absolute error |
|---|---:|---:|---:|
| Synthetic case 1, plasma clearance | 6 L/h | 6 L/h | 0 L/h |
| Synthetic case 2, plasma clearance | 10 L/h | 10 L/h | 0 L/h |
| Synthetic case 3, plasma clearance | 28 L/h | 28 L/h | 0 L/h |
| Synthetic case 4, plasma clearance | 18 L/h | 18 L/h | 0 L/h |

Maximum observed absolute error is **0 L/h across these four fixtures**, with a
test tolerance of `1e-12 L/h`. An experimental error bar or confidence interval
does not apply to these exact synthetic cases. This is not a claim of zero error
for arbitrary floating-point inputs. Full fixture values and results are in
[`results.json`](results.json).

Focused verification: **60 tests passed** (`tests/test_model_contracts.py` and
`tests/test_graph.py`, comprising 50 contract cases and 10 existing graph cases).
Regression cases reject blood/plasma reference confusion, incorrect units,
semantic-quantity confusion, changed/missing/extra context, missing/extra input
ports, unregistered model keys, unsupported numeric ranges, missing provenance,
out-of-domain inputs and invalid outputs. Input failures verify that the callable
was not invoked. Tests also check inclusive bounds, immutable metadata, explicit
dispatch after graph round-trip and preservation of provenance lineage.

Full engine and browser suites are outside this focused validation run; their
status must be checked before merging alongside other changes.

**Subsequent combined run, 2026-11-05:** 111 Python tests passed and JavaScript
engine checks passed with both checked-in and regenerated reference fixtures.
See the [integration record](../../docs/project-asma-integration.md) for code
revision, exact commands, runtime and limits. This does not broaden the biological
scope of the synthetic validation above.

## Limits

Matching metadata does not establish biological validity, causal direction,
identifiability, source reliability or model calibration. Relevant conditions
must be supplied by the caller; an omitted biological variable cannot be inferred.
`domain` and `validation` descriptions cannot be automatically certified. Numerical
bounds and exact contexts provide a limited, reviewable validity check only.
Provenance strings are identifiers, not verified evidence supporting an output.

Only individual scalar model calls are supported. There is no tensor interface,
unit algebra, uncertain-value propagation, automatic conversion, cycle solver,
whole-graph scheduling or SBML importer. Existing engine edges still require
explicit adapters. Structural endpoint rules remain in `Schema`. Callables should
be pure; a rejected output cannot reverse callback side effects, and this is not
a sandbox for untrusted Python. This completes a small part of track E, not its
stated BioModels/SBML/conservation milestone.

## Reproduce

From the repository root with its development dependencies installed:

```bash
PYTHONPATH=. python scripts/validate_model_contracts.py
PYTHONPATH=. python -m pytest -q tests/test_model_contracts.py tests/test_graph.py
git diff --check
```

The recorded run used `/tmp/vh-dev/bin/python` (Python 3.12.14) with `PYTHONPATH=.`;
no paid computation or external dataset access is required. Full regression
commands before merge remain `pytest -q`, `python scripts/export_web.py` and
`node web/test_engines.mjs`, as specified by the repository CI.

## References

- Existing graph contract: [`vh/graph/core.py`](../../vh/graph/core.py),
  `Edge.model`, graph serialization and `Schema` (upstream baseline
  `3d75e8e25d6f3fec08e1ccecd424532bccaa1b82`).
- Repository contribution and validation requirements:
  [`auto-research.md`](../../auto-research.md) and
  [`docs/accuracy_ladder.md`](../../docs/accuracy_ladder.md).
- Validation authority: definition of clearance, elimination rate divided by
  the concentration in its stated reference compartment; synthetic fixtures and
  exact rational reference calculations in
  [`scripts/validate_model_contracts.py`](../../scripts/validate_model_contracts.py).
