# Explicit evidence bindings for nominal scalar experiments

- **Track:** E, reusable graph/model interfaces
- **Author:** gpt-forge (OpenAI Codex, Astra), under @ahrazzle's direction
- **Date:** 2026-11-06
- **Branch:** `research/evidence-bindings-20261106`
- **Code:** `vh/graph/evidence.py`, `scripts/validate_evidence_bindings.py`, `tests/test_evidence_bindings.py`

## Question

Can a scalar experiment retain the evidence's complete context, source locator,
limits, and uncertainty while refusing unsupported inputs before model execution?
Can the same path perform an independently checkable positive synthetic example?

## Method

An opt-in immutable evidence record binds a fully specified Quantity to a
PortSpec. Recorded estimates and explicitly enabled synthetic fixtures can bind;
conjectures, unresolved conflicts and missing values cannot. Every supplied
quantity/unit/basis/context must match exactly. No conversions, semantic aliases,
parameter inference or evidence joins are implemented. The model registry checks
its own contract again before dispatch. Structured input audit records travel
through the registry's existing provenance lineage.

Uncertainty descriptions and optional paired endpoints are preserved. They are
not sampled or propagated. Outputs are nominal values; no confidence or
prediction interval is claimed. Synthetic fixtures remain artificial. No
parameters were fitted and no biological observations were used as a referee.

## Validation

The independent mathematical authority is the exact rational identity relating
elimination rate, plasma concentration and clearance. Three synthetic cases use
rate/concentration pairs (3, 1/2), (7, 1/4), and (9, 3/4) in mg/h and mg/L.
The executable floating-point model produces 6, 28 and 12 L/h respectively.

| Check | Result | Authority | Error / tolerance |
| --- | --- | --- | --- |
| Three nominal clearances | 6, 28, 12 L/h | Exact rational rate / concentration | Maximum 0 L/h; tolerance 1e-12 L/h |
| Missing kinetic inputs | Four refused | Checked-in ledger P015, P016, P019, P020 | All four lack estimates; software consistency check only |
| Interface tests | 33 new passing cases | Explicit boundary examples and exact arithmetic | No statistical accuracy claim |

The arithmetic error has no statistical error bar: these are fixed synthetic
fixtures, not sampled observations. Source uncertainty remains attached to the
inputs. Tests also check refusal before callback execution, incompatible
contexts/units/bases, immutable context snapshots, complete audit retention,
explicit synthetic permission, and inability to weaken registered contracts.

The CLI output in `results.json` pins the exact ledger bytes with SHA-256 and
preserves the full selected source rows. Those rows have not been re-verified
against literature in this contribution. The result is a reproducible missing
input finding, not an independent scientific validation of their content.

## Limits

A caller can mislabel an input or omit a relevant context field. This interface
cannot discover that mistake or certify scientific adequacy. It is opt-in and
does not prevent use of the underlying registry. CI/IQR endpoints are preserved
metadata, not hard bounds; only nominal values are checked against port bounds.
No output uncertainty or probability distribution is produced. Asma's qualitative
presence, presentation, recognition and injury assertions remain domain-owned;
they cannot be coerced to scalars through this interface.

The four ovarian requirements are necessary examples, not a sufficient model
specification. A future non-null ledger row still needs reviewed explicit semantic
binding; the readiness script deliberately never authorizes execution. There is
no patient simulation, clinical inference, data acquisition, or treatment advice.
Independent scientific review and held-out biological outcomes remain absent.

## Reproduce

```bash
PYTHONPATH=. python3 scripts/validate_evidence_bindings.py
python3 -m pytest -q tests/test_evidence_bindings.py tests/test_model_contracts.py tests/test_reaction_experiments.py tests/test_ovarian_parameter_ledger.py -m 'not network'
node web/test_engines.mjs
```

Observed environment: Python 3.14.6, pytest 9.1.1, NumPy 2.5.4, SciPy 1.18.1.
Focused pre-serialization check: 129 passed, 1 network test deselected; evidence-only suite before independent-review correction: 32 passed. JavaScript engine checks
passed against checked-in fixtures, including worst plasma-profile relative error
4.8e-13. Final full local suite: 218 passed, 15 skipped, 1 network test deselected. Skips reflect unavailable OpenMM, PySCF and libRoadRunner; those independent-engine checks were not run. No dependency installation was performed.

## References

- Existing `vh/graph/contracts.py` interfaces and exact clearance identity tests.
- `data/clinical/ovarian_hgsoc_parameter_ledger.json`, unchanged source ledger.
- `docs/research-foundation.md` and `docs/research-question-template.md` for the
  separation of evidence assertions, modeling assumptions and scientific validation.

Independent software review found an audit precision defect for large integers.
The correction preserves JSON integers and tagged exact rationals and rejects
other non-lossless representations before dispatch. A large-integer regression
verifies the executed value and recorded endpoints agree exactly. Final suite
after this correction: 218 passed, 15 skipped, 1 network test deselected.
