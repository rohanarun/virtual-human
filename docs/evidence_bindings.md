# Evidence at an executable scalar boundary

`vh.graph.evidence` adds an optional checked path from an explicitly quantified
record to `ModelRegistry`. It does not read papers, infer model parameters, or
certify an observation's truth or applicability. The original scalar registry
remains available; this is an opt-in interface, not a security enforcement layer.

`EvidenceInput` holds a record ID, status, an optional `Quantity`, source, exact
locator, limits, and `Uncertainty`. The quantity retains its complete context,
quantity name, unit, reference basis, and prior provenance. These fields must be
chosen and reviewed by the caller. A `recorded` status means a recorded estimate,
not a validated model, causal effect, or clinical finding.

- `recorded` estimates may bind when every `PortSpec` field and bound matches.
- `synthetic` estimates additionally require `allow_synthetic=True`.
- `conjecture`, `missing`, and `conflicting` records cannot bind. Missing and
  conflicting records carry no selected point estimate. The caller must resolve
  evidence outside this interface; changing a label alone is not validation.
- Qualitative assertions have no scalar representation here. Protein presence,
  antigen presentation, receptor recognition, and neural injury cannot substitute
  for one another. Asma owns those assertion distinctions and evidence rules.

`execute_evidence(registry, edge, evidence, specs)` first binds every input and
then delegates to the registry, which independently rechecks the registered model
contract. Supplying weaker binding specifications cannot override that contract.
A rejected input never invokes the registered callback. As with the registry,
callbacks should be pure: output validation cannot undo callback side effects.

Integer audit values retain exact JSON integers; rational values retain tagged
numerator/denominator pairs. Other real scalar values must convert losslessly to
a JSON float or binding fails before execution. Consumers must preserve these
representations when decoding the audit.

The complete input audit is serialized under an `evidence:` provenance entry and
therefore accompanies every returned output through the registry's existing
lineage union. Returned values are nominal scenario outputs. The recorded input
uncertainty is not output uncertainty. A source CI, IQR, or unquantified range is
neither a probability distribution nor an asserted hard domain; there is no
sampling, independence assumption, interval propagation, or automatic conversion.

## Reproduce a positive example and the current missing-input result

```bash
PYTHONPATH=. python3 scripts/validate_evidence_bindings.py
python3 -m pytest -q tests/test_evidence_bindings.py
```

The script executes three explicitly artificial clearance scenarios, independently
checked against exact rational rate/concentration identities, and prints a
deterministic JSON report. It also records the exact ovarian ledger hash and the
complete rows for four necessary inputs: growth, surgical retention, carboplatin
clearance, and paclitaxel volume/clearance. All four are currently missing. This
list is not a complete disease-model specification. The readiness checker never
activates a model: a future numeric ledger entry still needs an explicit reviewed
semantic binding. Existing prevalence values and trial hazard ratios do not
supply those parameters.

No source data were acquired or re-curated. The ledger's `KNOWN` label describes
its source-access convention and must not be translated into model validity.
The [research report](../auto-research/2026-11-06-E-evidence-bindings/report.md)
records the bounded software result and remaining validation work.
