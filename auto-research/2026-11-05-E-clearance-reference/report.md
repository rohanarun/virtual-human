# Explicit clearance references and independent elimination checks

- **Track**: E — graph invariants (partial progress; no SBML importer)
- **Author**: OpenAI Codex, under the direction of [@ahrazzle](https://github.com/ahrazzle)
- **Date**: 2026-11-05
- **Branch / PR**: `auto/E-clearance-reference`
- **Code added/changed**: `vh/graph/physiology_graph.py`, `vh/graph/render.py`,
  `scripts/validate_graph.py`, `tests/test_graph_clearance.py`

## Question

Can graph validation catch an actual inconsistency in the existing PBPK
formalization? The hepatic `clears` edge used blood clearance while the renal
edge used plasma clearance, and the validation script added them and compared
the same copied operands, incorrectly calling this independent verification of
the full model.

## Method

Preserve the PBPK ODE, parameters and existing hepatic well-stirred model.
Convert exported hepatic clearance to plasma reference using
`CL_plasma = bp * CL_blood`, which follows from conservation of elimination
rate and `C_blood = bp * C_plasma`. Both clearance edges now explicitly carry
units, reference basis, quantity kind and their source property. A new graph
invariant rejects incompatible references/units and negative or nonfinite
clearances, including missing values. Old serialized graphs require explicit
conversion; they are not silently treated as plasma-based.

The reporting sum remains the body's existing `cl_total_plasma_l_h` quantity.
The renal edge is explicitly **nominal renal filtration**, whereas the hepatic
edge is **effective well-stirred hepatic clearance**. These cannot be used as
the ODE's local tissue elimination coefficients merely because their units
match. The script now calls equality with the body property an export check.

Separately, obtain infinite-time IV plasma AUC by integrating the linear ODE:
`M_active * integral(x dt) = -x0`, for a 1 mg bolus and zero final circulating
drug. Compare dose/AUC with an independently derived steady-state organ balance.
For an organ supplied with plasma-equivalent flow `q = Q_blood * bp` and
intrinsic plasma-referenced loss `c`, that balance gives `CL = q*c/(q+c)`.
Use `c = fu*CLint` for the liver and `c = fu*GFR` for the shipped compound's
renal filtration. Inputs retain their existing provenance in `vh/body`;
no new physiological parameters or clinical datasets were introduced or fit.

The tests also use invented `fu=0.2` and `CLint=40 L/h`, blood:plasma ratios
0.5, 1 and 2, and both reference physiologies. Exact rational arithmetic solves
hepatic influx minus efflux at input plasma concentration 2 mg/L. This catches
the original bug without reading a derived clearance property as the oracle.

## Validation — the part that matters

**Authority:** exact mass balance, evaluated with rational arithmetic for the
synthetic hepatic cases and an analytic well-stirred solution for systemic
clearance. These are mathematical implementation checks, not clinical validation.

| Quantity | This work | Authority | Error |
|---|---|---|---|
| Hepatic elimination rate, six synthetic cases | exported plasma CL × 2 mg/L | exact rational organ influx minus efflux | maximum observed absolute error 0.0 mg/h; test tolerance 1e-12 mg/h |
| Reference male midazolam systemic clearance | dose/AUC = 17.103069745580346 L/h | analytic organ balance = 17.103069745580346 L/h | observed absolute error 0.0 L/h; test tolerance 1e-10 L/h |
| Three synthetic systemic clearances | exact integral of the ODE | rational hepatic + finite-flow renal clearance | all absolute errors below 1e-10 L/h |

There is no sampling error bar: these are deterministic calculations. The
tolerances bound accepted floating-point discrepancy, not biological uncertainty.
Zero observed discrepancy at printed precision does not imply exact arithmetic
throughout the numerical solver. No parameters were fitted.

The corrected **reporting sum** is 17.104405368562304 L/h, replacing a
mixed-reference sum of 28.48825272206046 L/h. Its difference from exact dose/AUC
is **0.001335622981958 L/h**: nominal renal filtration omits finite-flow extraction.
That distinction is reported explicitly rather than treating a shared plasma
reference as proof of identical model semantics.

Focused validation: **34 passed** with
`tests/test_graph.py tests/test_graph_clearance.py` (24 new cases). The full graph
validation script passed, including blood-flow conservation and the existing
1,687-pair molecular exclusion comparison. Full repository and browser-engine
checks are to be recorded by the integrating contributor before publication.

## Limits

**Subsequent combined verification, 2026-11-05:** 111 Python tests passed and
JavaScript engine checks passed with checked-in and regenerated references.
See the [integration record](../../docs/project-asma-integration.md) for the
revision, environment, commands and limits; numerical agreement does not establish
biological applicability.

- No change to the PBPK engine or evidence of a biological prediction improving.
- Clearance reporting is not automatic model execution. A schema can reject
  contradictory declared metadata; it cannot prove the original measurement's
  reference basis was truthful.
- The exact systemic expression applies to this linear perfusion-limited model,
  with the current renal/hepatic routes and a systemic IV bolus. It is not a
  general clearance formula for nonlinear transport or other model structures.
- The full Track E milestone (SBML import and conservation invariants) remains
  open. This contribution supplies one invariant that catches a real error.
- Updated Mermaid sources are checked in. SVG/PNG rendering was attempted with
  Mermaid CLI 12.2.0 and local Chromium, but the environment denied the renderer
  with `net::ERR_BLOCKED_BY_ADMINISTRATOR`. Stale affected images were removed;
  docs use a small correct inline Mermaid view and link the full source. Browser
  policy was not changed.

## Reproduce

```bash
# In an environment with the repository's dev dependencies installed:
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_graph.py tests/test_graph_clearance.py
PYTHONHASHSEED=0 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/validate_graph.py

# Optional rasterization, where the local browser policy permits it:
mmdc -i docs/img/pbpk_graph.mmd -o docs/img/pbpk_graph.svg -b transparent
mmdc -i docs/img/pbpk_elimination_subgraph.mmd \
  -o docs/img/pbpk_elimination_subgraph.svg -b transparent
```

The commands above used `/tmp/vh-dev/bin/python` for the isolated dev environment.
`docs/validation_graph.json` records the default-model measurements. The six-case
rate check can be reproduced with the following independent calculation:

```python
from fractions import Fraction as F
from vh.body import MIDAZOLAM, PBPKModel, reference_individual
from vh.graph.physiology_graph import build_pbpk_graph

errors = []
for sex in ("male", "female"):
    for bp in (0.5, 1.0, 2.0):
        ind = reference_individual(sex)
        model = PBPKModel(MIDAZOLAM, ind, bp=bp, fu=0.2, clint_hepatic_l_h=40.0)
        graph, _ = build_pbpk_graph(model)
        q, ratio = F(str(ind.liver_flow_total)), F(str(bp))
        cin = 2 * ratio
        cout = q * cin / (q + F(40) * F("0.2") / ratio)
        exported_rate = graph.edges_of_type("clears")[0].attrs["cl"] * 2
        errors.append(abs(exported_rate - float(q * (cin - cout))))
print(max(errors))  # observed: 0.0 mg/h
```

## References

- Existing model definitions and parameter provenance: `vh/body/pbpk.py`,
  `vh/body/compound.py`, `vh/body/physiology.py`, `vh/body/partition.py`.
- The upstream baseline inspected was
  `3d75e8e25d6f3fec08e1ccecd424532bccaa1b82`.
- The independent authority is the stated conservation equation and exact
  linear-system integral; no new empirical reference or fitting dataset is claimed.
