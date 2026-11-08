# Foundation-0 development fork and Project Asma

This is a development fork of [Rohan Arun's Virtual Human](https://github.com/rohanarun/virtual-human),
created by the Project Asma owner on 2026-11-05. Upstream remains the original
project. Its MIT license, creator credit and contribution requirements are
preserved. The upstream README credits Claude for the original implementation
under the maintainer's direction. The graph contributions below were prepared
with OpenAI Codex under [@ahrazzle](https://github.com/ahrazzle)'s direction.

## Shared purpose and repository boundary

[Project Asma](https://github.com/Foundation-0/project-asma) uses a pinned,
unchanged copy of `vh/graph/core.py` for an offline public-evidence prototype.
It supplies a concrete test case: do not silently substitute antibody binding,
protein function, HLA annotation, receptor recognition and tissue injury for one
another. The Asma repository owns the literature curation and domain-specific
rules. This fork owns reusable graph interfaces and numerical-model checks.

Asma does not currently adopt the molecular, PBPK or quantum engines, nor does it
use the new scalar model registry to invent numerical values for unknown biology.
Its qualitative assertions stay qualitative. Recognition and injury are separate
questions: direct recognition of a CDR2L ligand and indirect injury can coexist.
Neither is established by a graph or a successful software test.

No patient records, personal correspondence, private clinical analysis or sensitive
derived hypotheses belong in this public fork. Only public-source requirements,
original software, permitted upstream inputs and clearly artificial fixtures are
used for this contribution.

## Contributions and acceptance

Reviewable changes: [fork PR #1](https://github.com/Foundation-0/virtual-human/pull/1)
and [Project Asma PR #45](https://github.com/Foundation-0/project-asma/pull/45).
Both were opened as drafts. They are independently reviewable: Asma uses the
unchanged baseline graph core, so its prototype does not require merging this
fork PR first. Upstream submission is still pending contributor access.

The initial integration combines two separately reviewable commits and reports:

| Contribution | Validation authority and outcome |
| --- | --- |
| [Explicit model contracts](../auto-research/2026-11-05-E-model-contracts/report.md) | Exact scalar mass balance: maximum error 0 L/h over four synthetic cases, tolerance 1e-12 L/h; 50 new contract cases. |
| [Clearance reference correction](../auto-research/2026-11-05-E-clearance-reference/report.md) | Exact organ mass balance and integrated linear ODE; 24 new cases. Plasma reporting sum 17.104405 L/h remains distinct from IV dose/AUC 17.103070 L/h. |

Neither completes the full Track E SBML milestone. Existing PBPK differential
equations are unchanged. Domain/validation strings in a model contract are
declarations; the runtime verifies supplied interface semantics and bounds, not
biological truth. The clearance check catches a real reference-basis error in
graph reporting and avoids overstating copied export arithmetic as independent
physiological validation.

## Combined verification, 2026-11-05

On code revision `d45ca7f50188dc3580dbec11bd0e14367e107fd6` (the two implementation
commits applied to upstream `3d75e8e25d6f3fec08e1ccecd424532bccaa1b82`):

- **111 Python tests passed**, no skips, in 76.12 s.
- JavaScript engine checks passed using the checked-in reference fixtures.
- `scripts/export_web.py` completed, then JavaScript checks passed again against
  regenerated references. Worst plasma-profile relative error was 4.8e-13 over
  48 points. Molecular energy/force comparisons and 500-step stability checks
  also passed. This is numerical verification against the supplied fixtures,
  not validation of anti-Yo mechanisms or individual outcomes.
- Regenerated molecular web fixtures were not retained: the contribution does
  not change those engines or their inputs. Checked-in fixture checks had already
  passed independently. No unrelated model or clinical dataset files changed.
- Focused tests and the analytical comparisons were independently rerun by an
  internal reviewing agent. This is not external scientific review.

Runtime: Linux x86_64; Python 3.12.14; Node 24.19.0. Installed numerical/test
versions: NumPy 2.5.4, SciPy 1.18.1, JAX/JAXlib 0.12.0.1, OpenMM 8.6.1,
PDBFixer 1.12.0, PySCF 2.15.0, geomeTRIC 1.1.1, pytest 9.2.0,
Matplotlib 3.11.2. This records the observed environment, not a new repository
dependency lock or a claim of portability to every allowed package version.

```bash
python -m venv /tmp/vh-dev
/tmp/vh-dev/bin/python -m pip install -e '.[dev]'
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/vh-dev/bin/python -m pytest -q
node web/test_engines.mjs
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/vh-dev/bin/python scripts/export_web.py
node web/test_engines.mjs
```

The clearance report records a diagram-rendering limitation: browser policy
blocked Mermaid's local renderer. Correct Mermaid source and inline diagrams
replace the affected stale images; no browser policy was changed.

## How the repositories progress together

1. Record a concrete Asma failure case or missing interface, without importing
   private data. Decide whether it is a general library concern or a domain rule.
2. Put general fixes here with an independent validation authority and a report
   following `auto-research.md`; put evidence semantics and scientific critique
   in Asma. Preserve unresolved and negative findings.
3. Review each change in a PR. Asma updates its vendored source only deliberately,
   preserving hashes and the upstream MIT notice, then reruns its offline checks.
4. Submit reusable changes to the original repository when contributor access is
   available. The owner is arranging that access. Fork PRs do not imply upstream
   acceptance, and fork-specific coordination docs need not accompany upstream
   feature PRs.

**2026-11-05 14:41 UTC update:** the team merged
[Asma PR #45](https://github.com/Foundation-0/project-asma/pull/45) as
`3afeb2b9b14b9bc7fc126e3503554d7673b43673` after its
[full CI](https://github.com/Foundation-0/project-asma/actions/runs/42046224134)
passed: 227 Python tests, pinned container builds, offline checks and 40 execution
controls. This fork's reusable changes remain independently reviewable in PR #1.
The Asma merge adopts only the bounded graph use described here, not the simulation
stack or a clinically validated model. This contributing session did not merge it.

No automatic merge, deployment, background research run or patient simulation is
introduced by this workflow. Further simulation work needs a defined biological
question, applicable inputs and validation that can resolve that question.

## 2026-11-05 update: executable synthetic intervention checks

Fork [PR #1](https://github.com/Foundation-0/virtual-human/pull/1) was merged by
the team at 14:53:50 UTC as
`3236b915e659ab96ba862dce3c07930846b561d8`. The contribution below branches from
that merged main. This contributing session did not merge it, and no original
upstream posting or permission retry was needed.

The next reusable increment, [fork PR #3](https://github.com/Foundation-0/virtual-human/pull/3), adds
[reaction networks and ordered interventions](reaction_experiments.md): elementary
mass-action dynamics in one fixed compartment, exact rational conserved-pool
checks, graph export, analytic local rate-multiplier sensitivities, and explicit
per-phase state/rate interventions. Its
[report](../auto-research/2026-11-05-E-reaction-experiments/report.md) distinguishes
exact structural statements from integration errors and externally changed pools.
Maximum trajectory discrepancy in the report is 1.479e-11 mol/L against exact
kinetics; nonlinear binding agrees with an independently written DOP853 system
to 8.520e-12 mol/L. All states, rates and interventions are artificial.

The immediate Asma research need is to test treatment-sequence **assumptions**
raised by the public Rosen/Gu work, implemented in
[Asma PR #47](https://github.com/Foundation-0/project-asma/pull/47). Its initial
implementation commit is `aca2556dfdde3ceca8c0c2da1bfab0105eb21e7d`;
The bridge was rerun against clean Asma revision
`03d6cad7eaa97cf6e332515f6e5e0a9172b5addc`; the exact module is unchanged.
`tools/analyze_treatment_sequence.py` has SHA-256
`0b3492e8b9fcc9dd2b7100d81552e212bc0f9c1aeed420b40bb76a9c93cb7a1b`. Scalar proportional removal commutes with
homogeneous linear dynamics. A fixed absolute residual operation is a different,
state-dependent map. The VH engine exposes those choices rather than assigning
clinical meaning to them. Selective removal and nonlinear kinetics are explicit
counterexamples to extrapolating the scalar/linear null case.

An optional fork-specific bridge check is provided by
[`scripts/validate_asma_interventions.py`](../scripts/validate_asma_interventions.py).
It uses a **trusted local** Asma checkout containing
`tools/analyze_treatment_sequence.py`; it does not download code or add Asma as a
runtime/test-suite dependency. The three artificial fixtures compare twelve
named intervention-order outcomes with exact rational operators from the Asma
experiment. Maximum absolute discrepancy is 4.8531e-14 mol/L, below the declared
1e-10 mol/L tolerance. A fixed-residual operation remains explicit caller logic,
converted to named retention fractions before passing it to the VH protocol.

```bash
# Use a trusted checkout of the companion Asma research branch/PR.
PYTHONPATH=. python scripts/validate_asma_interventions.py \
  --asma-root /path/to/project-asma
```

The [recorded bridge output](../auto-research/2026-11-05-E-reaction-experiments/bridge-results.json)
contains checkout commits, dirty flags and SHA-256 hashes of the actual three
executed source files. Dirty flags deliberately distinguish worktree code from a
clean committed revision. Values are reused as artificial mol/L concentrations
in VH and arbitrary burden units in Asma to compare the same mathematics; this
is **not a physiological unit conversion**. The bridge does not reproduce the
full published Gu model or analyze a clinical dataset.

This enables explicit hypothesis-testing software experiments, not simulated
human disease. No patient data, anti-Yo mechanism claim, treatment recommendation
or validation of individual outcomes enters either the bridge or the library.
The Track E BioModels SBML milestone remains open. Fork-specific coordination
and the optional bridge are separate from the reusable library commit for future
upstream review.

Verification for this increment: **153 Python tests passed with no skips** in
62.85 seconds, including 42 new focused cases. JavaScript engine checks passed
against both checked-in and regenerated fixtures, with worst plasma-profile
relative error 4.8e-13 over 48 points. Regenerated molecular fixtures were not
retained. This is numerical software verification, not biological validation.
