# Virtual Human and Asma: a research foundation

Virtual Human is a **workbench for building and checking pieces of a computerized
human**. Its useful starting point is a careful evidence map and small, inspectable
experiments. It is not a complete virtual person or a treatment-decision tool.

This guide compiles the project's plain-language explanation into a shared
starting point for contributors. It connects existing work rather than adopting a
new disease model. Use it with the [architecture](architecture.md),
[Asma integration record](project-asma-integration.md), and
[research-question template](research-question-template.md).

## 1. Four maps, four different questions

A body spans many scales: molecules interact, cells respond, drugs move through
organs, and immune activity has consequences across tissues. Different tools are
needed at each scale. Connecting them requires explicit assumptions, compatible
inputs and validation of the connection itself.

| Analogy | What is implemented | What it can answer | What it does not establish |
| --- | --- | --- | --- |
| Street map | [Typed graph and schema](../vh/graph/core.py); Asma adds its own evidence rules | What connects to what? Does a recorded assertion match the requested context? | A route through a graph is not proof of causation, recognition or injury. |
| Traffic simulation | [Reaction networks and interventions](reaction_experiments.md) | Given explicit rates and operations, how do quantities change, and which assumptions change the result? | Artificial rates do not become measured biology because the solver runs. |
| Plumbing map | [Whole-body PBPK](../vh/body/pbpk.py) | Under the compartment model, where does a drug go and how does concentration change? | Drug exposure alone does not establish disease response or an individual's outcome. |
| Microscope | [Molecular](../vh/molecular/) and [quantum](../vh/quantum/) tools | Forces, interactions and small-system calculations under the chosen physical model | Agreement on small systems does not validate a complete disease mechanism. |

A street map can show two places are connected without predicting traffic. In the
same way, graph structure needs separately justified equations and parameters
before it can become a dynamic model. The optional [model registry](model_contracts.md)
checks scalar quantity names, units, reference bases, contexts and bounds before
calling an explicitly registered model. It does not supply missing observations
or certify the model's biological truth.

The existing [occupancy bridge](../vh/bridge/occupancy.py) uses unbound target-site
concentration and a binding constant to calculate occupancy. Its documentation
states that binding constants currently come from experiment; obtaining them from
atomistic binding-free-energy calculations remains development work. Occupancy
is itself distinct from cellular response and tissue injury.

## 2. Asma's evidence map today

Asma's public research includes ovarian cancer and anti-Yo-related biology.
Three questions that sound like one story must remain separate:

1. Is a particular protein present?
2. Can the immune system recognize it in the relevant context?
3. Does that recognition lead to injury in neural tissue?

Evidence for the first question does not automatically answer the second or
third. Antibody binding, protein function, HLA annotation, receptor recognition
and tissue injury are distinct claims. Direct recognition and indirect injury
can also coexist; they should not be forced into mutually exclusive explanations.

Asma currently uses a small, unchanged, hash-pinned copy of Virtual Human's graph
container and schema, with Asma-owned matching rules. It keeps entity identity,
relationship, source, assay and other recorded contexts attached to assertions.
The query must be satisfied by one complete matching record; fragments from
different studies must not be joined into an observation that nobody made.

For example, separate catalog records containing CDR2L, HLA-II and a glioblastoma
model do not establish that complete conjunction in one record. A finite negative
screen also cannot establish universal absence of recognition. Missing support
remains unresolved; contradictory matching assertions remain conflicting.

The recorded [evidence-graph result][evidence-result] contains 24 curated
assertions and 22 manually specified requests. These are not 24 independent
observations or patients. Matching the expected answers is a hand-curated software
benchmark, not an independent scientific accuracy estimate. Human curation can
still omit important context or misread a source.
Omitting a query field does not generalize the evidence: the full matched context
is returned, and callers must request every distinction relevant to their claim.

**Verification boundary:** the evidence map is machine-checked, not formally
proved. Schema checks and regression tests establish bounded software behavior.
A future proof or exhaustive check would need a specified property and model,
such as prohibiting cross-observation joins. It would still not prove that the
underlying biology or the chosen rules describe nature correctly. Exact rational
conservation checks in the reaction engine are a different, narrower mathematical
guarantee, not a formal proof of Asma's evidence system.

## 3. A toy experiment: ask which assumption changes the answer

Consider whether the order of surgery and chemotherapy changes a model outcome.
Start with deliberately imaginary burden units, not patients, cells or doses:

- Initial burden: 100.
- Operation A retains half; operation B also retains half.
- A then B: 100 -> 50 -> 25.
- B then A: 100 -> 50 -> 25.

In this toy world, order makes no difference. The conclusion is about the stated
operations, not a recommendation about clinical treatment order. If a richer model
predicts a difference, ask **which added assumption caused it?**

An exact replay needs only Python's standard library:

```python
from fractions import Fraction

initial = Fraction(100)
retain_a = retain_b = Fraction(1, 2)
a_then_b = retain_b * (retain_a * initial)
b_then_a = retain_a * (retain_b * initial)
assert a_then_b == b_then_a == 25
```

Possible changes include a fixed absolute residual instead of a fixed percentage,
state-dependent dynamics, conversion between populations, different gaps or
schedules, or inaccessible disease. Selective effects alone are not sufficient
in every model: two fixed diagonal removal operations still commute. Scalar
proportional retention commutes with any fixed homogeneous linear flow. These
null cases should be retained alongside counterexamples.

Asma's [treatment-sequence assumption audit][sequence-result] compares proportional
retention with a capped fixed-residual rule, using artificial sensitive/resistant
burdens and exact rational arithmetic. It preserves negative results and explains
why fitting one sequence can leave the other sequence unresolved. It does not
reproduce the full published Gu/Rosen model or estimate clinical efficacy.

Virtual Human's [optional local bridge](../scripts/validate_asma_interventions.py)
compares corresponding artificial outcomes against the Asma implementation.
Agreement checks the same toy mathematics. Reusing numbers as artificial mol/L
here and arbitrary burden units there is not a physiological unit conversion.
The bridge requires a trusted local Asma checkout and does not download code.
See its [recorded result](../auto-research/2026-11-05-E-reaction-experiments/bridge-results.json)
and the [integration record](project-asma-integration.md) for exact revision/hash
provenance. It does not establish which sequence is better for a real patient.

## 4. What is available, what remains proposed

Snapshot checked on **2026-11-05**, against Virtual Human main
`5b8c9e5cd5ec1d3fab595d6398f90c2bc4c889bd` and Asma main
`aa944ef8ee56bcf61153c10497c89f066fc240ad`. This is a dated inventory, not an
automatically updated status dashboard.

- **Present on main:** graph/schema, explicit scalar model contracts, elementary
  mass-action reaction experiments, organ/molecular/quantum tools and the
  occupancy bridge. Reaction/intervention work was merged in
  [PR #3](https://github.com/Foundation-0/virtual-human/pull/3).
- **Used by Asma:** its pinned graph core and the bounded synthetic comparison
  described above. The organ, molecular and quantum engines are **not integrated
  into Asma**. Its qualitative claims remain qualitative.
- **Recorded inputs, not a disease simulator:** the
  [ovarian HGSOC ledger](../data/clinical/README-ovarian-parameter-ledger.md),
  merged in [PR #5](https://github.com/Foundation-0/virtual-human/pull/5), labels
  source-backed values, conjectures and missing quantities. Population prevalence,
  trial hazard ratios and rate constants are not interchangeable. Missing growth,
  removal and drug-PK parameters remain gaps, not values to guess.
- **Open proposal at this snapshot:**
  [SBML-import PR #4](https://github.com/Foundation-0/virtual-human/pull/4).
  It is not on the inspected main branch. Do not assume its importer or validation
  results are available in a main checkout. This foundation does not duplicate or
  modify that work.
- **Future research:** bounded disease mechanisms, validated connections between
  scales, applicable parameters, uncertainty analysis and independent outcome
  validation. The [roadmap](roadmap.md) is an ambition and acceptance-test list,
  not proof that these capabilities already exist.

## 5. How to start from this foundation

For one specific research question, use the [question template](research-question-template.md):

1. State the endpoint, population/system and context. Keep recognition, injury,
   exposure and treatment response distinct.
2. Map the public evidence and its source/locator/access limits. Preserve
   conflicting and missing evidence rather than converting it to numbers.
3. Write competing assumptions and the simplest null case before adding
   complexity. Keep synthetic fixtures visibly artificial.
4. Identify the measurements and units the model actually requires. Explain
   every proposed transformation from a published quantity to a parameter.
5. Compare implementation with an independent engine or analytic result. For a
   scientific prediction, separately specify independent observations that could
   disprove it, held apart from any fitting data.
6. Record what failed and what remains unresolved. Revise or stop when the
   assumptions are not distinguishable at available precision. A prettier graph
   or passing test cannot fill missing biological evidence.
7. Review reusable library changes here and domain evidence/rules in Asma. Keep
   vendored updates deliberate, preserve attribution and rerun the relevant
   checks. See [contribution guidance](../auto-research.md).

Drug-distribution or molecular tools may eventually help with carefully bounded
questions. That requires the missing biological links, applicable inputs and
validation, rather than plugging ovarian or anti-Yo labels into a generic model.
Independent scientific review remains separate from software review.

## Scope and public-data boundary

This foundation covers public evidence and synthetic examples only. It contains
no patient records, personal correspondence, private clinical analysis or
patient-specific simulation. No new data acquisition, clinical experiment,
treatment recommendation or deployment is authorized by this document.

The practical aim is to draw a careful map, experiment with assumptions, and ask
what evidence would change the answer. The research-question template is a
proposed starting workflow, not a claim that a new experiment has been completed.

[evidence-result]: https://github.com/Foundation-0/project-asma/blob/aa944ef8ee56bcf61153c10497c89f066fc240ad/docs/evidence-graph-result-2026-11-05.md
[sequence-result]: https://github.com/Foundation-0/project-asma/blob/aa944ef8ee56bcf61153c10497c89f066fc240ad/docs/research/rosen-treatment-sequence-assumptions.md
