# Kinetic invariants and paraneoplastic antigen exposure

- **Track**: E (graph and reaction formalization)
- **Author**: Claude (Anthropic), Foundation-0 fork, under the Project Asma owner's direction
- **Date**: 2026-11-05
- **Branch**: `claude/admiring-cori-0qyybb`
- **Code added**: `vh/graph/reactions.py` (`kinetic_invariant_basis`, `kinetic_residuals`),
  `scripts/validate_paraneoplastic_exposure.py`, `tests/test_kinetic_invariants.py`,
  `tests/test_paraneoplastic_exposure.py`

## Question

Project Asma asked what an irreversible, immune-mediated neural injury records
about a tumor's antigen over time, and whether treatment order can change it when
order provably cannot change the tumor endpoint. Answering that on this engine
needs a reusable capability the engine lacked: exact detection of conserved
quantities that depend on **rate values**, which is how an integral such as an
antigen exposure (AUC) or an eliminated amount becomes checkable.

## Method

**Library.** For mass-action `dC/dt = S v(C)`, group reactions by reactant
monomial. Distinct monomials are linearly independent functions, so `w` is a
linear invariant for every state exactly when `w` annihilates each group's
rate-weighted stoichiometry. `kinetic_invariant_basis()` solves that system with
the existing exact rational row reduction. Rates are the exact values of the
stored floats unless rationals are supplied whose float equals the stored rate.
`kinetic_residuals()` checks a declared invariant per monomial. Stoichiometric
pools are always included; rate-dependent vectors appear only when reactions share
a monomial.

**Experiment.** An invented, dimensionless network contains tumor burden `A`, an
antigen-driven effector `E`, target neurons `N`, an injury marker `F`, an
optional self-sustaining CNS effector `C`, and accumulators for burden exposure
`U`, burden killed in the body `R` and cumulative hazard `H`. Rate scales select
**burden-driven** presentation (`A -> A + E`) or **release-driven** presentation
(chemotherapy kill `A -> E + R`). Surgery is an instantaneous retain on `A`.
Schedules place one surgery among six chemotherapy cycles, either as instantaneous
operations matching Asma's exact null case, or with chemotherapy as a continuous
kill over one unit. Quantities are reused as mol/L and seconds only because the
engine requires units.

## Validation

Authorities: exact kinetic invariants, closed-form integrals and, as an optional
bridge, Project Asma's exact rational module `tools/analyze_paraneoplastic_exposure.py`.
Fixed-step RK4 with step 0.005.

| Quantity | This work | Authority | Error |
|---|---|---|---|
| Exposure identity `H + (κ/δ)E − (κα/δ)·source` | Found by the finder | Hand derivation | Exact (4/4 hypothesis × chemotherapy settings) |
| Clearance identity `X − kU` | Found by the finder | Hand derivation; bolus `dose/k` | Exact; numerical `≤ 1e-12` |
| Invariant drift along all discrete protocols | RK4 trajectories | Constant of motion | max `8.7e-14` |
| Hazard difference between surgery positions | RK4 | Identity at finite time | max `8.9e-14` |
| Survival `N = exp(−H)` | RK4 | Exact solution | max `1.2e-12` |
| Exposure and release by position (28 values) | `U`, `R` accumulators | Asma exact rationals | max `8.5e-12` absolute |
| Continuous-chemotherapy exposure, release, endpoint | RK4 | Closed-form integrals | max `5.0e-12` relative |
| Committed hazard after complete antigen removal | RK4 | `κE(t*)(1−e^{−δT})/δ` | `2.6e-15` |

Executed findings on the invented fixture:

| Finding | Result |
| --- | --- |
| Equal tumor endpoint, different neural exposure | Surgery first versus after three cycles, surgery retaining 1/10: hazard difference 0.700 (burden) and 0.7875 (release), each equal to κ times Asma's exact exposure difference of 70 and 78.75 |
| Hypotheses disagree when surgery is weak | Surgery retaining 3/4: burden-driven exposure is lowest with surgery last; release is lowest with surgery first |
| Continuous chemotherapy moves the threshold | Burden-driven optimum is surgery-first when `q < 1/(1 + c)` (here `q < 0.591`), not `q < a`; both endpoints confirmed by simulation |
| Regimes after antigen removal | Marker ratio t=20 to t=2: `1.5e-4` antigen-driven versus `0.15` autonomous |
| No-rise theorem | From steady state, burden-driven effector never rises (largest step `7.2e-13`, numerical noise); release-driven rises during the first chemotherapy cycle |

A wiring error was caught by the finder during development: the first version of
the release hypothesis left burden presentation on, and the pure release identity
correctly failed. The test suite now asserts that the pure identity fails for the
mixture.

## Limits

- Every rate, burden and unit is invented. Hazard differences show bookkeeping and
  direction, not effect sizes. No anti-Yo biology, NfL behavior, patient course or
  treatment benefit is claimed or validated.
- Tumor growth, resistance, order-dependent surgical effectiveness and drug
  pharmacokinetics are omitted; each can change the order conclusions.
- The invariant finder is exact for the supplied mass-action equations only. It
  does not find nonlinear invariants such as `ln N + H`, which is checked here
  numerically. Exact rational elimination can be expensive for large networks.
- The Asma bridge needs a trusted local checkout and imports its module; Asma is
  not a dependency of the library or the test suite.

## Reproduce

```bash
PYTHONPATH=. python -m pytest -q tests/test_kinetic_invariants.py tests/test_paraneoplastic_exposure.py
PYTHONPATH=. python scripts/validate_paraneoplastic_exposure.py > results.json
PYTHONPATH=. python scripts/validate_paraneoplastic_exposure.py \
  --asma-root /path/to/project-asma > bridge-results.json
```

`results.json` and `bridge-results.json` in this folder record the executed runs,
checkout commits, dirty flags and SHA-256 hashes of the executed sources.

## References

- Project Asma record: [What the brain records](https://github.com/Foundation-0/project-asma/blob/main/docs/research/paraneoplastic-exposure-observability.md)
  (public literature premises, exact rational results and the proposed study design).
- Albert ML et al. Tumor-specific killer cells in paraneoplastic cerebellar
  degeneration. Nat Med 1998;4:1321–1324. doi:10.1038/3315 (motivates release-driven presentation).
- Venkatraman A, Opal P. Paraneoplastic cerebellar degeneration with anti-Yo
  antibodies – a review. Ann Clin Transl Neurol 2016;3:655–663. doi:10.1002/acn3.328.
