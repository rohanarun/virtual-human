# Conserved reaction networks and explicit ordered interventions

- **Track**: E (incremental; full SBML milestone remains open)
- **Author**: OpenAI Codex (Astra), under @ahrazzle's direction
- **Date**: 2026-11-05
- **Branch / PR**: `auto/E-reaction-experiments`, [Foundation-0/virtual-human PR #3](https://github.com/Foundation-0/virtual-human/pull/3)
- **Code added**: `vh/graph/reactions.py`, `scripts/validate_reaction_experiments.py`,
  `tests/test_reaction_experiments.py`; [usage guide](../../docs/reaction_experiments.md)

## Question

Can the graph layer support small, explicit numerical experiments that distinguish
reaction conservation from external interventions and test whether an apparent
order effect follows from stated assumptions? This is useful before connecting
literature hypotheses to numerical models: the equations, interventions and
mathematical null cases must be inspectable first.

## Method

Added deterministic elementary mass-action dynamics in one fixed well-mixed
compartment, exact integer stoichiometry and rational left-nullspace conservation
checks, and a bipartite graph export preserving catalysts and coefficients. Rates
carry canonical concentration/time units and provenance. Each species and the
network carry provenance; context names the modeled system/compartment. All
states and rates in this contribution are explicitly artificial, with no fitting.

A from-scratch classical RK4 solver integrates `dC/dt = S v(C)` and optional
analytic forward sensitivities with respect to local dimensionless multipliers
on effective reaction rates. The product-rule Jacobian is valid at zero
concentration. Ordered phases expose retain/add operations and rate multipliers,
record both sides of every state jump, and reset unspecified rate multipliers to
base values in each phase. Negative or nonfinite states fail without clipping.
The numerical implementation reuses NumPy arithmetic, not SciPy's ODE solver.

The main implementation is general library functionality. Project Asma context,
companion work and optional cross-repository reproduction stay in the fork's
[coordination document](../../docs/project-asma-integration.md).

## Validation — independent mathematical and engine authorities

Authorities were not tuned to implementation outputs:

1. For `A -> B`, `A(t)=A0 exp(-kt)`, `B(t)=A0+B0-A(t)` and multiplier derivative
   `dA/dm=-kt A(t)`. Reversible conversion is separately tested against its exact
   equilibrium-plus-exponential solution.
2. For `2A -> D`, `A(t)=A0/(1+2kA0t)`, `D(t)=D0+(A0-A(t))/2`, and
   `dA/dm=-A0(2kA0t)/(1+2kA0t)^2`. `A+2D` is exactly conserved structurally.
3. For `A+B <-> C`, a separate, hand-written three-equation system is integrated
   by SciPy DOP853 (`rtol=2e-13`, `atol=2e-14`). This referee does not reuse our
   stoichiometric matrix, flux or Jacobian. Rate sensitivities are independently
   checked by central differences of that solver.
4. Scalar retention and homogeneous linear flow commute exactly by linear
   algebra. Selective retention and nonlinear dimerization provide boundary
   counterexamples; their outcomes are also checked against exact solutions.

The [machine-readable results](results.json) were generated with Python 3.12.14,
NumPy 2.5.4 and SciPy 1.18.1. Maximum absolute discrepancies over the stated
sampling grids are below; they are numerical errors, **not** statistical or
biological uncertainty estimates. Rate/state parameters and grids are in the
reproduction script, with no external dataset or parameter search.

| Quantity | Authority | Maximum absolute error (mol/L) | Acceptance bound (mol/L) |
|---|---|---:|---:|
| Linear trajectory, 21 samples to 5 s, step <=0.01 s | Exact exponential | 1.479e-11 | 2e-10 |
| Linear multiplier sensitivity | Exact derivative | 6.065e-11 | 2e-10 |
| Dimerization trajectory, 17 samples to 4 s, step <=0.005 s | Exact rational solution | 1.160e-11 | 2e-10 |
| Dimerization multiplier sensitivity | Exact derivative | 4.777e-11 | 2e-10 |
| Dimer conserved pool drift | Exact initial monomer pool | 2.887e-15 | 1e-14 |
| Nonlinear binding, 26 samples to 5 s, step <=0.005 s | Independent DOP853 equations | 8.520e-12 | 2e-10 |
| Binding multiplier sensitivities | DOP853 central difference, epsilon=1e-4 | 8.755e-10 | 2e-9 |
| Scalar-retention linear order difference | Exact zero commutator | 1.221e-15 | 1e-14 |

The binding finite-difference discrepancy falls from 8.697e-8 to 8.755e-10 mol/L
when epsilon falls from 1e-3 to 1e-4, consistent with central-difference truncation
error. Linear step-halving errors at 0.2, 0.1 and 0.05 s are 2.484e-6, 1.465e-7
and 8.890e-9 mol/L, consistent with fourth-order convergence. These are measured
checks on these cases, not universal solver guarantees.

An intentionally incorrect dimer pool `A+D` returns exact reaction residual -1;
`A+2D` returns exact zero. This catches a **synthetic** modeling mistake, not a
claimed new real-world biological inconsistency. A retained fraction 0.4 before
versus after nonlinear dimerization gives final A concentrations 0.327868852460
versus 0.173913043489 mol/L, an expected 0.153955808971 mol/L difference under
those artificial equations. The same operation on the linear conversion gives
zero within 1.221e-15 mol/L. A separate remove/add event changes the conserved
pool from 2 to 1.3 mol/L; the subsequent final sum is 1.3000000000000003 mol/L.

No parameter was fitted. The full Python suite passed: **153 tests, no skips,
62.85 seconds**, including 42 new focused cases; checked-in and
regenerated JavaScript engine fixtures both passed. Regenerated molecular
fixtures were restored because the molecular engines and inputs are unchanged.
Test coverage includes source/sink and inert networks,
zero concentrations/rates, exact conservation, graph catalyst topology, state
mapping order, invalid stoichiometry/units/provenance, failed integration,
observation grids, resource bounds and explicit intervention semantics.

## Limits

This is a deterministic single-compartment elementary-rate-law engine. It does
not support SBML, changing volumes, inter-compartment transport, stiff adaptive
integration, empirical kinetic laws, events triggered by state, uncertainty
propagation or parameter fitting. A fixed step must be convergence-checked for
each new model. Exact nullspace calculations can be costly for large networks.

Sensitivity is local to a single trajectory with initial conditions held fixed;
it does not propagate through protocol interventions. The zero-rate multiplier
derivative is zero, not an estimate of the effect of adding a new process. Source
strings and contexts are required declarations, not evidence certification.

No anti-Yo recognition, ovarian cancer response, patient outcome or clinical
benefit is modeled or validated here. No private data was used. Mathematical
commutativity under a specific linear model does not establish real treatment
order equivalence. The full Track E BioModels import/invariant milestone remains
open, and none of the upstream molecular/PBPK/quantum engines are changed.

## Reproduce

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/validate_reaction_experiments.py \
  --output auto-research/2026-11-05-E-reaction-experiments/results.json
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q
node web/test_engines.mjs
```

## References

- Exact solutions above follow by separation of variables and differentiation;
  the linear intervention null follows from scalar-matrix commutation.
- Hairer, Nørsett and Wanner, *Solving Ordinary Differential Equations I:
  Nonstiff Problems*, 2nd edition (1993), classical Runge–Kutta methods.
  <https://doi.org/10.1007/978-3-540-78862-1>
- Virtanen et al., SciPy 1.0 (2020), *Nature Methods* 17:261–272.
  <https://doi.org/10.1038/s41592-019-0686-2>
- SciPy `solve_ivp` DOP853 documentation:
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>
