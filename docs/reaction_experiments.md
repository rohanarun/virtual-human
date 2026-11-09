# Explicit reaction and intervention experiments

`vh.graph.reactions` supplies a small numerical testing engine for elementary
mass-action networks in **one fixed, well-mixed compartment**. Its purpose is to
make assumed dynamics, conserved pools and ordered interventions executable and
checkable. Model equations and parameter provenance must be supplied by the
caller. A graph cannot choose those assumptions or establish biological truth.

The engine computes concentrations in **mol/L** and time in **seconds**, with no
implicit conversions. All examples below are artificial mathematical fixtures;
the magnitudes do not represent biological measurements. Initial values and the
choice of observation times must be covered by the caller's recorded experiment
provenance. Required source strings are audit metadata, not verified evidence.

```python
from vh.graph.reactions import (
    Species, Reaction, ReactionNetwork, Phase, run_protocol,
)

source = ("Artificial demonstration: initial A=2, B=0 mol/L; k=0.7 /s",)
network = ReactionNetwork(
    [Species("A", source), Species("B", source)],
    [Reaction("convert", {"A": 1}, {"B": 1}, 0.7, "1/s", source)],
    context={"system": "synthetic", "compartment": "fixed-volume"},
    provenance=source,
)
trajectory = network.simulate(
    {"A": 2, "B": 0}, [0, 1, 2, 3], max_step_s=0.01,
    sensitivity_reactions=["convert"],
)
assert network.pool_residuals({"A": 1, "B": 1}) == {"convert": 0}
print(trajectory.final)

protocol = run_protocol(network, {"A": 2, "B": 0}, [
    Phase("paused", 2, source, rate_scales={"convert": 0}),
    Phase("exchange_then_evolve", 1, source,
          retain={"A": 0.5}, add={"B": 0.3}),
], max_step_s=0.01)
assert protocol[1].before == {"A": 2, "B": 0}
assert protocol[1].after == {"A": 1, "B": 0.3}
```

## Equations and structural checks

For stoichiometric matrix `S = products - reactants`, the engine evaluates
`dC/dt = S v(C)` with `v_j = k_j product_i C_i**reactant_ij`. Stoichiometric
coefficients are positive integers, including the elementary reaction order.
A source has an empty reactant mapping and a sink has an empty product mapping.
Reversible reactions are two separately named one-way reactions. This is a
deterministic concentration rate law, without stochastic combinatorial factors.

Canonical rate units are `mol/L/s` (order zero), `1/s` (order one), and
`(mol/L)^-1/s` (order two); `rate_unit(order)` supplies the exact string for any
nonnegative integer order. A non-elementary empirical rate law must not be
encoded by pretending its stoichiometry defines the kinetic orders. No Hill,
Michaelis–Menten, activity, diffusion or transport law is inferred.

`conservation_basis()` computes the exact rational left nullspace of `S` by row
reduction. Its basis may contain signed combinations and is not automatically a
set of physical moieties. `pool_residuals(weights)` independently checks a named
integer/rational pool per reaction. In `2A -> D`, `A + 2D` has residual zero,
whereas `A + D` has residual -1. Those are exact structural statements, distinct
from numerical drift along an integrated trajectory. Elemental composition,
charge and mass are not known unless separately supplied and checked as pools.
The basis does not prove that the asserted reactions occur in a real system.

`to_graph()` produces a **declarative bipartite graph** with species and reaction
nodes and reactant/product edges. Coefficients, rates, context and provenance
survive export; catalysts occur on both sides. Prefixes disambiguate a species
and a reaction with the same identifier. The exported graph does not register
executable callbacks and is not an SBML representation or round-trip importer.

## Numerical execution and sensitivity

The solver is from-scratch classical fixed-step RK4. Each requested observation
interval is split into equal steps no larger than `max_step_s`. It neither clips
negative concentrations nor returns a partial trajectory as success: negative or
nonfinite intermediate/final states raise `ValueError`. `max_steps` defaults to
one million for each `simulate` call, bounding accidental requests for enormous
work. This is not an adaptive error estimate; rerun at smaller steps and verify
convergence for the model and outputs being tested. Stiff networks need another
solver. Large networks can also make exact rational elimination expensive.

Forward sensitivity for reaction j is `dC/dm_j` at `m_j=1`, where `m_j` multiplies
that reaction's **effective rate after `rate_scales`**. It is not an absolute-rate
derivative. The augmented equation is `dZ/dt = J(C) Z + S[:,j] v_j`, with zero
initial sensitivity and the analytic product-rule Jacobian. This avoids dividing
by concentration at zero. Sensitivities have concentration units because the
multiplier is dimensionless. A zero effective rate gives zero multiplier
sensitivity; it does not imply that adding a previously absent reaction is
irrelevant. Initial conditions are held fixed. No parameter fitting, uncertainty
propagation, global sensitivity or identifiability certification is performed.

## Ordered interventions

Each phase first applies `C_after = retain * C_before + add`, then evolves for its
specified duration. Omitted retain/add entries are one/zero. A phase may have zero
duration to record an instantaneous event. `rate_scales` are relative to network
base rates in **each phase**; omitted entries reset to one, never inherit a prior
phase's setting. Phase names must be unique. Every returned `PhaseResult` records
`before`, `after`, `start_s`, the phase definition/provenance and a trajectory with
local time. Additions are mol/L, with the same fixed compartment volume throughout.

External events can deliberately change a conserved pool. Check its dynamics
against the **post-event** pool value and account for the jump separately. In the
example, `A+B` changes from 2 to 1.3 mol/L at the exchange and remains 1.3 during
the following dynamics. A protocol does not expose sensitivities: derivatives
through interventions or across phases have not been implemented.

Equal scalar retention commutes with any homogeneous linear flow:
`exp(Mt) (q I) C = (q I) exp(Mt) C`. The exact null case is useful for checking a
claimed intervention-order effect. It does not extend to selective retention,
nonlinear reactions, differing schedules or nonlinear/state-dependent growth. Tests include both
selective linear and scalar nonlinear counterexamples. These are mathematical
boundaries, not comparative treatment recommendations.

Keep a network definition fixed after construction; its public definitions and
cached matrices are not a supported mutation interface. Make a new network for
changed equations, or use explicit rate scales for parameter interventions.

## Validation and current boundary

Run the report and tests with:

```bash
PYTHONPATH=. python scripts/validate_reaction_experiments.py
PYTHONPATH=. python -m pytest -q tests/test_reaction_experiments.py
```

The [report](../auto-research/2026-11-05-E-reaction-experiments/report.md) records
closed-form linear and dimerization comparisons, independently written nonlinear
binding equations integrated by SciPy DOP853, central-difference sensitivity
checks, step convergence and intervention null/counterexamples. No model was fit.
This is an incremental Track E contribution: there is no BioModels SBML import,
real biological inconsistency claim, patient simulation or validated clinical use.
