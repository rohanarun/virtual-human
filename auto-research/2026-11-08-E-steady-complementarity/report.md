# Steady flow with valves: a unique solution, a finite solver, and an honest `converged`

- **Track**: E (graph formalization), follow-up to `2026-11-08-E-vein-flow-graph`
- **Author**: MikeMike88 with Claude Code (Claude Opus 5.5)
- **Date**: 2026-11-08
- **Branch / PR**: `auto/E-steady-complementarity` (based on `auto/E-vein-flow-graph`)
- **Code added**: `vh/graph/flow.py` (steady solver, transient Newton tolerance),
  `tests/test_flow_complementarity.py`, `docs/steady_complementarity.md`, probe scripts in
  this folder

## Question

`FlowNetwork.steady_state()` solves networks with one-way valves and collapsible
veins by an active-set loop.  Is its answer the right one, and does `converged=True`
mean anything?  What does the mathematics guarantee here?

## Method

1. **Referee by enumeration.** A valve is either open or closed, and a collapsible
   edge is closed, open-normal, or open-waterfall.  Every pattern is a linear system,
   so solving all `2^k` (or `3^k`) systems and keeping the self-consistent solutions
   finds **every** steady state exactly.  No iteration and no tolerance tuning are
   involved, and nothing from the solver under test is used.
2. **Theory** (`docs/steady_complementarity.md`).
   - The steady problem with valves and monotone losses is the KKT system of a
     strictly convex *minimum-content* program (Millar 1951).  So the flows are
     unique, and the junction pressures are its multipliers.
   - With linear edges it reduces to a P-matrix LCP, for which Murty's least-index
     single-pivot rule is finite.
3. **Solver changes.**
   - Residual rows are scaled by a characteristic pressure and flow.
   - Each pattern is solved by Newton with the analytic Jacobian; for linear edges
     the first step is exact.
   - Pivoting follows Murty's least-index rule instead of flipping all violators at
     once.
   - `converged` now requires a scaled residual below `1e-10` **and**
     complementarity on every one-way edge.
   - The transient implicit-Euler Newton step uses the same scaling, and it warns
     instead of silently returning a non-converged iterate.

## Validation

Authority: exact enumeration of all valve/collapse regimes (piecewise-linear
steady states), plus closed-form roots for the single expanding tube.

| Quantity | Before | After | Authority |
|---|---|---|---|
| random valve networks, `converged=True` but wrong | **39 / 2000** (seed 0) | **0 / 6000** (seeds 0–2) | enumeration of `2^k` patterns |
| worst flow error among "converged" | **23%** | **8e-15** relative | same |
| regression network (4 nodes, 7 edges) | flows off by **87%**, balance off by **22%**, `converged=True` | exact, residual 6e-17 | same |
| random collapsible networks | not separately measured | **6000 / 6000** unique steady states; solver matches to 1e-8 | enumeration of `3^k` regimes |
| headline venous network (all tilts) | foot 88.6688 mmHg | identical (to 2e-16) | enumeration |
| expanding tube, 0 < dp < R²/(4\|c\|) | Poiseuille branch | Poiseuille branch, error < 1e-10 | exact quadratic root |
| `tests/test_flow_complementarity.py` | 4 of 11 tests fail, 66 s | 11 / 11 pass, 3.7 s | — |

The original 24 flow tests and the graph tests still pass.  Full run:
`tests/test_flow.py tests/test_graph.py tests/test_flow_complementarity.py`, with
`-W error::RuntimeWarning`.

## Limits

- **Expansions make the lumped model ill-posed.**  For `A_dst > A_src` the edge law
  `R q + c q|q|` with `c < 0` stops increasing at `q_crit = R/(2|c|)`.  For a 10 cm
  vessel widening from 3 to 6 mm, `q_crit = 0.53 L/min`.  On a single such edge,
  `0 < dp < R²/(4|c|) = 48.7 Pa` gives three steady states.  A larger `dp` gives
  only reverse flow (about 3 L/min backwards at `dp = 1000 Pa`), and both solvers
  return it as converged.  This is a modelling issue (static-pressure nodes plus
  lossless pressure recovery), not a solver bug.  It is reported here and **not
  fixed**.
- **Starling uniqueness is a conjecture.**  It holds in every one of the 6000
  enumerated networks, but Theorem 1 does not cover three-terminal waterfall
  edges, and the maximum-principle proof fails for them (section 6 of the doc).
- **Murty finiteness needs a connectivity assumption.**  The proof needs the
  two-way edges to connect every junction to a reservoir.  Otherwise flows are
  still unique, but the pivot count is only capped (`50 + 10k`); a run that hits
  the cap reports `converged=False`.
- **Collapsible edges stay one-way, as before.**  Reverse flow through a
  collapsible vessel is forbidden.  This is a pre-existing modelling choice; it
  is now documented but unchanged.
- **The transient test is a guard, not a regression demonstration.**  It also
  passes on the old code.

## Reproduce

```bash
pip install -e . numpy scipy pytest
pytest -q tests/test_flow_complementarity.py tests/test_flow.py -W error::RuntimeWarning
cd auto-research/2026-11-08-E-steady-complementarity
python probe_active_set.py 0 2000 results_valves_s0.json     # enumeration vs solver
python probe_collapsible.py 0 2000 results_collapsible_s0.json
python expansion_roots.py                                    # three roots / no forward root
python check_veins.py                                        # headline network unchanged
```

## References

- W. Millar, *Phil. Mag.* 42 (1951) 1150–1160: content minimum principle for
  monotone resistor networks.
- K. G. Murty, *Opsearch* 11 (1974) 123–130: finiteness of the least-index
  principal pivoting rule for P-matrix LCPs.
- R. W. Cottle, J.-S. Pang, R. E. Stone, *The Linear Complementarity Problem*
  (1992).
