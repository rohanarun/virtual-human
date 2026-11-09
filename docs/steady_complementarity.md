# Steady flow with valves and collapse: existence, uniqueness, and a finite solver

This note gives the mathematics behind `FlowNetwork.steady_state()` for networks
with one-way valves and collapsible (Starling) vessels, and states exactly what is
proved, what is only checked numerically, and where the model itself breaks.
Tests: `tests/test_flow_complementarity.py`.

## 1. The problem

A network has reservoirs `S` (fixed pressure `p_r`), junctions/compartments `J`
(unknown pressure, source `s_j`), and edges `E` with incidence `B`
(`B[src,e] = -1`, `B[dst,e] = +1`).  Edge `e` carries flow `q_e` and has the law

    p_src - p_dst - h_e = phi_e(q_e),        phi_e(q) = R_e q + c_e q|q| + k_e q|q|

(`R_e > 0` Poiseuille, `c_e` the Bernoulli kinetic term, `k_e >= 0` minor loss,
`h_e` gravity head).  At steady state every junction balances, `(B q)_j + s_j = 0`.

A **valve** edge `v` additionally allows only forward flow.  With the blocking
pressure `w_v = phi_v(q_v) - (p_src - p_dst - h_v)` the valve law is the
complementarity condition

    q_v >= 0,    w_v >= 0,    q_v * w_v = 0.                            (C)

An open valve has `w_v = 0`.  A closed valve has `q_v = 0` and holds back
`w_v >= 0`.

## 2. Valves with monotone losses: a unique flow solution

**Theorem 1.** Let every `phi_e` be continuous and strictly increasing with
`phi_e(0) = 0` and `|phi_e(q)| -> infinity` as `|q| -> infinity`.  Assume the
balance equations have a solution with `q_V >= 0` (for example, every junction
reaches a reservoir through two-way edges).  Then the steady problem with valves
(C) has a solution, and its **flow vector is unique**.  If every junction is
connected to a reservoir by two-way edges, the pressures are unique too.

*Proof.* Let `Phi_e(q) = int_0^q phi_e` (strictly convex and coercive). Minimise
the content

    F(q) = sum_e [ Phi_e(q_e) + h_e q_e ] + sum_{r in S} p_r (B q)_r

subject to the linear constraints `(B q)_j + s_j = 0` for `j in J` and `q_v >= 0`
for `v in V`.  `F` is strictly convex and coercive on a nonempty polyhedron, so
there is exactly one minimiser `q*`.  The constraints are linear, so the KKT
conditions are necessary and sufficient.  Write `lambda_j` for the multipliers of
the balance rows and `mu_v >= 0` for the sign constraints.  Stationarity in `q_e`
reads

    phi_e(q_e) + h_e + sum_{i} p_i B[i,e] - mu_e = 0 ,    with p_j := lambda_j on J,

and `sum_i p_i B[i,e] = p_dst - p_src`.  So `phi_e(q_e) = p_src - p_dst - h_e + mu_e`:
for a two-way edge (`mu_e = 0`) this is the edge law, and for a valve `mu_v = w_v`.
`mu_v >= 0`, `q_v >= 0` and `mu_v q_v = 0` are exactly (C).  The KKT system is
therefore *identical* to the steady equations, and the multipliers are the junction
pressures.  Any steady solution is a KKT point, hence equals the minimiser: the
flows are unique.  If two-way edges connect every junction to a reservoir, the
pressures follow uniquely from the flows along those edges.  ∎

Without the connectivity assumption the pressures can be non-unique.  For example,
a junction whose edges are all closed valves may sit at any pressure that keeps
them closed.  The flows are still unique.

## 3. Linear valves: a P-matrix LCP and a finite pivoting rule

Take linear edges (`phi_e = R_e q`), and let two-way edges connect every junction
to a reservoir.  Eliminate the junction pressures through the grounded Laplacian
`L_J = B_{J,T} R_T^{-1} B_{J,T}^T` of the two-way subnetwork `T`; `L_J` is symmetric
positive definite under the connectivity assumption.  The blocking pressures
become an affine function of the valve flows,

    w = M q_V + c,     M = R_V + B_{J,V}^T L_J^{-1} B_{J,V},

with `M` symmetric positive definite (`R_V` is a positive diagonal matrix, the
second term is PSD).  So (C) is the linear complementarity problem `LCP(c, M)`
with a P-matrix: it has exactly one solution for every `c`.

`steady_state()` uses single principal pivots with **Murty's least-index rule**:
1. Solve the equations of the current open/closed pattern.
2. Among the valves whose state contradicts the solution (open with `q < 0`, or
   closed with positive driving pressure), flip only the one with the lowest
   index.

For P-matrix LCPs this terminates after finitely many pivots (Murty 1974), and the
final pattern gives the unique solution.

The previous loop flipped *every* violating valve at once (block pivoting) and
stopped after 20 sweeps.  Simultaneous flipping has no general finiteness
guarantee, unlike the single least-index pivot.  The new
loop also stops at a cap (`50 + 10 k` pivots), but then it reports
`converged = False` instead of returning the last iterate.

**What is not covered.** If some valve pattern cuts a junction off from every
reservoir, the elimination above does not exist for that pattern.  Then the
finiteness argument does not apply, although Theorem 1 still gives unique flows.

## 4. The convergence test, and the bug it fixes

The steady residual mixes units: junction rows are flows (m³/s) and edge rows are
pressures (Pa).  At physiological scale they differ by ~9 orders of magnitude
(`q ~ 1e-6 m³/s`, `p ~ 1e3 Pa`).  The previous code compared the raw residual
with one absolute number (`1e-7` inside the loop, `1e-6` for `converged`).  A
node imbalance of `3e-7 m³/s` passed as "zero", although it was 22% of the
largest flow.  `tests/test_flow_complementarity.py::regression_network` is such a
case: 4 nodes and 7 edges, flows wrong by 87%, reported `converged=True`.  The
final `max(q, 0)` clamp then hid the negative flows through closed valves.

Now every row is divided by its own scale: pressures by `P*`, flows by
`Q* = max(|s|, P*/median(R), |q|)`.  `converged` requires a scaled residual below
`tol` (default `1e-10`) **and** complementarity on every one-way edge.
`residual_norm` is the scaled (dimensionless) residual.

Measured against brute-force enumeration of all `2^k` valve patterns, on 6000
random networks (seeds 0–2, up to 10 nodes and 8 valves):

| solver | `converged=True` but wrong | worst flow error when "converged" |
|---|---:|---:|
| previous active-set loop (seed 0) | 39 / 2000 | 23% |
| this one (seeds 0–2) | 0 / 6000 | 8e-15 relative |

The headline venous network of `scripts/validate_flow.py` was **not** affected.
All its valves are open, and the first linear solve is already exact (foot
pressure 88.6688 mmHg standing, identical to 1e-16 with both solvers).

## 5. The kinetic term breaks monotonicity at an expansion

For an edge that widens (`A_dst > A_src`), the Bernoulli coefficient
`c = (alpha rho / 2)(1/A_dst² - 1/A_src²)` is negative (pressure recovery).  Then
`phi(q) = R q + c q|q|` increases only for `|q| < q_crit = R / (2|c|)` (with no
minor loss), and Theorem 1 no longer applies.  On a single such edge between two
reservoirs with `dp = p_src - p_dst > 0`:

- for `0 < dp < R²/(4|c|)` there are **three** steady states:
  `q = (R ∓ sqrt(R² - 4|c| dp)) / (2|c|)` forward, and one reverse root;
- for `dp > R²/(4|c|)` there is **no forward solution at all**, only reverse flow
  against the pressure difference.

With the code's constants (`mu = 3.5e-3 Pa s`, `rho = 1060 kg/m³`), a 10 cm vessel
widening from 3 mm to 6 mm radius has `q_crit = 8.9e-6 m³/s = 0.53 L/min`, which
is inside the physiological range.  It also has `R²/(4|c|) = 48.7 Pa`
(0.37 mmHg).  At `dp = 1000 Pa` the only steady state of the equations is a
reverse flow of about 3 L/min.  Both the old and the new solver return it as
converged, because it does satisfy the equations.  Between 49 and 100 Pa both
report `converged=False`.

This is a property of the lumped model, not of the solver.  Static-pressure
continuity at nodes, plus a lossless kinetic term, lets the recovered kinetic
pressure grow like `q²` while the viscous drop grows like `q`.  Fixing it is a
modelling decision: total-pressure nodes, a Borda–Carnot loss, or a bounded
recovery.  This change does not take that decision.  The solver starts Newton
from the Poiseuille solution, so it follows the physical branch while that branch
exists (`test_expanding_tube_follows_the_poiseuille_branch`).

## 6. Collapsible (Starling) edges: unique in every test, not proved

In `vh.graph.flow` a collapsible edge is one-way with
`q = max(0, (p_src - max(p_dst, p_ext) - h) / R)`.  It is piecewise linear with
three regimes: closed, open-normal, and open-waterfall.  Enumerating all `3^k`
regime combinations finds every steady state exactly.  On 6000 random networks
(seeds 0–2 of `probe_collapsible.py`, up to 4 collapsible edges plus valves)
every network had **exactly one** steady state, and the solver matched it to
`1e-8` relative.  The suite re-checks 100 such networks.

Theorem 1 does not cover this case: a waterfall edge has three terminals, and its
flow depends on absolute pressure through `p_ext`, not only on `p_src - p_dst`.
The maximum-principle argument also fails.  Shifting all pressures by `t > 0`
raises a waterfall's flow, so the junction map is not "diagonally dominant" under
uniform shifts.  So uniqueness here is a **conjecture backed by enumeration**,
not a theorem.

Modelling remark (unchanged here): the code treats collapsible edges as one-way,
so reverse flow through a collapsible vessel is forbidden.  In a Starling
resistor the flow can reverse, with the collapse then forming at the other end.
This assumption is not stated elsewhere in the docs.

## 7. Transients

`_nonlinear_step` (implicit Euler with kinetic/minor terms) had the same
mixed-unit absolute stopping test (`1e-10`), and it silently returned the last
iterate when Newton did not converge.  It now scales the rows the same way, and
it emits a `RuntimeWarning` on non-convergence.  The new test
`test_nonlinear_transient_relaxes_to_the_steady_state` guards this.  The old code
also passes it, so it is not a regression demonstration.

## References

- K. G. Murty, "Note on a Bard-type scheme for solving the complementarity
  problem", *Opsearch* 11 (1974) 123–130: finiteness of least-index principal
  pivoting for P-matrix LCPs.
- R. W. Cottle, J.-S. Pang, R. E. Stone, *The Linear Complementarity Problem*,
  Academic Press 1992: P-matrices and uniqueness of LCP solutions.
- W. Millar, "Some general theorems for non-linear systems possessing resistance",
  *Phil. Mag.* 42 (1951) 1150–1160: the content/co-content minimum principle for
  networks of monotone resistors.
