# Auto-Research Experiment `2026-11-06-E-competitive-binding-envelopes`: Competitive Two-Ligand Groove Occupancy, Shared-Core Non-Identifiability, and Deterministic Evidence Corner Envelopes

## Question

When two ligands compete for a shared receptor or MHC class II groove (for example, paralogous CDR2L and CDR2 peptides sharing an interior 9-mer binding core as established in `project-asma` assay `18456218`), how do we:
1. Compute exact competitive equilibrium and transient Motulsky-Mahan competitive binding kinetics without single-ligand approximations?
2. Characterize the exact concentration locus across which two cross-reactive ligands produce an indistinguishable bulk groove-occupancy readout?
3. Evaluate how source-reported parameter intervals (`Uncertainty.low`, `Uncertainty.high`) propagate across the $2^d$ corners of an evidence-bound model to detect when an output conclusion straddles a decision threshold, without silently treating interval metadata as a probability distribution?

## Implementation

1. **`vh/bridge/occupancy.py`**:
   - `competitive_equilibrium_occupancy(cu_a_nm, kd_a_nm, cu_b_nm, kd_b_nm)`: exact Gaddum two-ligand competitive binding equilibrium returning `occupancy_a`, `occupancy_b`, `occupancy_total`, and `free_receptor_fraction`.
   - `shared_core_identifiability_locus(target_total_occupancy, kd_a_nm, kd_b_nm, cu_a_nm)`: solves the affine equivalence locus $C_A / K_{d,A} + C_B / K_{d,B} = O_{\text{total}} / (1 - O_{\text{total}})$ for the concentration $C_B$ that yields identical total groove occupancy alongside $C_A$.
   - `competitive_kinetic_occupancy(...)`: exact piecewise $2 \times 2$ Putzer matrix-exponential propagator for coupled Motulsky-Mahan competitive kinetics, bounded in $(0, 1]$ for all $\Delta t \ge 0$ and smooth across the $\Delta \to 0$ equal-rate limit.
2. **`vh/graph/evidence.py`**:
   - `OutputEnvelope` and `execute_evidence_envelope(registry, edge, evidence, specs, *, allow_synthetic=False, thresholds=None)`: fail-closed validation that every bound input carries explicit `[low, high]` endpoints bracketing `estimate.value` (`low <= estimate.value <= high`) and satisfying `PortSpec` bounds before invoking the registered model at the nominal point and all $2^d$ distinct interval corners, classifying any supplied output decision threshold as `"above_across_envelope"`, `"below_across_envelope"`, or `"straddles_threshold"`.

## Three-Authority Validation (`scripts/validate_competitive_binding_and_envelopes.py`)

| Validation authority | Scope checked | Quantitative result |
| --- | --- | --- |
| **Authority 1: Exact rational arithmetic (`fractions.Fraction`)** | Three competitive equilibrium cases, five shared-core equivalence pairs at $O_{\text{total}} = 3/4$, and 8-corner ($2^3$) competitive evidence envelope bounds (`3/15` to `24/27` for $O_A$; `5/9` lower bound for $O_{\text{total}}$) | Maximum absolute error **$1.11 \times 10^{-16}$** (1 float ulp; **$0.0$** on rational-denominator cases and locus pairs) |
| **Authority 2: Elementary mass-action network (`vh.graph.reactions.ReactionNetwork`)** | Three-species elementary competitive network (`R`, `RL_A`, `RL_B`) with exact rational left-nullspace conservation basis and RK4 trajectory (`dt = 0.001 h`) | Exact rational conservation basis `R + RL_A + RL_B = 1` (per-reaction residual `0`); max conservation drift **$6.00 \times 10^{-15}$**; trajectory agreement vs. Putzer propagator **$7.83 \times 10^{-12}$** |
| **Authority 3: Adaptive 8th-order Runge-Kutta (`scipy.integrate.solve_ivp` `DOP853`)** | 301-point transient competitive overshoot trajectory ($C_A = 30\text{ nM}, K_{d,A} = 10\text{ nM}, k_{\text{off},A} = 0.2\text{ h}^{-1}$ vs. fast competitor $C_B = 40\text{ nM}, K_{d,B} = 40\text{ nM}, k_{\text{off},B} = 4.0\text{ h}^{-1}$) | Max absolute error **$1.98 \times 10^{-15}$** ($O_A$) and **$2.69 \times 10^{-14}$** ($O_B$); fast competitor $B$ peaks at **$0.4280$** ($t = 0.40\text{ h}$) before decaying to its **$0.2000$** equilibrium (**$+0.2280$** kinetic overshoot) |

## Reproduce

```sh
python3 scripts/validate_competitive_binding_and_envelopes.py
python3 scripts/validate_evidence_bindings.py
pytest -q -m "not network"
node web/test_engines.mjs
```

## Limits

- `execute_evidence_envelope` evaluates the $2^d$ corners of source-reported input intervals; it is exact for componentwise monotonic functions on the hyperrectangle (such as Gaddum competitive occupancy with respect to each concentration and dissociation constant), does not bound interior extrema of non-monotonic functions, and does not assign probability mass to the interval.
- `competitive_kinetic_occupancy` assumes excess/buffered free ligand concentrations on each step interval $[t_{i-1}, t_i]$; when receptor concentration is comparable to total ligand concentration (ligand depletion), use `vh.graph.reactions.ReactionNetwork` with explicit free-ligand species instead.
