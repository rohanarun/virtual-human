# Conserved-flow graphs: from topology to a validated venous pressure prediction

- **Track**: E — Grow the graph formalization
- **Author**: OpenCode agent (DeepSeek V4.1 Flash Free), directed by Rohan Arun
- **Date**: 2026-11-08
- **Branch / PR**: `auto/E-vein-flow-graph`
- **Code added**: `vh/graph/flow.py`, `vh/graph/algorithms.py` (weighted
  Laplacian, effective resistance, current flow), `vh/graph/render.py` (conduit
  labels/styles), `tests/test_flow.py`, `scripts/validate_flow.py`,
  `docs/bernoulli_veins.md`, `docs/prediction_methods.md`,
  `docs/validation_flow.md`, `docs/validation_flow.json`,
  `docs/flow_explained.md`, `scripts/plot_flow_explainer.py`,
  `docs/img/womersley_vs_lumped.png`,
  `docs/img/vein_pressure_profile.png`, `docs/img/vein_graph.png`,
  `docs/img/flow_explained_venous_pressure.png`,
  `docs/img/flow_explained_terms.png`,
  `docs/img/flow_explained_scorecard.png`.
- **Addendum code** (beating heart + drug transport, same branch):
  `vh/graph/circulation.py`, `vh/graph/uptake.py`,
  `tests/test_circulation.py`, `scripts/validate_circulation.py`,
  `docs/circulation.md`, `docs/validation_circulation.{md,json}`,
  `docs/img/circulation_wiggers.png`, `docs/img/circulation_sweeps.png`,
  `docs/img/circulation_beat_flows.png`, `docs/img/uptake_overlay.png`.

## Question

Can a typed graph do more than say *what is connected*? Can it predict real
pressures and flows — specifically, can it predict venous pressure and re-derive
Bernoulli's principle, including the terms where classical Bernoulli fails? The
existing graph layer answered structural questions exactly; this contribution
adds the constitutive physics that makes the same graph quantitative.

## Method

### The graph

`vh/graph/flow.py` builds a typed graph from `FlowNodeSpec` and `VesselSpec`
lists. A node is a `reservoir` (fixed pressure), `compartment` (finite
compliance), or `junction` (zero compliance). A conduit edge carries:

- `R = 8μL/(πr⁴)` (Hagen–Poiseuille; explicit value allowed),
- `L = ρL/A` (inertance),
- `C = 2πr³L/(Eh)` assigned to a compartment endpoint (thin-wall hoop strain),
- `h = ρg(z_dst-z_src)` (gravity head),
- areas for the Bernoulli kinetic term and minor losses,
- optional `collapsible`/`external_pressure_pa` (Starling resistor) and `valve`
  (one-way) behaviour.

The solver enforces conservation at every node and the integrated momentum
balance on every edge. Steady linear networks are solved as one square linear
system; kinetic/minor terms use a nonlinear root finder; collapsible veins and
valves use an active-set complementarity loop (open edges obey the pressure-flow
law, closed edges carry zero flow, and edges open/close when the driving pressure
or flow changes sign). Transients use an exact matrix exponential for linear R–C
networks (all `L=0`, positive compliance) and implicit Euler otherwise.

### The independent referees

1. **Exact analytic solutions**: Hagen–Poiseuille, the Windkessel exponential
   `qR(1-e^{-t/RC})`, the R–L step `(Δp/R)(1-e^{-Rt/L})`, and the Starling
   waterfall `max(0,(p_up-max(p_down,p_ext))/R)`.
2. **The exact Womersley solution** of the pulsatile Navier–Stokes equations for
   a rigid circular tube:
   `Z = iωρL/(A F)`, `F = 1 - 2J₁(z)/(zJ₀(z))`, `z = α√(-i)`. This is
   independent of the graph's lumped `R+iωL` edge and is used to measure where
   that edge stops being valid.
3. **Physiological reference ranges** for venous pressure, used only after the
   solve and never to set a parameter: supine foot/ankle 5–10 mmHg, standing
   85–90 mmHg.

### The venous network

A lower-limb network with 9 nodes and 9 valved conduits: foot → ankle → calf →
popliteal → femoral → iliac → IVC → right atrium, plus a superficial saphenous
path from ankle to femoral. Geometry is anatomical (1.75 m person, right atrium
1.05 m above the floor, ankle 0.06 m), not fitted to the pressure target. The
capillary inflow (0.5 L/min per leg) is an input. Posture is modeled by rotating
the height coordinate, `z(θ)=z₀ sin θ`; the pressures are then predictions.

## Validation — the part that matters

| Quantity | This work | Independent authority | Error |
|---|---|---|---|
| Steady Poiseuille flow on a graph | 6.6667×10⁻⁴ m³/s | Δp/R | **1.6×10⁻¹⁶ relative** |
| Gravity-driven flow, 1 m column | 1.0395×10⁻² m³/s | ρgΔz/R | **0 absolute** |
| Bernoulli stenosis pressure | 1328.14 Pa | Rq + ½ρq²(1/A₂²−1/A₁²) | **0 relative**; Bernoulli residual 0 Pa |
| Starling waterfall, 4 regimes | 1.0/1.5/0/0 ×10⁻³ m³/s | analytic waterfall | **2.2×10⁻¹⁹ m³/s max absolute** |
| Series/parallel equivalent resistance | 5.0/1.0 MΩ | Laplacian / circuit rules | **2.3×10⁻¹⁰ Pa s/m³** |
| Windkessel transient, exact | 993.262053 Pa | qR(1−e^{−t/RC}) | **0 Pa** |
| Windkessel, implicit Euler | 993.09 Pa at dt=1 ms | same | 0.169 Pa; observed order **1.01** |
| R–L step response | 9.93245×10⁻⁶ m³/s | (Δp/R)(1−e^{−Rt/L}) | **1.7×10⁻⁵ relative** |
| Lumped R–L vs exact Womersley | — | exact Womersley Z(ω) | <1% up to **α=0.48**; 4.1% at α=1; 20.7% at α=5 |
| Venous pressure, supine | foot **8.4** mmHg, ankle 7.2 | 5–10 mmHg | inside range |
| Venous pressure, standing | foot **88.7** mmHg, ankle 84.4 | 85–90 mmHg | foot inside; ankle 0.6 mmHg below |
| Standing−supine hydrostatic difference | **80.3** mmHg | ρgΔz (81.9 mmHg for 1.05 m) | 1.6 mmHg |
| Effective resistance cross-check (0° network) | Δp/q = 5.376338×10⁷ | Laplacian R_eff | **3.5×10⁻¹⁵ relative** |
| Extended Bernoulli on every vein edge | edge momentum balance | **2.0×10⁻¹² Pa** max residual |

**Blind prediction.** No parameter was fitted to the venous pressure reference.
The geometry and capillary inflow were fixed from anatomy; the reference range
was consulted only after the solve. The standing foot pressure landed inside the
published range, and the supine value landed inside its range. This is the
successful prediction the contribution is built around.

**What is new vs reused.** The flow graph, solvers, Bernoulli/Womersley helpers,
venous network, tests, and validation script are new. The typed `Graph`, schema
machinery, renderer, and the PBPK graph are reused unchanged (apart from adding
conduit labels/styles to the renderer).

## Limits

- **The venous prediction is steady-state.** It does not model the calf muscle
  pump or ambulatory venous pressure (the post-contraction fall to ~20–30 mmHg).
  That needs transient collapse and valve opening/closing.
- **Transient valve/collapse is not implemented.** The steady solver uses
  active-set complementarity; the transient solver raises `NotImplementedError`
  for these edges. A closed valve can isolate a segment whose pressure is set by
  stored volume, which the current steady formulation does not represent.
- **The lumped R–L edge is a low-α model.** It is within 1% only for α≤0.48;
  large veins at 1 Hz (α≈7–14) have 12–18% pulsatile-impedance error. The steady
  pressure result is unaffected because steady flow has ω=0. The next rung is a
  frequency-domain graph solve with the exact Womersley `Z(ω)` on each edge.
- **Geometry is approximate.** Segment lengths/radii are order-of-magnitude
  values; the reference ranges are broad. The result establishes the hydrostatic
  mechanism and the solver's correctness, not patient-specific accuracy.
- **Blood is Newtonian by default.** At low shear in small veins the apparent
  viscosity is higher; the API accepts an explicit resistance/viscosity.
- **No wave propagation.** A node is a lumped compliance, not a distributed
  tube. Wave speed and reflection sites need the 1-D PDE rung.
- **Active-set is a local method.** It converged on all tested Starling/valve
  cases, but a non-smooth complementarity problem can have multiple solutions;
  the solver reports the one reached from the linear seed.

## Reproduce

```bash
pip install -e ".[dev]"                 # numpy, scipy, ...
python scripts/validate_flow.py          # writes validation_flow.json + 3 figures
pytest -q tests/test_flow.py             # 24 tests against exact authorities
pytest -q                                # full suite: 68 passed, 6 skipped
```

The validation report is `docs/validation_flow.json`; the derivation is
`docs/bernoulli_veins.md`; the method ladder is `docs/prediction_methods.md`.

## References

- Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow
  and viscous drag in arteries when the pressure gradient is known.
  *Journal of Physiology* 127(3), 553–563.
- Nichols, W. W., O'Rourke, M. F., & Vlachopoulos, C. (2011).
  *McDonald's Blood Flow in Arteries*, 6th ed.
- Fung, Y. C. (1997). *Biomechanics: Circulation*, 2nd ed. Springer.
- Shapiro, A. H. (1977). Steady flow in collapsible tubes.
  *Journal of Biomechanical Engineering* 99(3), 126–147.
- Guyton, A. C., & Hall, J. E. *Textbook of Medical Physiology*, 13th/14th ed.
  (venous pressure in the standing and recumbent positions).
- Pedley, T. J. (1980). *The Fluid Mechanics of Large Blood Vessels*.
  Cambridge University Press.

---

# Addendum (2026-11-10): the beating heart, and the drug on its flow

## Question

The flow graph so far answered steady questions: what pressure does a given
network produce, where does a flow go. The next question is dynamical and
coupled: **can the same conserved-flow graph carry a beating four-chamber heart
and the closed circulation, and can the drug mass balance of the validated PBPK
engine ride the flows that heart actually produces?** Concretely: does a
circulation assembled entirely from published components predict the normal
cardiac output, pressures and volumes without fitting any output — and how much
of a 24 h plasma profile is changed by the *pulse*, as opposed to the
cycle-mean flow?

## Method

- **Heart.** Four chambers with time-varying elastance
  `p_i = E_i(t)(V_i − V0_i)`, `E_i` a double-Hill activation (Suga & Sagawa;
  Stergiopulos). Emax/Emin/V0 and chamber volumes are the published normal
  values (Broomé et al. 2013, Table 1); nothing is fitted to the outputs.
  Parameters: LV 2.8/0.06/15, RV 0.55/0.04/10, LA 0.30/0.07/5, RA 0.22/0.05/8;
  HR 70.
- **Valves.** Four one-way Bernoulli orifices
  `q = A_eff √(2 max(Δp,0)·133.32/ρ)`, exactly zero reversed; published normal
  areas 5/3.5/6/3.5 cm², Cd 0.90.
- **Pools and beds.** Four elastic pools (systemic arteries 1.6, systemic veins
  70, pulmonary arteries 6, pulmonary veins 12 mL/mmHg) and the twelve ICRP-89
  organ beds, each `R_t = (MAP_ref − RAP_ref)/(share_t · CO_ref)` with
  MAP_ref 87, RAP_ref 4.5 mmHg, CO_ref 5 L/min, blood 5.3 L. Venous return is a
  Starling waterfall; the pulmonary bed is bidirectional.
- **Solver.** One closed graph, node balance `dV/dt = Σq_in − Σq_out`, RK4.
  `ΣV` is a linear invariant, so conservation and equal ventricular stroke
  volumes are identities, not tuning targets.
- **Drug layer.** `vh.graph.uptake` keeps the validated PBPK state, volumes,
  partition coefficients, clearance and oral absorption exactly; only the flows
  change. Each segment's transport matrix is exponentiated exactly on a 4 ms
  grid, and full beats regroup into a matrix power of the beat propagator, so a
  24 h beat-coupled simulation takes ~0.2 s. At ICRP reference flows the
  transport matrix is identical to the PBPK engine's, which anchors the layer.

## Validation — the part that matters

The baseline is compared with the *published normal column* of Broomé et al.
2013 Table 1 (a different model's normal values), graded by distance in
population SD: ≤1 SD pass, ≤3 SD note.

| Quantity | This work | Broomé 2013 Table 1 | Distance |
|---|---|---|---|
| Cardiac output | **6.36 L/min** | 6.3 ± 2.4 | +0.02 SD |
| Stroke volume | **90.8 mL** | 89 ± 30 | +0.06 SD |
| Ejection fraction | **63.4 %** | 59 ± 4 | +1.10 SD (note) |
| LVEDV / RVEDV | **143.2 / 138.5 mL** | 123 ± 28 / 111 ± 22 | +0.72 / +1.25 SD (RV note) |
| Mean RAP / LAP | **3.46 / 5.91 mmHg** | 4.8 ± 2.4 / 8.3 ± 2.5 | −0.56 / −0.96 SD |
| Aortic S/D | **107.3 / 69.6 mmHg** | 112/74 ± 12 | −0.39 / −0.36 SD |
| Pulmonary artery S/D | **21.6 / 12.4 mmHg** | 22 ± 6 / 9 ± 5 | −0.07 / +0.68 SD |
| MAP / pulse pressure | **90.5 / 37.7 mmHg** | — | derived |
| LV dP/dt max | **1471 mmHg/s** | 1200–2200 | inside |
| Aortic valve open time | **0.296 s** | ~0.3 s systole at HR 70 | inside |

Every quantity is within one population SD except EF and RVEDV (1.1–1.25 SD);
all are within three. The Wiggers morphology — atrial kick, isovolumic
contraction, ejection, incisura, E and A waves — is reproduced.

Identities of the equations themselves (machine precision, not fits):

| Identity | Measured |
|---|---|
| Total blood-volume drift over 8 recorded beats | **1.6×10⁻¹¹ mL** |
| Graph blood volume == ICRP 5.3 L | **6.1×10⁻¹¹ mL** |
| LV and RV eject identically (one closed loop) | 1.0×10⁻⁴ relative |
| Valve backflow | **exactly 0** mL/s |
| Instantaneous bed split == ICRP shares | **2.2×10⁻¹⁶** relative |

Physiological sweeps reproduce the published directions: +10 % blood volume →
CO 8.79 L/min and EDV 193 mL (Frank–Starling); −10 % → CO 5.10; halved Emax →
EF 49 %, EDV 159 mL, LAP 6.4 mmHg (systolic-failure direction); Emax ×1.3 →
EF 68 %, CO 6.60; HR 100 → SV 69.9 mL, CO 6.99 L/min. These are **open-loop**
(no baroreflex, no venoconstriction), so the −20 % hemorrhage case
(CO 1.20 L/min, MAP 17 mmHg) exaggerates the collapse of an intact animal; the
sweep verifies mechanism and direction, not compensated magnitude.

Drug layer, against the validated PBPK engine and exact algebra:

| Check | Measured |
|---|---|
| Transport matrix at ICRP flows == validated PBPK matrix | **0.0** max abs difference |
| Pulsatile solver on a flat flow template == mean solver | **1.3×10⁻¹⁰** relative |
| Mean-stream solver == PBPK engine (24 h, IV + oral) | **1.2×10⁻¹³** relative |
| 24 h beat-coupled mass conservation (IV 5 mg + oral 7.5 mg) | **1.6×10⁻¹⁰ mg** |
| Pulsatile 24 h infusion integral (10 mg/h → 240 mg) | 5.5×10⁻⁷ of delivered |
| Steady state == engine's 96 h infusion plateau | 2.5×10⁻⁷ relative |
| Plasma C_ss vs scalar R/CL_total | 7.8×10⁻⁵ low |
| Discretization converged (dt 4 → 2 → 1 ms) | 4.4×10⁻⁶ / 5.7×10⁻⁶ relative |

Clinical overlay: on the Olkkola 1993 control arms — the same study's IV and
oral datasets, each the one nearest its route's panel-median AUC ratio, chosen
a priori, not by looking at the curves — the pulse-coupled predictions give AUC
pred/obs **1.39** (IV, mean-stream 1.25) and **1.07** (oral, mean-stream 1.15),
with AAFE 1.45 and 1.79. The flow-grain change stays inside the mean engine's
accuracy band.

**The one prediction with no external authority.** How much does the *waveform*
of the flow, rather than its cycle mean, change a plasma profile? There is no
textbook value to check this against, so it is measured with both solvers locked
to the *same* cycle-mean flows: the 24 h AUC of a combined IV + oral midazolam
profile changes by **−7.1 %**. It converges to 0.00 % under dt halving. This
number is a quantitative prediction of this layer, not a validated fact; the
mean-heart flows themselves differ from ICRP reference flows by only +0.28 %,
and the within-beat plasma swing is +2.7 % in a settled beat.

## Limits

- **Open-loop circulation.** No baroreflex, chemoreflex, venoconstriction,
  pericardial or intrathoracic pressure. Normal resting circulation only.
- **Ideal valves and blood.** Bernoulli orifices, no leaflets or regurgitation;
  Newtonian blood; lumped compliances, no distributed waves.
- **Constant PBPK volumes in the drug layer.** Only flows are modulated; the
  circulating pool *volume* waves live in the hydraulic layer. This isolates the
  flow-waveform effect and is a documented next coupling step.
- **Oral absorption is not flow-coupled.** The validated stomach/intestine chain
  is retained; villous blood flow is not pulsed.
- **Boluses land on the 4 ms transport grid** (within one segment of nominal).
- **Hepatic extraction.** Bottom-up in-vitro CLint gives E_h 0.284, just below
  the published 0.30–0.50 band; the one-factor middle-out calibration gives
  0.309. Both are reported.
- **The waveform effect has no independent referee.** Its convergence and its
  anchoring to identical mean flows are what is checked; its magnitude is not
  externally validated.
- **Tissue concentration swings within a beat are not reported as an
  observable.** In the first seconds after a bolus, far-from-equilibrium
  compartments swing by >200 %; only the plasma swing (+2.7 %) is a sampled,
  comparable quantity.

## Reproduce

```bash
python scripts/validate_circulation.py   # writes validation_circulation.{json,md} + 4 figures
pytest -q tests/test_circulation.py      # 27 tests
pytest -q                                # full suite: 68 passed, 6 skipped
```

Reports: `docs/circulation.md` (model, equations, limits),
`docs/validation_circulation.md` / `.json` (measured numbers). Figures:
`docs/img/circulation_wiggers.png`, `circulation_sweeps.png`,
`circulation_beat_flows.png`, `uptake_overlay.png`.

## Sources (addendum)

- Suga, H., & Sagawa, K. (1974). Instantaneous pressure–volume relationships
  and their ratio in the excised, supported canine left ventricle.
  *Circulation Research* 35(1), 117–126.
- Stergiopulos, N., Meister, J. J., & Westerhof, N. (1996). Determinants of
  stroke volume and ejection fraction: a closed-loop analysis. *American
  Journal of Physiology* 270(1), H205–H212.
- Broomé, M., Maksuti, E., Bjällmark, A., Frenckner, B., & Janerot-Sjöberg, B.
  (2013). Closed-loop real-time simulation model of hemodynamics and oxygen
  transport. *BioMedical Engineering OnLine* 12:69, Table 1.
- International Commission on Radiological Protection (2002). *Basic
  Anatomical and Physiological Data for Use in Radiological Protection:
  Reference Values*. ICRP Publication 89. Ann. ICRP 32(3–4).
- Olkkola, K. T., Aranko, K., Luurila, H., Hiller, A., Saarnivaara, L., Himberg,
  J. J., & Neuvonen, P. J. (1993). A potentially hazardous interaction between
  erythromycin and midazolam. *Clinical Pharmacology & Therapeutics* 53(3),
  298–305.
- Thummel, K. E., et al. (1996). Human liver midazolam 1′-hydroxylation:
  in vitro–in vivo scaling. *Drug Metabolism and Disposition* 24(9),
  1000–1008.
