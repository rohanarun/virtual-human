# The beating heart, the closed circulation, and the drug riding it

`vh.graph.circulation` turns the conserved-flow machinery of
[`bernoulli_veins.md`](bernoulli_veins.md) into a **beating heart driving the
whole circulation**; `vh.graph.uptake` puts the drug mass balance of the
validated PBPK engine on the flows that heart actually produces. This page
explains the model and its limits; the measured numbers and their independent
authorities are in [`validation_circulation.md`](validation_circulation.md).

Reproduce:

```bash
python scripts/validate_circulation.py
pytest -q tests/test_circulation.py
```

## Structure

```
  atria ──valves──▶ ventricles ──valves──▶ arteries ──▶ 12 ICRP organ beds ──▶ veins ──▶ atria
     ▲                                                                                    │
     └──────────────────────── pulmonary: RV ─▶ lungs ─▶ LA ◀──────────────────────────────┘
```

One closed loop, eight storage nodes, four one-way valves, twelve organ beds.
Everything is a typed graph (`build_circulation_graph`) with a schema that makes
the structural claims checkable: every storage node has an inflow and an
outflow, the bed shares sum to 100 %, every resistance and valve area is
positive.

<p align="center">
  <img src="img/circulation_wiggers.png" width="720" alt="One cardiac cycle: chamber and arterial pressures, ventricular volumes, valve flows">
</p>

## The equations

Units: pressure mmHg, volume mL, flow mL/s, time s.

| Element | Law | Source of the parameters |
|---|---|---|
| chamber `i` | `p_i = E_i(t) (V_i − V0_i)`, `E_i` a double-Hill time-varying elastance | Suga & Sagawa 1974; Stergiopulos et al. 1992/1996; Emax/Emin/V0 from the published normal table |
| valve `e` | `q_e = A_eff √(2 max(p_src − p_dst,0) · 133.32 / ρ)`, exactly zero reversed | published normal valve areas (4–6, 3–4 cm²), Cd 0.90 |
| pool `j` | `p_j = (V_j − V0_j) / C_j` | compliance and blood-volume distribution of the published normal table |
| bed `t` | `q_t = (p_sa − p_sv) / R_t`, with `R_t = (MAP_ref − RAP_ref) / (share_t · CO_ref)` | ICRP 89 organ flow shares |
| venous return | `q = max(p_sv − p_ra, 0) / R` (Starling waterfall) | published caval pressure drop at rest |
| pulmonary bed | `q = (p_pa − p_pv) / R` (bidirectional) | published normal pulmonary vascular resistance |

The node balance is `dV/dt = Σ q_in − Σ q_out`, integrated with RK4. `Σ V` is a
linear invariant of the right-hand side, so total blood volume is conserved to
machine precision however the valves open and close — and because the two
ventricles sit in one loop, their stroke volumes must agree. Both are checked in
the tests: they are identities of the equations, not fitted behaviour.

## What is predicted, and against what

No parameter is fitted to an output. Each component is anchored to a published
source (valve areas, compliances, Emax/Emin, blood-volume split, ICRP shares),
and the **joint output** is compared with the published normal-physiology column
of Broomé et al. 2013 (Table 1), a different model's normal values:

| Quantity | Model | Published normal (Broomé 2013, Table 1) |
|---|---|---|
| cardiac output | **6.36 L/min** | 6.3 ± 2.4 |
| stroke volume | **90.8 mL** | 89 ± 30 |
| ejection fraction | **63 %** | 59 ± 4 |
| LVEDV / RVEDV | **143 / 139 mL** | 123 ± 28 / 111 ± 22 |
| aortic pressure | **107/70 mmHg** | 112/74 ± 12 |
| mean RAP / LAP | **3.5 / 5.9 mmHg** | 4.8 ± 2.4 / 8.3 ± 2.5 |
| pulmonary artery | **22/12 mmHg** | 22 ± 6 / 9 ± 5 |
| LV dP/dt max | **1471 mmHg/s** | 1200–2200 (Guyton; Braunwald) |
| aortic valve open time | **0.30 s** | ~0.3 s systole at HR 70 (Wiggers) |

Every quantity is within one population SD of the published column except EF
(+1.1 SD) and RVEDV (+1.25 SD); all are within three. The pressure and flow
waveforms have the published morphology: atrial kick, isovolumic contraction,
ejection, incisura, E-wave and A-wave.

<p align="center">
  <img src="img/circulation_sweeps.png" width="760" alt="Frank-Starling preload sweep and contractility sweeps">
</p>

The sweeps reproduce the published *directions*: preload raises output
(Frank–Starling), a halved Emax lowers EF and congests the left atrium, and
tachycardia shortens the beat and lowers stroke volume while output rises. They
are **open-loop** — no baroreflex, no venoconstriction — so a −20 % hemorrhage
collapses output further than an intact animal would; the sweep verifies the
mechanism, not the compensated magnitude.

## The drug layer: riding the beat

`vh.graph.uptake` does not invent a second pharmacokinetics. The state vector,
volumes, partition coefficients, clearance and oral absorption chain are exactly
those of the validated [`vh.body`](../vh/body) engine; only the **flows** change:

```
at ICRP reference flows:    transport matrix  ==  validated PBPK matrix   (exactly)
mean organ flows of the heart:  same matrix, heart-mean flows
beat-coupled:               q_i(t) piecewise-constant on a 4 ms grid, exact
                            matrix exponential per segment; whole beats regroup
                            into a matrix power of the full-beat propagator
```

Because every edge is a first-order transfer and eliminated amounts are states,
every transport matrix has exactly zero column sums: total drug is conserved by
each exponential step, and a 24 h beat-coupled run drifts by **1.6×10⁻¹⁰ mg**.
The whole 24 h run takes ~0.2 s.

<p align="center">
  <img src="img/circulation_beat_flows.png" width="720" alt="Flow waves around the closed loop and the within-beat plasma concentration">
</p>

The pools filter the ejection pulse: aortic flow peaks at ~7× its mean, but the
organ beds see only ±20 % and venous return less. The observable consequence at
the sampling site is small — the venous plasma concentration swings ~3 % within
a beat. Tissue compartments far from equilibrium in the first seconds after a
bolus swing much more, but that is a transient of the loading phase, not a
steady beat effect.

## The one prediction with no external authority

How much does the **waveform** of the flow, as opposed to its mean, change a
plasma profile? That question has no textbook answer to check against, so the
report measures it with the two solvers locked to the *same cycle-mean flows*:
the answer is **−7.1 % on the 24 h AUC** of a combined IV + oral midazolam
profile, converged to 0.00 % under dt halving. The mean engine's own
(V̇, not waveform) flows differ by +0.3 %, and the beat-coupled curves stay inside
the mean engine's clinical accuracy band on the representative datasets
(`validation_circulation.md`).

<p align="center">
  <img src="img/uptake_overlay.png" width="820" alt="Pulsatile and mean-stream predictions against digitized clinical plasma data">
</p>

## Limits

- **Open-loop circulation.** No baroreflex, no chemoreflex, no venoconstriction,
  no pericardial or intrathoracic pressure. The model is a normal resting
  circulation; responses to volume loss are the passive Starling response only.
- **Ideal valves and blood.** Valves are Bernoulli orifices (no leaflets, no
  regurgitation except the intentional pulmonary reverse flow); blood is
  Newtonian; no distributed waves (this is a lumped model, one compliance per
  pool).
- **Constant PBPK volumes in the drug layer.** The beat-coupled transport rides
  the PBPK engine's constant blood-pool volumes; only the flows are modulated.
  Circulating pool *volume* waves live in the hydraulic layer and are a
  documented next coupling step — this isolates the flow-waveform effect.
- **Boluses land on the 4 ms transport grid.** Within one segment (~4 ms) of the
  nominal time.
- **Oral absorption is not flow-coupled.** The stomach/intestine chain is the
  validated PBPK one; villous blood flow is not pulsed by the heartbeat. This is
  a neglected second-order coupling, documented rather than hidden.
- **Hepatic extraction** from the bottom-up in-vitro CLint is 0.284, just below
  the published 0.30–0.50 band (Thummel 0.44); the one-factor middle-out
  calibration reaches 0.309. The report states both.

## Files

- `vh/graph/circulation.py` — heart, valves, pools, beds, graph + schema, solver.
- `vh/graph/uptake.py` — transport matrix, beat algebra, pulsatile/mean solvers,
  organ uptake and extraction.
- `tests/test_circulation.py` — 27 tests: structural identities, published
  ranges, sweeps, and the drug-layer identities across both solvers.
- `scripts/validate_circulation.py` — writes `validation_circulation.json`,
  `validation_circulation.md` and the four figures on this page.
