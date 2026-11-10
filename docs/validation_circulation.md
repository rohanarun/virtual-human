# Heart + circulation + the drug layer riding them: validation

`vh.graph.circulation` is a beating four-chamber heart (time-varying elastance), four one-way
valves, four elastic pools and the twelve ICRP-89 organ beds on one conserved-flow graph.
`vh.graph.uptake` puts the drug mass balance of the validated PBPK engine exactly on the flows
this heart produces.  This page records what was checked and the measured numbers; the raw
report is in [`validation_circulation.json`](validation_circulation.json).

Reproduce with:

```bash
python scripts/validate_circulation.py
pytest -q tests/test_circulation.py
```

## Headline

- Baseline (ICRP reference male, 5.3 L blood): **CO 6.36 L/min, SV 90.8 mL, EF 63%, LVEDV 143 mL, 107/70 mmHg, MAP 90.5, RAP 3.5, LAP 5.9, PAP 22/12 mmHg** -- every quantity within one population SD of the published normal column (Broomé et al. 2013 Table 1) except EF and RVEDV at ~1.1-1.25 SD; every output is a prediction from components anchored upstream, none is fitted.
- The whole 24 h beat-coupled drug transport runs in ~0.2 s (regrouped beat algebra).
- The only quantity with no external authority -- the effect of the real flow *waves* on a
  24 h plasma profile -- is measured here as -7.1% with identical cycle-mean flows (pure waveform effect), documented with its approximations.

## Structural identities

These are identities of the equations themselves, checked at machine precision; they are properties any correct implementation must show.

Authority: exact identities of the model's own equations + ICRP volume/shares

| Check | Independent authority | Measured | Status |
|---|---|---|---|
| total blood volume drift over 8 recorded beats | linear invariant: d(sum V)/dt = 0 by the node balance | 1.64e-11 mL | pass |
| graph blood volume == the individual's ICRP volume (5.3 L) | ICRP 89 referenceMale blood volume | 6.1e-11 mL | pass |
| LV and RV eject identically (one closed loop) | conservation at the pools: in-flow == out-flow | relerr 1.0e-04 | pass |
| valves carry exactly zero backflow | one-way orifice law: q = A eff sqrt(2 dp rho), dp>0 only | min over beat: 0.00e+00 mL/s | pass |
| declarative graph + schema validate clean | invariants: closed loop, ICRP shares sum, positive parameters | 0 violations | pass |
| organ-bed shares sum to 100 % | ICRP 89 flow split | 100.00 % | pass |
| instantaneous bed split == ICRP shares (all beds, any instant) | structure: every bed sees the same (p_sa - p_sv) gradient | relerr 2.2e-16 | pass |

## Baseline hemodynamics vs the published normal table

Parameters are each anchored to a published source; every *output* here is a prediction.

Authority: Broomé et al. 2013, Biomed Eng Online 12:69, Table 1 (published normal column; NO output quantity was fitted)

| Check | Independent authority | Measured | Status |
|---|---|---|---|
| cardiac output (L/min) | Broomé 2013 Table 1 normal: 6.3 +- 2.4 | 6.355 (+0.02 SD) | pass |
| stroke volume (mL) | Broomé 2013 Table 1 normal: 89.0 +- 30.0 | 90.791 (+0.06 SD) | pass |
| ejection fraction (-) | Broomé 2013 Table 1 normal: 0.59 +- 0.04 | 0.634 (+1.10 SD) | note |
| LVEDV (mL) | Broomé 2013 Table 1 normal: 123.0 +- 28.0 | 143.184 (+0.72 SD) | pass |
| RVEDV (mL) | Broomé 2013 Table 1 normal: 111.0 +- 22.0 | 138.528 (+1.25 SD) | note |
| mean RAP (mmHg) | Broomé 2013 Table 1 normal: 4.8 +- 2.4 | 3.463 (-0.56 SD) | pass |
| mean LAP (mmHg) | Broomé 2013 Table 1 normal: 8.3 +- 2.5 | 5.907 (-0.96 SD) | pass |
| aortic systolic (mmHg) | Broomé 2013 Table 1 normal: 112.0 +- 12.0 | 107.347 (-0.39 SD) | pass |
| aortic diastolic (mmHg) | Broomé 2013 Table 1 normal: 74.0 +- 12.0 | 69.637 (-0.36 SD) | pass |
| pulm. arterial systolic (mmHg) | Broomé 2013 Table 1 normal: 22.0 +- 6.0 | 21.570 (-0.07 SD) | pass |
| pulm. arterial diastolic (mmHg) | Broomé 2013 Table 1 normal: 9.0 +- 5.0 | 12.406 (+0.68 SD) | pass |
| aortic pulse pressure (mmHg) | published normal PP 30-40 mmHg (upper arm) | 37.71 | pass |
| mean arterial pressure (mmHg) | published normal MAP 75-105 mmHg | 90.46 | pass |
| LV dP/dt max (mmHg/s) | published normal 1200-2200 mmHg/s (Guyton; Braunwald) | 1471 | pass |
| aortic valve open time (s) | published systole ~0.3 s at HR 70 (Wiggers) | 0.296 | pass |

## Physiological sweeps

Each sweep must reproduce the published direction/shape; the numbers are the model's predictions, reported as measured.  The sweeps are open-loop -- no baroreflex or venoconstriction -- so they verify direction and shape, not the intact-animal magnitude of hemorrhagic collapse.

Authority: direction/shape of published physiological responses (preload/Starling, inotropy, hemorrhage, tachycardia)

| Check | Independent authority | Measured | Status |
|---|---|---|---|
| preload +10 % -> CO up (Frank-Starling) | published filling-response direction (Guyton; Starling) | CO 8.79 vs 6.36 L/min (EDV 193) | pass |
| preload -10 % -> CO down, EDV down | same authority | CO 5.10 L/min (EDV 118) | pass |
| hemorrhage -20 % -> CO collapses, MAP < 40 (shock) | published hemorrhage physiology | CO 1.20 L/min, MAP 17.0 mmHg | pass |
| Emax x0.5 -> EF falls, EDV and LAP rise (systolic failure direction) | published direction of every systolic-HF intervention | EF 49%, EDV 159 mL, LAP 6.4 mmHg | pass |
| Emax x1.3 -> EF rises, CO rises | published contractility response (dose-dependent acute inotropy) | EF 68%, CO 6.60 L/min | pass |
| HR 100 -> SV falls, CO rises modestly (no contractility change) | published resting response to tachycardia | SV 69.9 vs 90.8 mL, CO 6.99 L/min | pass |

## The drug layer vs the validated PBPK engine

The uptake layer is a mirror of the validated engine at constant flow; every deviation from it must be exactly attributable (listed) and no structural quantity may drift.

Authority: mirror identities vs the validated PBPK engine + exact algebraic conservation

| Check | Independent authority | Measured | Status |
|---|---|---|---|
| at ICRP flows the transport matrix == validated PBPK matrix | mirror of the validated engine (same state, volumes, CL) | max abs(M1-M2) = 0.0e+00 | pass |
| pulsatile solver on a FLAT flow template == mean solver | two independent code paths over identical constant flows | 1.3e-10 relative | pass |
| mean-stream solver == PBPK engine (24 h, IV+oral) | the validated engine itself | 1.2e-13 relative | pass |
| 24 h pulse-coupled run conserves total drug (IV 5 mg + oral 7.5 mg) | zero column sums of every transport matrix | 1.62e-10 mg | pass |
| pulsatile infusion integral exact (10 mg/h x 24 h = 240 mg) | affine plateau of the augmented exponential | 5.5e-07 of delivered | pass |
| steady state balances exactly through the elimination sinks | M x = -r x from below: r cancels (linear system) | 2.13e-14 mg/h | pass |
| matrix SS == engine's 96 h infusion plateau | long-run of the validated engine | SS 0.5847 vs engine 0.5847 ("1 --" 2.5e-07) | pass |
| plasma C_ss vs scalar R/CL_total | the engine's CL aggregate convention (7.8e-5 low; documented) | 0.5847 vs 0.5846 (7.8e-05 low) | pass |
| piecewise-constant transport discretization converged at dt = 4 ms | numerical self-consistency: dt/2 and dt/4 give the same orbit | dt 2 ms: 4.4e-06; dt 1 ms: 5.7e-06 relative | pass |

## Hepatic extraction

The ICRP-flow bottom-up CLint lands slightly below the published band; the one-factor middle-out calibration used by validate_body.py is shown too.

Authority: published hepatic extraction of midazolam (Thummel et al.)

| Check | Independent authority | Measured | Status |
|---|---|---|---|
| hepatic extraction ratio E_h (bottom-up) | published band 0.30-0.50 (Thummel et al. 0.44) | 0.284 | note |
| E_h with the one-factor middle-out CLint (fitted on IV data by validate_body) | published band 0.30-0.50 (Thummel 0.44) | 0.309 | pass |

## The flow-waveform effect on uptake

No external authority exists; this is the layer's own prediction, quantified with matched means and numerical convergence.

Authority: NO external authority: the quantified sensitivity of organ uptake to real flow waves (documented approximation: PBPK pool volumes held constant, only the flows modulate), numerically converged (dt 2/1 ms -> 0.0 %)

| Check | Independent authority | Measured | Status |
|---|---|---|---|
| AUC 0-24 h, pulsatile vs mean (SAME heart-mean flows: pure waveform effect) | the difference question this layer exists to answer | -7.1% | pass |
| AUC 0-24 h, mean-heart flows vs ICRP reference flows (CO/pattern scale effect) | heart CO 6.36 vs ICRP 6.50 L/min | +0.28% | pass |
| within-beat plasma swing, settled cycle 3 of the IV bolus | the beat-scale variation the PC-mean engine cannot resolve | +3% | pass |

Plasma profile behind the AUC number (IV 5 mg + oral 7.5 mg):

| t (h) | pulse-coupled (mg/L) | mean-flow (mg/L) | pulse vs mean |
|---|---|---|---|
| 0.5 | 0.09856 | 0.10648 | -7.4% |
| 2.0 | 0.05608 | 0.06075 | -7.7% |
| 6.0 | 0.01600 | 0.01711 | -6.5% |
| 12.0 | 0.00546 | 0.00603 | -9.4% |
| 24.0 | 0.00069 | 0.00083 | -17.1% |

## Observed clinical overlay

Same clinical data the body engine is scored against (validate_body.py); the beat-grain curves must stay inside its accuracy band.

Authority: clinical midazolam plasma observations (OSP digitized set); the flow-grain change must stay inside the accuracy band of the mean engine (which validate_body.py scores across the full dataset panel)

| Check | Independent authority | Measured | Status |
|---|---|---|---|
| Olkkola 1993 - iv Control (Perpetrator Placebo) (aggregated, dose 0.05 mg/kg, n = 6) | clinical plasma data (OSP digitized; NOT used to fit the flow layer); study nearest the panel-median AUC ratio of validate_body.py | AUC pred/obs: mean-stream 1.25, pulse-coupled 1.39 (beat-waveform shift +10.8%); AAFE 1.45 | pass |
| Olkkola 1993 - po Control (Perpetrator Placebo) (aggregated, dose 15 mg, n = 12) | clinical plasma data (OSP digitized; NOT used to fit the flow layer); study nearest the panel-median AUC ratio of validate_body.py | AUC pred/obs: mean-stream 1.15, pulse-coupled 1.07 (beat-waveform shift -6.6%); AAFE 1.79 | pass |

## Documented approximations

- Pulsatile transport rides the PBPK engine's constant blood-pool volumes; only the flows are
  modulated.  Pool volume waves live in the hydraulic layer (next coupling step); this isolates
  the flow-waveform effect the layer exists to quantify.
- Boluses land on the transport grid within one segment (~4 ms) of the nominal time.
- E_h bottom-up comes from in-vitro CLint scaled to microsomal protein (Simcyp convention,
  Rowland Yeo 2004) and is mildly lower than the clinical band; the middle-out convention is
  scored in the extraction table.

## Sources
- Suga & Sagawa 1974 (time-varying elastance); Stergiopulos et al. 1992/1996 (double-Hill shape).
- Broomé et al. 2013, Biomed Eng Online 12:69, Table 1 (published normal values).
- ICRP Publication 89 (organ flow shares, blood volume); Guyton & Hall (sweep directions).
- Thummel et al. (hepatic extraction of midazolam).
- Clinical plasma data digitized by the Open Systems Pharmacology project.