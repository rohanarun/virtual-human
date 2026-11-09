# auto-research

Published research contributions to the Virtual Human project, from AI agents and
people. Each entry is a self-contained, independently reviewable report with its
validation result.

**Before contributing, read [`../auto-research.md`](../auto-research.md)** — it
explains the one rule (validate against an independent authority), the open
research tracks, and the submission format.

## Layout

```
auto-research/
  README.md                         this index
  TEMPLATE/report.md                copy this for a new contribution
  <YYYY-MM-DD>-<track>-<slug>/      one folder per contribution
    report.md                       what, how, validation result + error, limits, refs
    results.json, *.png, ...        data, figures, scripts (or links into scripts//vh/)
```

## Index

| Date | Track | Contribution | Validated against | Result | Author |
|---|---|---|---|---|---|
| 2026-11-05 | E (partial) | [Explicit model contracts](2026-11-05-E-model-contracts/report.md) | Exact scalar mass balance; interface rejection cases | 0 L/h max error on 4 synthetic cases; 60 focused tests pass | OpenAI Codex / @ahrazzle |
| 2026-11-05 | E | [Clearance reference validation](2026-11-05-E-clearance-reference/report.md) | Exact organ mass balance and linear ODE integral | Mixed references caught; reported sum 17.104405 L/h, dose/AUC 17.103070 L/h | OpenAI Codex / @ahrazzle |
| 2026-11-05 | E (partial) | [Reaction and intervention experiments](2026-11-05-E-reaction-experiments/report.md) | Exact kinetics/conservation/commutativity; independent DOP853 | Maximum trajectory error 1.479e-11 mol/L; no biological validation | OpenAI Codex / @ahrazzle |
| 2026-11-05 | E | [SBML reaction networks as verified graphs](2026-11-05-E-sbml-import/report.md) | Exact rational algebra; libRoadRunner as independent SBML engine | 3 CC0 BioModels: exact conservation laws, trajectories ≤7e-10, steady states ≤2e-12; positive-mass invariant flags 8 of 49 surveyed models with exact certificates | Claude (Anthropic) |
| 2026-11-08 | E | [Conserved-flow graphs: topology → venous pressure](2026-11-08-E-vein-flow-graph/report.md) | Exact Hagen–Poiseuille, Windkessel, Starling, R–L solutions; exact Womersley impedance; physiology reference ranges | Steady flow exact to ~1e-16; standing foot venous pressure **88.7 mmHg** vs **85–90** (foot inside range, ankle 0.6 mmHg below); lumped R–L within 1% up to α=0.48, with measured breakdown at high α | OpenCode agent (DeepSeek V4.1 Flash Free) |
| 2026-11-08 | E | [Steady flow with valves: unique solution, finite solver, honest `converged`](2026-11-08-E-steady-complementarity/report.md) | Exact enumeration of all valve / collapse regimes; closed-form roots | Previous solver: 39/2000 random networks wrong but `converged=True` (worst 23%); now 0/6000, error ≤ 8e-15. Flow uniqueness proved for valves; Starling uniqueness 6000/6000 (conjecture). Expansion kinetic term makes the model ill-posed above 0.53 L/min (documented, not fixed) | MikeMike88 with Claude Code (Claude Opus 5.5) |

- [2026-11-06: Explicit evidence bindings](2026-11-06-E-evidence-bindings/report.md): nominal scalar execution with preserved context and uncertainty; exact synthetic mass balance and missing kinetic-input report.
