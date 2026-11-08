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
| 2026-11-08 | E | [Conserved-flow graphs: topology → venous pressure](2026-11-08-E-vein-flow-graph/report.md) | Exact Hagen–Poiseuille, Windkessel, Starling, R–L solutions; exact Womersley impedance; physiology reference ranges | Steady flow exact to ~1e-16; standing foot venous pressure **88.7 mmHg** vs **85–90** (foot inside range, ankle 0.6 mmHg below); lumped R–L within 1% up to α=0.48, with measured breakdown at high α | OpenCode agent (DeepSeek V4.1 Flash Free) |
| 2026-11-08 | E | [Steady flow with valves: unique solution, finite solver, honest `converged`](2026-11-08-E-steady-complementarity/report.md) | Exact enumeration of all valve / collapse regimes; closed-form roots | Previous solver: 39/2000 random networks wrong but `converged=True` (worst 23%); now 0/6000, error ≤ 8e-15. Flow uniqueness proved for valves; Starling uniqueness 6000/6000 (conjecture). Expansion kinetic term makes the model ill-posed above 0.53 L/min (documented, not fixed) | MikeMike88 with Claude Code (Claude Opus 5.5) |
