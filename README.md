# Virtual Human

> *Solving problems in the human body by formalizing the real world into code for agents.*

A multi-scale simulation of the human body for testing medicines and proteins,
built from scratch and validated at every level. It spans from **coupled-cluster
quantum mechanics** on the molecules the body is built from, up through an
**atom-level physics engine** for proteins and drugs, to a **whole-body engine**
that predicts what a drug does in a person — with a **real-time browser demo**.

<p align="center">
  <img src="docs/img/atom_viewer.png" width="460" alt="Browser viewer: the Trp-cage protein simulated atom by atom in real time">
</p>

> The Trp-cage protein, every atom moving under real force-field physics, **live in the browser** — drag to rotate, heat it, watch native contacts and energy update each step. This is the from-scratch engine (ported to JavaScript and checked against the Python engine to 1e-12) running in real time; the full demo also drives a virtual patient. See [`web/`](web/).

> Status: under active development. Every accuracy claim below is backed by a test
> or a validation report in `docs/`, checked against an independent engine, an
> exact analytic result, or experiment.

## Two directions at once

- **Up for coverage**: molecule → cell → organ → whole body, so a result can
  reach a clinical endpoint. The whole-body layer runs faster than real time.
- **Down for accuracy**: start at the *highest accuracy physics allows* on the
  *smallest real parts of the body* (water, the peptide bond, amino acids),
  prove them correct against known truth, then carry that accuracy upward. This
  is the [accuracy ladder](docs/accuracy_ladder.md), and its top rung —
  CCSD(T) at the complete-basis limit — is the standard that "molecular accuracy"
  is itself measured against. We reproduce experiment there to a fraction of a
  percent with no fitting.

## Why multi-scale (and not "every atom of a body")

A body has ~7×10²⁷ atoms. Atomic simulation needs ~5×10¹⁴ time steps per
simulated second, so one second of a whole body is ≳10⁴⁵ floating-point
operations. On today's fastest supercomputers (~2×10¹⁸ FLOP/s), that's billions
of times the age of the universe, and a single human cell for one second would
take ~800,000 years. See [`docs/feasibility.md`](docs/feasibility.md).

So molecular accuracy goes where molecules decide the outcome (a drug binding its
target, a protein's shape and motion). The numbers it produces are passed up to
the body model, which works in organs, blood flows and concentrations and runs
faster than real time.

```
 vh.molecular   atoms        proteins + drugs in water, femtosecond steps
      │  binding strength, unbound fraction, ...
      ▼
 vh.bridge      target engagement (receptor occupancy over time)
      ▲
      │  unbound drug concentration at the target organ
 vh.body        organs       whole-body drug distribution and elimination (hours/days)
```

## What exists today

| Layer | What it does | How it's validated |
|---|---|---|
| `vh.quantum` | **The accuracy ceiling.** Ab initio electronic structure (HF, DFT, MP2, CCSD, **CCSD(T)**) via PySCF, with counterpoise (BSSE) correction, complete-basis-set extrapolation, geometry optimization, and QM/MM electrostatic embedding | Water geometry & dipole, the water-dimer hydrogen bond, the peptide bond and glycine reproduced from first principles to a fraction of a percent vs experiment/benchmark; QM/MM vs the exact classical limit and finite-difference forces ([`docs/accuracy_ladder.md`](docs/accuracy_ladder.md)) |
| `vh.molecular` | From-scratch JAX engine: AMBER ff14SB force field, GB-OBC2 implicit water, autodiff forces, Langevin (BAOAB) and Verlet dynamics, L-BFGS minimization | Matches OpenMM (used only as a referee) term by term to ~1e-14 relative on real proteins; trajectories identical step by step; energy conserved ([`docs/validation_molecular.md`](docs/validation_molecular.md)) |
| `vh.body` | From-scratch whole-body PBPK: 13 organs + blood, Rodgers & Rowland tissue partitioning, liver (well-stirred, IVIVE), kidney (filtration), oral absorption (stomach + 7-segment intestine) with gut-wall first-pass, virtual populations; solved exactly with matrix exponentials | Bottom-up midazolam predictions vs 100 clinical datasets (IV and oral) digitized by the Open Systems Pharmacology project |
| `vh.bridge` | Unbound drug at the target organ → receptor occupancy over time; Kd ↔ binding free energy | Analytic round-trip; kinetic occupancy |
| `vh.graph` | **Formalizes the body as a typed graph** across scales (organs↔blood flow, atoms↔bonds); graph algorithms for reachability, pathways, cycles, flow balance, bottlenecks; **conserved-flow graphs** with R/C/L edge laws, gravity, Bernoulli, Starling collapse and valves; renders and verifies subgraphs | Blood conservation (Kirchhoff) to 1e-16 on the real PBPK model; a pure graph BFS reproduces the force field's exclusion list exactly; steady flow exact against Hagen–Poiseuille/Windkessel/Starling, and the standing foot venous pressure predicted at **88.7 mmHg** vs the physiological **85–90 mmHg** range ([`docs/graph_formalization.md`](docs/graph_formalization.md), [`docs/bernoulli_veins.md`](docs/bernoulli_veins.md)) |
| `web/` | Real-time demo: atom-level protein dynamics and a virtual patient, live in the browser | JS engine checked against the Python engine to ≤1e-12 |

The explicit-solvent engine (`vh.molecular.periodic`, `.constraints`, `.explicit_md`) adds rung 3 of the accuracy ladder: every water molecule, exact long-range electrostatics (Ewald / smooth PME), rigid-water SHAKE/RATTLE and a Monte-Carlo barostat — validated term-by-term against OpenMM Ewald (energy to 1e-13, PME to 1e-11, constraints to 1e-16).

See [`docs/accuracy_ladder.md`](docs/accuracy_ladder.md) for the full five-rung ladder and what each rung is validated against.

## Formalizing the body as a graph

**Can we solve problems with code by formalizing the body as a graph?** Yes — for
the structural/relational class of problems — and `vh.graph` proves it. The honest
boundary (in [`docs/graph_formalization.md`](docs/graph_formalization.md)): a graph
solves *where / what-connects / how-much-flows / is-it-consistent*; it does **not**
solve *how-strong / how-fast / how-it-folds* — that stays physics. The real win is a
**typed graph that is also executable**: every edge carries the validated model
behind it, so the graph wires the scales together and lets any part be verified
alone.

The body is already a graph at every scale — organs joined by blood flow, atoms
joined by bonds. Formalized from the real, validated engines:

<p align="center">
  <img src="docs/img/pbpk_graph.png" width="760" alt="The whole-body model as a flow graph of organs">
</p>

Graph algorithms (all written from scratch) then give exact answers:

<p align="center">
  <img src="docs/img/graph_results.png" width="820" alt="Blood delivered to each organ and the structural hubs, from graph algorithms">
</p>

| Question | Algorithm | Result on the real model |
|---|---|---|
| Is it physically consistent? | flow balance (Kirchhoff) | blood conserved at every node to **1.5×10⁻¹⁶** |
| Where can an IV drug reach? | reachability (BFS/DFS) | all 14 compartments, incl. brain |
| What does an oral drug hit first? | path enumeration | `gut → liver` before systemic — **first-pass metabolism from topology** |
| Does anything recirculate? | strongly-connected components | one loop spanning 15 compartments (systemic circulation) |
| Where are the hubs? | betweenness centrality | arterial, venous, lung |
| **Verify a part in isolation** | subgraph extraction | elimination subgraph clearance = full model **exactly (28.488 L/h)** |
| **The graph verifying the engine** | BFS on the bond graph | reproduces the force field's **1687 exclusion pairs exactly** (symmetric difference 0) |

That last row is the sharpest: a pure topological fact (atoms within bond-distance ≤ 3)
and the physics engine's own nonbonded bookkeeping check each other. Render any
subgraph to a diagram with `vh.graph.render`; reproduce all of the above with
`python scripts/validate_graph.py`.

### From structure to a quantitative flow prediction

Topology alone cannot give a pressure. Add a validated constitutive law to every
edge and the same graph becomes a lumped 1-D Navier–Stokes solver. `vh.graph.flow`
does this with resistance, compliance, inertance, gravity, the Bernoulli kinetic
term, Starling collapse and one-way valves:

$$
C_i\dot p_i=\sum_e B_{ie}q_e+s_i,\qquad
L_e\dot q_e=p_{\text{src}}-p_{\text{dst}}-h_e-R_eq_e-\Delta p_{\text{kinetic}}-\Delta p_{\text{minor}}.
$$

The edge equation is exactly extended Bernoulli; classical Bernoulli is the
special case with no losses and no unsteady term. Because the laws are explicit,
each one is checked against an exact solution, not against itself: Poiseuille flow
to 1e-16, the Windkessel transient to 0 Pa, the Starling waterfall to 2e-19 m³/s,
and the lumped pulsatile edge against the exact Womersley impedance. The first
real prediction: a lower-limb venous network with anatomical geometry predicts a
**standing foot venous pressure of 88.7 mmHg**, inside the physiological
**85–90 mmHg** range, with no parameter fitted to that range (supine: 8.4 mmHg vs
5–10).

<p align="center">
  <img src="docs/img/vein_graph.png" width="620" alt="Lower-limb venous flow graph with predicted pressures and flows">
</p>

The derivation, the failure modes (viscosity, gravity, unsteady flow, collapse,
valves), and the measured limits are in
[`docs/bernoulli_veins.md`](docs/bernoulli_veins.md); the method ladder from
lumped graphs to 1-D PDEs, CFD, surrogates and digital twins is in
[`docs/prediction_methods.md`](docs/prediction_methods.md); the measured report is
[`docs/validation_flow.md`](docs/validation_flow.md). A plain-language walkthrough
with charts is in [`docs/flow_explained.md`](docs/flow_explained.md). Reproduce
with `python scripts/validate_flow.py`.

## Quick start

```bash
pip install -e ".[dev]"              # numpy, scipy, jax (+ openmm/pdbfixer for structure prep & tests)
pip install -e ".[quantum]"          # pyscf + geometric, for the vh.quantum accuracy ceiling
pytest -q                            # full test suite

# the accuracy ceiling: water, the hydrogen bond, the peptide bond, glycine, from first principles
python scripts/validate_quantum.py
python scripts/plot_quantum.py       # -> docs/img/quantum_accuracy.png

# simulate a protein atom by atom (writes a multi-model PDB you can open in PyMOL/ChimeraX)
python -m vh.molecular.run data/structures/1l2y_trpcage.pdb --ps 50 --temp 300

# re-run the molecular / body validation reports
python scripts/validate_molecular.py
python scripts/validate_body.py

# re-run the conserved-flow graph report: exact Poiseuille/Windkessel/Starling/Womersley + venous pressure
python scripts/validate_flow.py
```

## Where the inputs come from

- **Force field**: AMBER ff14SB + OBC2 parameters, assigned via OpenMM's `ForceField`
  used purely as a parameter compiler (`vh/molecular/prep.py`). All energies, forces
  and dynamics are computed by our engine.
- **Structures**: chignolin (PDB 1UAO) and Trp-cage (PDB 1L2Y), first NMR models.
- **Physiology**: ICRP Publication 89 reference adults.
- **Tissue composition**: Rodgers & Rowland tables via open-access transcriptions
  (see `data/tissue_composition_rr.json` for per-value provenance).
- **Clinical data**: Open Systems Pharmacology model library (each dataset cites its
  original publication).

## Layout

```
vh/quantum     ab_initio.py (HF/DFT/MP2/CCSD(T), CBS, BSSE), qmmm.py (QM/MM embedding)
vh/molecular   energy.py (force field), integrate.py, minimize.py, prep.py, io.py, run.py, reference.py
               periodic.py (Ewald/PME explicit solvent), constraints.py (SHAKE/RATTLE), explicit_md.py
vh/body        physiology.py, partition.py, compound.py, pbpk.py
vh/bridge      molecular → body coupling (Kd ↔ ΔG, receptor occupancy)
vh/graph       core.py (typed property graph), algorithms.py, flow.py (conserved-flow solver), physiology_graph.py, render.py
tests/         pytest suite
scripts/       validation report + figure generators
docs/          accuracy_ladder, graph_formalization, feasibility, validation reports, architecture, roadmap
web/           real-time browser engine + demo
data/          structures, tissue composition, clinical observations
```

## Not a medical device

This is research software for hypothesis generation. It is not validated for
clinical decision-making, and no simulation replaces clinical testing.

## Contributing

This project is built to be extended by AI agents as well as people. Read
[`auto-research.md`](auto-research.md): it explains the one rule (every
quantitative claim is validated against an independent authority — an independent
engine, an exact analytic result, or experiment not used to fit), the open
research tracks, and how to publish work under [`auto-research/`](auto-research/).

## Contributors

- **Rohan Arun** ([@rohanarun](https://github.com/rohanarun)) — creator and maintainer; direction and design.
- **Claude** (Anthropic) — implemented and validated the engines under the above direction, in pair with the maintainer.
- **OpenCode agent (DeepSeek V4.1 Flash Free)** — conserved-flow graph, vein model, Bernoulli derivation, and flow validation under the above direction.
- **MikeMike88 with Claude Code (Claude Opus 5.5)** — steady-state complementarity: uniqueness proof, finite pivoting solver, scaled convergence test, enumeration referee.

New contributors (human or agent): add yourself here in your pull request. See
[`auto-research.md`](auto-research.md) for how to help.

## License

MIT — see [LICENSE](LICENSE). Contributions welcome. If you use this in research,
a citation back to the repository is appreciated.
