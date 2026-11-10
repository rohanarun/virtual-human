# auto-research — a guide for AI agents (and people) contributing research

This project is built to be extended by autonomous agents as well as people. This
document tells a contributor how to help without breaking the one rule that makes
the project trustworthy, what the open research tracks are, and how to publish
work so it accumulates instead of colliding.

If you are an AI agent picking this up cold: read this file, then
`docs/accuracy_ladder.md`, `docs/architecture.md`, and `docs/roadmap.md`. Then
pick a track below, do the work on a branch, and publish a report under
`auto-research/` as described at the end.

## The one rule: validate against an independent authority

Every quantitative claim in this repo is checked against something that could
prove it wrong and was not used to produce it:

- an **independent engine** (our molecular engine vs OpenMM, to ~1e-14);
- an **exact analytic result** (QM/MM vs the classical charge–dipole limit;
  constraints to 1e-16);
- a **high-level reference or experiment** (CCSD(T)/CBS vs measured geometries,
  dipoles, and the water-dimer benchmark; PBPK vs clinical data never used to fit).

A contribution is not done when the code runs. It is done when it is checked
against an authority of this kind, the check is in the test suite, and the result
is written down with its error bar. **Never tune a model on the same data you then
report agreement with.** If you fit a parameter, say so, fit it on one dataset,
and report a blind prediction on another (as `scripts/validate_body.py` does).

Other standing expectations:

- **Provenance on every input.** Each physical constant, force-field parameter,
  physiological value and clinical dataset carries where it came from. Keep it
  that way (see `data/tissue_composition_rr.json`, `vh/body/compound.py`).
- **From scratch, with a referee.** The value of this project is engines we
  control and can differentiate, proven correct against an independent
  implementation — not wrappers. Add the physics; keep the referee for the test.
- **Check the experiment's inputs.** A prior circulation result converged while
  its transport sampler stretched half a beat over the full cycle. For a
  waveform comparison, check the actual solver inputs against an asymmetric
  reference over the full period, derive the steady comparator from those
  exact sampled means, and integrate exposure independently of output cadence.
  Test dosing between grid points against the prescribed dose integral. Report
  standard deviations only where the cited source supplies them.
- **Tests and CI.** New capability ships with tests. Keep `pytest -q` and
  `node web/test_engines.mjs` green.
- **Honesty about limits.** State what a method cannot do as plainly as what it
  can. The feasibility ceiling (`docs/feasibility.md`) is real; do not paper over
  it.

## The accuracy ladder (where the engines sit)

| Rung | Physics | Status |
|---|---|---|
| 5 | CCSD(T)/CBS quantum chemistry | built (`vh.quantum`) |
| 4 | QM/MM electrostatic embedding | built (`vh.quantum.qmmm`) |
| 3 | Explicit-solvent MD + Ewald/PME | built (`vh.molecular.periodic`, `.explicit_md`) |
| 2 | Rigorous free energy (TI/MBAR) | **open — highest-value next step** |
| 1 | Implicit-solvent MD | built (`vh.molecular`) |
| — | Whole-body PBPK | built (`vh.body`) |
| — | Graph formalization | built (`vh.graph`) |

## Research tracks

Each track lists a concrete first deliverable and its "done when" test. Pick one,
or propose another in an `auto-research/` report.

### Track A — Free energy (rung 2). *Highest value.*
Turn the explicit-solvent engine into predicted binding/solvation free energies.
The engine already exposes alchemical soft-core (`lambda_vdw`, `lambda_elec`) and
exact `dU/dlambda` (`PeriodicModel.dudl`). Build a thermodynamic-integration +
MBAR driver on top.
- **First deliverable**: hydration free energy of a small neutral molecule.
- **Done when**: it matches the FreeSolv experimental value within ~1 kcal/mol,
  and TI vs MBAR agree; also reproduce an analytic case (ion charging ≈ Born).

### Track B — Scale the quantum ceiling toward larger systems.
Fragment / linear-scaling CCSD(T) (DLPNO-style) or ML interatomic potentials
(ANI/MACE) trained on QM, so quantum-grade accuracy reaches a whole amino acid in
water, then a binding site.
- **Done when**: a solvated amino acid matches canonical CCSD(T) within
  0.3 kcal/mol, or an ML potential matches reference QM torsions to ≤0.5 kcal/mol.

### Track C — Broaden the body model.
More compounds from the Open Systems Pharmacology library; mechanistic drug–drug
interactions (competitive + time-dependent inhibition, induction); transporters.
- **Done when**: clinical DDI ratios predicted within 2-fold for ≥80% of a named
  OSP DDI set, with the compounds parameterized from in-vitro data, not fit to the
  clinical DDI.

### Track D — Organ electrophysiology (heart safety).
A human ventricular myocyte model (e.g. O'Hara–Rudy / CiPA) in the engine; drive
ion-channel block from the molecular layer to action-potential changes.
- **Done when**: the CiPA reference drugs are sorted into their published
  torsade-risk categories.

### Track E — Grow the graph formalization.
More scales and relationships in `vh.graph` (reaction networks via SBML import;
cell-level compartments; signaling). New structural invariants that catch modeling
errors as graph-algorithm checks.
- **Done when**: a BioModels SBML model imports to a graph whose steady-state /
  conservation invariants are verified, and at least one new invariant catches a
  real inconsistency.

### Track F — Performance and scale.
GPU execution (JAX on CUDA/Metal) for the molecular and explicit-solvent engines;
neighbor-list and PME tuning; mixed precision. No accuracy regression.
- **Done when**: a ≥25,000-atom solvated system runs ≥100 ns/day on one GPU and
  still matches the referee term-by-term.

### Track G — Explicit-solvent production runs.
Solvate one of the shipped proteins, equilibrate (NVT→NPT), and validate bulk
observables (TIP3P density ≈ 1.0 g/cm³ at 298 K/1 bar; radial distribution
functions) against literature.
- **Done when**: density and the O–O RDF first-peak match published TIP3P values.

## How to publish your work

Publish every contribution as a self-contained report under `auto-research/`, so
results accumulate and are reviewable independently of the code diff.

1. Work on a branch: `auto/<track-letter>-<short-slug>` (e.g. `auto/A-hydration-fe`).
2. Create a folder `auto-research/<YYYY-MM-DD>-<track-letter>-<slug>/` containing:
   - `report.md` — follow `auto-research/TEMPLATE/report.md`: what you did, the
     method, the **validation result with its error vs the authority**, the limits,
     and references.
   - any figures, `results.json`, and the scripts you added (or links to them in
     `scripts/` / `vh/`).
3. Add your tests to `tests/`; keep CI green.
4. Add one line to the index in `auto-research/README.md`.
5. Add yourself to the Contributors section of the top-level `README.md`.
6. Open a pull request. In the PR body, state the authority you validated against
   and the measured error. A contribution with no independent validation will be
   asked for one before merge.

Keep reports honest and specific. A negative result that is correctly validated
(“method X does not reach chemical accuracy here, by this much, for this reason”)
is a valuable contribution and belongs in `auto-research/` too.
