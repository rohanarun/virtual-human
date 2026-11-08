# Roadmap

Each milestone has a "done when" test. Nothing counts as done on demos alone.
See [`accuracy_ladder.md`](accuracy_ladder.md) for the five-rung accuracy ladder.

## Accuracy ceiling (done): the smallest parts of the body, exactly

- [x] **CCSD(T) / complete-basis-set** electronic structure (`vh.quantum`): water
  geometry and dipole, the water-dimer hydrogen bond, the peptide bond and
  glycine, reproduced from first principles to a fraction of a percent vs
  experiment/benchmark. Counterpoise (BSSE) correction and CBS extrapolation.
  *Done when* water dipole, O–H length and the dimer binding energy fall within
  ~1% of experiment with no fitting (achieved).
- [x] **QM/MM** electrostatic embedding: a quantum region inside a classical
  field. *Done when* it recovers the exact classical electrostatics limit at long
  range and its forces match finite difference (achieved to 1e-6).

### Scaling the ceiling toward a cell (the long road)

A cell is ~10¹⁴ atoms, so pure CCSD(T) will never reach it. The path up:
1. **Linear-scaling / fragment quantum methods** (DLPNO-CCSD(T), fragment
   molecular orbital) to take CCSD(T)-quality energies from ~20 atoms to
   hundreds, then thousands. *Done when* a solvated amino acid matches canonical
   CCSD(T) within 0.3 kcal/mol at a fraction of the cost.
2. **ML interatomic potentials trained on CCSD(T)/DFT** (ANI, MACE) for
   near-quantum accuracy at force-field cost, so quantum accuracy can drive
   dynamics of whole proteins. *Done when* conformational and interaction
   energies match the reference QM to ≤0.5 kcal/mol.
3. **QM/MM at scale**: the quantum region is the drug + binding site, the MM
   region the rest of the protein and explicit water (rung 3 below). This is the
   realistic route to molecular-accuracy drug binding inside a real target, and
   the building block from which a cell is assembled one macromolecule at a time.

## v0.1 (now)

- [x] From-scratch atomistic engine (JAX): ff14SB + GB-OBC2, autodiff forces, BAOAB/Verlet, L-BFGS.
  *Done when* every energy term and force matches an independent engine on real proteins (achieved: ~1e-14).
- [x] From-scratch whole-body PBPK: 13 organs, Rodgers & Rowland partitioning, IVIVE liver, renal filtration,
  CAT absorption with gut-wall first pass, virtual populations, exact matrix-exponential solver.
  *Done when* bottom-up predictions fall within 2-fold of most clinical datasets for a probe drug (achieved for midazolam: 81–93%).
- [x] Bridge: unbound target-tissue concentration → receptor occupancy (equilibrium and kinetic), Kd ↔ ΔG.
- [x] Browser engines (JS) identical to Python to ≤1e-12, live demo.

## Next: make the molecular layer test medicines

1. **Drug molecules in the engine.** Native SMIRNOFF (OpenFF Sage) atom typing with RDKit SMARTS matching, and
   quantum-derived partial charges (RESP from HF/6-31G* with PySCF, or AM1-BCC-equivalent).
   *Done when* gas-phase and implicit-solvent energies of a drug set match a reference implementation, and
   hydration free energies on FreeSolv reach RMSE ≤ 1.5 kcal/mol.
2. **Explicit water and scale.** PME electrostatics, SETTLE/SHAKE constraints (2–4 fs steps), neighbor lists,
   GPU execution (JAX on CUDA/Metal), mixed precision.
   *Done when* a 25,000-atom solvated protein–ligand complex runs ≥100 ns/day on one GPU, conserves energy,
   and matches the referee engine term by term.
3. **Binding free energies.** Alchemical FEP/TI with MBAR and replica exchange, protein–ligand restraints.
   *Done when* relative binding free energies on a public benchmark (e.g. the OpenFE / Schrödinger
   protein–ligand benchmark sets) reach ≤1.2 kcal/mol RMSE vs experiment. Kd then flows into
   `vh.bridge` automatically: this is where "molecular accuracy" reaches the body.
4. **Quantum accuracy where it matters.** ML interatomic potentials (ANI-2x, MACE-OFF, UMA-class models) for the
   ligand inside a classical protein (ML/MM).
   *Done when* ligand conformational energies match DFT to ≤0.5 kcal/mol on a torsion benchmark.
5. **Native parameter assignment for proteins.** Drop OpenMM from the prep step entirely (it stays as a test referee).

## Next: make the body layer broader

6. **Validation suite of ~100 drugs.** Import every compound and clinical dataset in the Open Systems
   Pharmacology model library; publish the scorecard on every commit.
7. **Mechanistic interactions.** Competitive and time-dependent enzyme inhibition (Ki, kinact), induction,
   transporters (OATP, P-gp), nonlinear Michaelis–Menten kinetics.
   *Done when* clinical DDI ratios for the OSP DDI network are predicted within 2-fold for ≥80% of pairs.
8. **Special populations.** Pediatrics (organ growth, enzyme ontogeny), pregnancy, hepatic and renal impairment.

## Next: fill in the middle scales

9. **Heart safety.** A human ventricular myocyte model (CiPA's O'Hara–Rudy) in the same engine; channel block
   from the molecular layer → action-potential prolongation → torsade risk category.
   *Done when* the 28 CiPA reference drugs are classified into their published risk categories.
10. **Cells and pathways.** SBML import (BioModels) so target occupancy drives signaling and cellular response.
    *Started 2026-11-05:* a from-scratch SBML core reader with exact conservation and positive-mass
    invariants, validated against libRoadRunner ([report](../auto-research/2026-11-05-E-sbml-import/report.md)).
    Still open: events, rate rules, delays, units checking, and wiring `vh.bridge` occupancy into a pathway.
11. **Uncertainty end to end.** Propagate parameter uncertainty across scales; Bayesian calibration; because every
    layer is differentiable (JAX), gradients of a clinical endpoint with respect to molecular parameters.

## Explicitly out of scope

Simulating a whole body atom by atom. See `feasibility.md`: it needs ~10²⁷ times today's largest computers.
