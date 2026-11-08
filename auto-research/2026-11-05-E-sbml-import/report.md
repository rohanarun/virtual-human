# SBML reaction networks as verified graphs

- **Track**: E — grow the graph formalization (SBML import; conservation and consistency invariants)
- **Author**: Claude (Anthropic), under the direction of the Foundation-0 owner
- **Date**: 2026-11-05
- **Branch / PR**: `auto/E-sbml-import`
- **Code added**: `vh/graph/sbml.py`, `vh/graph/stoichiometry.py`, `vh/graph/render.py` (styles),
  `scripts/validate_sbml.py`, `tests/test_sbml.py`, `data/sbml/` (three CC0 BioModels with provenance)

## Question

Track E's milestone: *a BioModels SBML model imports to a graph whose
steady-state / conservation invariants are verified, and at least one new
invariant catches a real inconsistency.* Can a from-scratch reader turn
curated reaction networks into executable models and typed graphs whose
structural claims are proven exactly rather than asserted, and whose dynamics
agree with an independent SBML engine?

## Method

**Reader.** `read_sbml` parses SBML Level 2 (v1–5) and Level 3 (v1–2) core with
the standard-library XML parser. It refuses documents containing a DOCTYPE or
entity declarations. MathML is compiled to closures over a fixed operator table;
nothing is passed to `eval`. Supported constructs:
- compartments of constant size, species, and global and local parameters;
- function definitions, inlined at compile time;
- initial assignments, and assignment rules in dependency order (loops are
  rejected);
- reactions with exact rational stoichiometry, modifiers and kinetic laws.

Every other construct raises `UnsupportedSBML` and names it: events, rate and
algebraic rules, constraints, delays, fast reactions, stoichiometryMath,
conversion factors, non-constant compartments and required packages.

The reader implements the SBML semantics the kinetics depend on:
- a species symbol means concentration unless `hasOnlySubstanceUnits`;
- kinetic laws are in substance per time;
- boundary species are not changed by reactions.

It also rejects documents that violate them, such as a constant, non-boundary
species consumed by a reaction. Units are not checked.

**Dynamics.** `d amount / dt = N v(x)` is integrated with SciPy (LSODA) on the
concentration-scaled state. Steady states come from a hybrid Newton solve on the
conservation-reduced system. Stability uses the Jacobian restricted to the
stoichiometric class: central differences reduced through the exact link matrix.

**Exact stoichiometry** (`vh.graph.stoichiometry`). All coefficients are
`Fraction`s.
- **Conservation laws** are the RREF of the left nullspace of `N`, scaled to
  coprime integers. That basis is canonical for the space, so two
  implementations can be compared row by row.
- **Positive-mass (stoichiometric) consistency** follows Gevorgyan, Poolman &
  Fell (2008): every reacting species must admit a strictly positive mass that
  every internal reaction conserves. A floating-point LP classifies the
  species, and the classification is then proved with exact certificates:
  - for a conserved species, a mass vector `m ≥ 0` with `mᵀN = 0` and `mᵢ > 0`;
  - for an unconserved species, a reaction combination `v` with `N v ≥ 0` and
    `(N v)ᵢ > 0`, which shows by Farkas' lemma that no positive mass exists.

  Rounding is removed exactly. Rows the LP leaves numerically at zero are
  constrained to be exactly zero in their exact nullspace, and the LP is solved
  again. A result is labelled `certified` only if every certificate checks
  exactly.
- **Internal reactions.** Boundary and constant species keep their rows,
  because holding a species fixed does not change its mass. A reaction counts as
  internal only if each side has a species that is not held fixed. Reactions
  that touch only fixed pools on one side, such as `A → EmptySet`, are
  exchanges.

**Graph.** `SBMLModel.to_graph()` builds a typed graph:
- nodes for `compartment`, `species`, `reaction` and `moiety`;
- `reactant`/`product` edges carrying exact stoichiometry strings, plus
  `modifier`, `located_in` and `member` edges.

`sbml_schema()` recomputes `N` from the edges alone and checks four invariants:
stoichiometry, located, conservation (exact, complete, with matching totals)
and stoichiometric consistency.

**Inputs and provenance.** Three curated BioModels whose notes state CC0 are
vendored unmodified. `data/sbml/sources.json` records each one's URL, SHA-256,
size and PubMed IDs:
- BIOMD0000000009, Huang & Ferrell 1996
- BIOMD0000000010, Kholodenko 2000
- BIOMD0000000012, Elowitz & Leibler 2000

Survey models (BIOMD0000000001–060) are identified by SHA-256 in
`results.json` and are not vendored.

## Validation — the part that matters

The authorities are of two kinds.
- **Exact rational algebra** for the structural claims, together with
  closed-form solutions for synthetic networks.
- **libRoadRunner 2.11.0** (libSBML + CVODE, C++) as an independent SBML
  implementation for the dynamics. It was not used to produce any value here.

Tolerances were `rtol = 1e-12` and `atol = 1e-16` in both engines for the
vendored models. The survey used `rtol = 1e-10` and `atol = 1e-14`.

| Quantity | This work | Authority | Error |
|---|---|---|---|
| Conservation laws, Kholodenko 2000 | 3 laws: MKKK, MKK and MAPK totals = 100, 300, 300 | exact `l·N = 0`; rank(N) + laws = 8; libRoadRunner moiety space | residual **0** (exact); same row space |
| Conservation laws, Huang & Ferrell 1996 | 7 laws (E1, E2, KKK, KK, K, KPase, KKPase totals) | same | residual **0**; same row space |
| Conservation laws, repressilator | none | rank(N) = 6 | exact |
| Trajectories, 3 models, 101 points | — | libRoadRunner | ≤ **6.9e-10** × species range |
| Drift of conserved totals along trajectories | — | exact invariance | ≤ **1.1e-13** relative |
| Steady state, Huang & Ferrell 1996 (stable) | max Re λ = −0.0716 | libRoadRunner steady state and reduced eigenvalues | state **1.7e-12**; eigenvalues **8.2e-10** |
| Fixed point, Kholodenko 2000 (unstable) | max Re λ = +0.00158 | same | state **1.5e-15**; eigenvalues **1.3e-9** |
| Fixed point, repressilator (unstable) | max Re λ = +0.0144 | same | state **1.2e-16**; eigenvalues **4.5e-10** |
| Limit-cycle period, Kholodenko 2000 (MKKK) | 1299.7705 time units | libRoadRunner trajectory | **8.9e-12** relative |
| Limit-cycle period, repressilator (PX) | 125.35416 time units | same | **1.8e-12** relative |
| First-order decay, `V = 2.5`, both species conventions | — | `A₀ e^{−kt}` | ≤ 1e-8 relative |
| Reversible `A ⇌ B`: relaxation, steady state, eigenvalue | — | closed form | 1e-8 / 1e-12 / 1e-6 |
| Survey trajectories, 49 models, `[0, 100]`, `rtol` 1e-10 | — | libRoadRunner | median **1.7e-9**; 47 of 49 ≤ 1e-6; worst 1.1e-5, which falls to **2.9e-7** at `rtol` 1e-12 |

No parameter was fitted.

**Survey of BIOMD0000000001–060.**
- **Read:** 49 of 60.
- **Refused by name (11):** rate rules (5), events (3), delays (2) and
  stoichiometryMath (1).
- **Referee check:** every model that was read was also simulated by
  libRoadRunner over the same window. The two Borghans 1997 calcium
  oscillators (BIOMD0000000044/045) agree least closely: 1.1e-5 and 1.3e-6 at
  `rtol` 1e-10. The script reruns them automatically at `rtol` 1e-12, where
  the gap falls to 2.9e-7 and 3.3e-8. In an exploratory check (not recorded in
  `results.json`), our own LSODA and Radau runs differed from each other by a
  similar amount. The shrinking gap points to integration error on sharp
  calcium spikes, not a semantic difference.

**The new invariant catches real inconsistencies.** Positive-mass consistency
fails, with exact certificates, for 8 of the 49 models that were read:
BIOMD0000000008, 015, 018, 035, 049, 050, 051 and 052. The model-level check
and the graph invariant agree on all 49. Three cases were inspected:

- **Chassagnole 2002, *E. coli* carbon metabolism (051).** The
  phosphotransferase reaction is written
  `cglcex + 65 cpep → 65 cg6p + 65 cpyr`. The model's own notes say the
  biomass-to-volume ratio "is taken into account by appropriately adjusting
  the stoichiometries", because both compartments have size 1. One glucose
  therefore formally yields 65 glucose-6-phosphate: a unit conversion hidden
  in stoichiometry. That is harmless for this model in isolation, but it would
  silently corrupt any mass balance or model composition built on it.
  Neutralising the factor does not make the model consistent. A second
  certificate runs through `vPGDH`, written
  `6-phosphogluconate → ribulose 5-phosphate` with no CO₂.
- **Vilar 2002 oscillator (035).** Transcription and translation are template
  reactions (`DA → DA + MA`, `MA → A + MA`), and `C → R` degrades the bound
  activator inside the complex. These are deliberate abstractions with
  precursors and degradation products omitted.
- **Gardner 1998 cell cycle (008).** `Z → C` and `Z → Y` degrade one partner
  inside a complex. This is also a deliberate abstraction.

A positive-mass failure is therefore a proven property of the network as
written, not proof of a modelling error. The invariant makes the abstraction
visible and machine-checkable. Judging whether it is acceptable needs the
model's context.

An earlier version of the check dropped boundary species before testing mass
balance. That made binding to a clamped ligand look like creation from nothing:
Schoeberl 2002 (019) was flagged because EGF is a boundary species. The final
rule, described under Method, removes that artifact without hiding real
template synthesis.

## Limits

- **Construct coverage.** 11 of the 60 surveyed models use constructs outside
  the subset: events, rate rules, delays and stoichiometryMath. They are
  refused, not approximated.
- **Units are not checked.** Numbers are used as written, as an SBML simulator
  would use them. A dimensionally inconsistent kinetic law is not detected.
- **Consistency is structural.** It ignores kinetic laws and modifiers, and it
  cannot tell an intended abstraction from an error.
- **Steady states depend on the starting point.** The solve finds the fixed
  point of the given stoichiometric class nearest the guess. Multistability is
  not enumerated.
- **Speed.** Kinetics run as Python closures. That is adequate here (the
  largest surveyed model, about 100 species, simulates in seconds), but it is
  neither JAX-differentiable nor fast. Making it so is a natural next step.
- **What agreement with libRoadRunner shows.** It establishes that both engines
  implement the same SBML semantics on these models. It does not validate the
  biology of any model.
- **No link to the rest of the stack.** Nothing here touches `vh.bridge`,
  occupancy, or patient data.

## Reproduce

```bash
pip install -e '.[dev]'                      # includes libroadrunner (referee only)
PYTHONPATH=. pytest -q tests/test_sbml.py
mkdir -p /tmp/biomodels && for i in $(seq -w 1 60); do id=BIOMD00000000$i; \
  curl -sSL -o /tmp/biomodels/$id.xml "https://www.ebi.ac.uk/biomodels/model/download/$id?filename=${id}_url.xml"; done
PYTHONPATH=. python scripts/validate_sbml.py --survey /tmp/biomodels   # writes results.json here
```

The survey files' SHA-256 values are recorded in `results.json`. A later
BioModels release may serve different bytes; compare against those hashes.

## References

- Hucka M. et al. The Systems Biology Markup Language (SBML): language specification for Level 3 Version 2 Core (2019); Level 2 Version 4 (2008).
- Gevorgyan A., Poolman M. G., Fell D. A. Detection of stoichiometric inconsistencies in biomolecular models. *Bioinformatics* 24:2245–2251 (2008).
- Welsh C. et al. libRoadRunner 2.0: a high performance SBML simulation and analysis library. *Bioinformatics* 39:btac770 (2023).
- Le Novère N. et al. BioModels Database. *Nucleic Acids Res.* 34:D689 (2006); models used under CC0 as stated in each file.
- Models, with the PubMed IDs annotated in each SBML file:
  - Kholodenko B. N. *Eur. J. Biochem.* 267:1583 (2000), PMID 10712587.
  - Huang C.-Y. F., Ferrell J. E. *PNAS* 93:10078 (1996), PMID 8816754.
  - Elowitz M. B., Leibler S. *Nature* 403:335 (2000), PMID 10659856.
  - Vilar J. M. G. et al. *PNAS* 99:5988 (2002), PMID 11972055.
  - Gardner T. S. et al. *PNAS* 95:14190 (1998), PMID 9826676.
  - Chassagnole C. et al. *Biotechnol. Bioeng.* 79:53 (2002). BIOMD0000000051 annotates PMID 17590932; it was not checked against this citation.
