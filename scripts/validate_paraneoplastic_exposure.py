"""Paraneoplastic antigen-exposure experiment on the VH reaction engine.

Every quantity is invented and dimensionless, reused as mol/L and seconds only
because the engine requires units. This is not a physiological conversion, an
anti-Yo disease model, a patient simulation or a treatment comparison.

Authorities: exact kinetic invariants found by ``kinetic_invariant_basis``,
closed-form integrals, and (optionally) Project Asma's exact rational module
``tools/analyze_paraneoplastic_exposure.py`` from a trusted local checkout.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
from hashlib import sha256
import importlib.util
import json
from math import exp, log
from pathlib import Path
import subprocess

import numpy as np

from vh.graph.reactions import Phase, Reaction, ReactionNetwork, Species, rate_unit, run_protocol

SOURCE = ("Original artificial fixture for Project Asma exposure mathematics; no empirical calibration",)
ROOT = Path(__file__).resolve().parents[1]
# Base rates: effector gain/decay, injury, marker clearance, autonomous seeding/injury.
RATES = {"present": F(1), "exposure": F(1), "kill_release": F(1), "kill_silent": F(1),
         "decay": F(1), "hazard": F(1, 100), "injure": F(1, 100), "clear": F(1, 2),
         "seed": F(1, 4), "cns_decay": F(1), "cns_hazard": F(1, 200), "cns_injure": F(1, 200)}
SPECIES = ("A", "E", "N", "F", "U", "R", "H", "C")
STEP = 0.005


def build_network():
    """Tumor burden A, effector E, neurons N, injury marker F, CNS effector C.

    U accumulates burden exposure (integral of A), R burden killed in the body,
    H cumulative hazard. Rate scales choose the hypothesis in each phase.
    """
    first = "1/s"
    reactions = [
        Reaction("present", {"A": 1}, {"A": 1, "E": 1}, 1.0, first, SOURCE),
        Reaction("exposure", {"A": 1}, {"A": 1, "U": 1}, 1.0, first, SOURCE),
        Reaction("kill_release", {"A": 1}, {"E": 1, "R": 1}, 1.0, first, SOURCE),
        Reaction("kill_silent", {"A": 1}, {"R": 1}, 1.0, first, SOURCE),
        Reaction("decay", {"E": 1}, {}, 1.0, first, SOURCE),
        Reaction("hazard", {"E": 1}, {"E": 1, "H": 1}, 0.01, first, SOURCE),
        Reaction("injure", {"N": 1, "E": 1}, {"E": 1, "F": 1}, 0.01, rate_unit(2), SOURCE),
        Reaction("clear", {"F": 1}, {}, 0.5, first, SOURCE),
        Reaction("seed", {"E": 1}, {"E": 1, "C": 1}, 0.25, first, SOURCE),
        Reaction("cns_decay", {"C": 1}, {}, 1.0, first, SOURCE),
        Reaction("cns_hazard", {"C": 1}, {"C": 1, "H": 1}, 0.005, first, SOURCE),
        Reaction("cns_injure", {"N": 1, "C": 1}, {"C": 1, "F": 1}, 0.005, rate_unit(2), SOURCE),
    ]
    return ReactionNetwork([Species(s, SOURCE) for s in SPECIES], reactions,
                           context={"system": "synthetic", "compartment": "constant-volume, dimensionless"},
                           provenance=SOURCE)


def scales(hypothesis, *, chemo=0.0, presentation=None, autonomous=False):
    """Rate scales for one phase; chemo is the kill rate constant in that phase.

    Pure hypotheses by default: burden-driven presentation scale 1, release-driven
    presentation scale 0. A positive ``presentation`` with ``release`` is the
    mixture (spontaneous turnover plus treatment release).
    """
    s = {"kill_release": 0.0, "kill_silent": 0.0, "seed": 0.0, "cns_decay": 1.0,
         "cns_hazard": 0.0, "cns_injure": 0.0, "present": 0.0}
    if hypothesis == "burden":
        s.update(present=1.0 if presentation is None else presentation, kill_silent=chemo)
    elif hypothesis == "release":
        s.update(present=0.0 if presentation is None else presentation, kill_release=chemo)
    else:
        raise ValueError("hypothesis must be burden or release")
    if autonomous:
        s.update(seed=1.0, cns_decay=0.0, cns_hazard=1.0, cns_injure=1.0)
    return s


def initial(burden=100.0, effector=0.0):
    state = dict.fromkeys(SPECIES, 0.0)
    state.update(A=burden, E=effector, N=1.0)
    return state


def final(results):
    return results[-1].trajectory.final


def run(network, state, phases):
    return run_protocol(network, state, phases, max_step_s=STEP)


def discrete_schedule(network, hypothesis, operations, tail=0.0):
    """Instantaneous operations, each followed by one holding unit (Asma's null case).

    A chemotherapy release is explicit caller logic: in the release hypothesis the
    killed burden is added to E and R at the event; otherwise only to R.
    """
    state, phases = initial(), []
    held = scales(hypothesis)
    for i, (kind, keep) in enumerate(operations):
        killed = state["A"] * (1 - keep) if kind == "chemo" else 0.0
        add = {"R": killed}
        if hypothesis == "release":
            add["E"] = killed
        phase = Phase(f"{i}-{kind}", 1, SOURCE, retain={"A": keep}, add=add, rate_scales=held)
        state = final(run(network, state, [phase]))
        phases.append(state)
    if tail:
        state = final(run(network, state, [Phase("tail", tail, SOURCE, rate_scales=held)]))
    return state


def continuous_schedule(network, hypothesis, operations):
    """Surgery is instantaneous then held; chemotherapy kills continuously for one unit."""
    phases = []
    for i, (kind, keep) in enumerate(operations):
        if kind == "surgery":
            phases.append(Phase(f"{i}-surgery", 1, SOURCE, retain={"A": keep}, rate_scales=scales(hypothesis)))
        else:
            phases.append(Phase(f"{i}-chemo", 1, SOURCE, rate_scales=scales(hypothesis, chemo=-log(keep))))
    return final(run(network, initial(), phases))


def continuous_closed_form(operations, burden=100.0):
    """Exact integrals for the continuous schedule: exposure, release, end burden."""
    exposure = released = removed = 0.0
    for kind, keep in operations:
        if kind == "surgery":
            removed += burden * (1 - keep)
            burden *= keep
            exposure += burden
        else:
            exposure += burden * (1 - keep) / -log(keep)
            released += burden * (1 - keep)
            burden *= keep
    return {"exposure": exposure, "released": released, "removed": removed, "final": burden}


def sweep(q, a, cycles=6):
    rows = []
    for position in range(cycles + 1):
        ops = [("chemo", a)] * cycles
        ops.insert(position, ("surgery", q))
        rows.append(ops)
    return rows


def invariant_weights(hypothesis):
    """Exposure identity H + (kappa/delta) E - (kappa*gain/delta) * source, exactly."""
    kappa, delta = RATES["hazard"], RATES["decay"]
    source = "U" if hypothesis == "burden" else "R"
    return {"H": F(1), "E": kappa / delta, source: -kappa * RATES["present" if hypothesis == "burden" else "kill_release"] / delta}


def invariant_value(state, weights):
    return sum(float(w) * state[s] for s, w in weights.items())


def revision(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def load_asma(asma_root):
    path = asma_root / "tools/analyze_paraneoplastic_exposure.py"
    spec = importlib.util.spec_from_file_location("asma_exposure", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the trusted Asma exact module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def validate(asma_root=None):
    network = build_network()
    exact = {k: v for k, v in RATES.items()}
    checks = {}

    # 1. The finder recovers the exposure identity for each hypothesis, exactly.
    found = {}
    for hypothesis in ("burden", "release"):
        for chemo in (0.0, 1.0):
            w = invariant_weights(hypothesis)
            residuals = network.kinetic_residuals(w, rate_scales=scales(hypothesis, chemo=chemo), exact_rates=exact)
            found[f"{hypothesis}_chemo_{int(chemo)}"] = all(v == 0 for v in residuals.values())
    checks["exposure_identity_is_exact_kinetic_invariant"] = found

    # 2. Discrete schedules: exposure/release bookkeeping, invariant drift, Asma bridge.
    discrete, drift, asma_errors = [], 0.0, []
    asma = None
    if asma_root is not None:
        asma, asma_path = load_asma(asma_root)
    for q in (0.1, 0.75):
        for hypothesis in ("burden", "release"):
            w = invariant_weights(hypothesis)
            states = [discrete_schedule(network, hypothesis, ops, tail=12.0) for ops in sweep(q, 0.5)]
            for s in states:
                drift = max(drift, abs(invariant_value(s, w) - invariant_value(initial(), w)))
            key = "U" if hypothesis == "burden" else "R"
            gain = float(RATES["hazard"] * RATES["present" if hypothesis == "burden" else "kill_release"] / RATES["decay"])
            k_over_d = float(RATES["hazard"] / RATES["decay"])
            # Hazard difference between surgery positions 3 and 0 from the identity at finite time.
            late, early = states[3], states[0]
            identity_dh = gain * (late[key] - early[key]) - k_over_d * (late["E"] - early["E"])
            discrete.append({"surgery_keep": q, "hypothesis": hypothesis,
                             "exposure_by_position": [s[key] for s in states],
                             "best_position": int(np.argmin([s[key] for s in states])),
                             "hazard_difference_late_minus_early": late["H"] - early["H"],
                             "identity_prediction": identity_dh,
                             "identity_error": abs(late["H"] - early["H"] - identity_dh),
                             "survival_ratio_late_to_early": late["N"] / early["N"],
                             "survival_exp_minus_h_error": max(abs(s["N"] - exp(-s["H"])) for s in states)})
            if asma is not None:
                for position, s in enumerate(states):
                    ops = [("chemo", F(1, 2))] * 6
                    ops.insert(position, ("surgery", F(1, 10) if q == 0.1 else F(3, 4)))
                    expected = asma.run_schedule(F(100), ops)
                    target = expected["burden_auc"] if hypothesis == "burden" else expected["killed_in_body"]
                    # Tail adds final burden x tail duration to U; release is complete at the last event.
                    if hypothesis == "burden":
                        target += expected["final"] * 12
                    asma_errors.append(abs(s[key] - float(target)))
    checks["discrete_schedules"] = discrete
    checks["max_exposure_invariant_drift"] = drift

    # 3. Continuous chemotherapy: closed-form integrals and the order criteria.
    continuous, closed_error = [], 0.0
    for q in (0.1, 0.75):
        for hypothesis in ("burden", "release"):
            rows = sweep(q, 0.5)
            states = [continuous_schedule(network, hypothesis, ops) for ops in rows]
            forms = [continuous_closed_form(ops) for ops in rows]
            for s, c in zip(states, forms):
                closed_error = max(closed_error, abs(s["U"] - c["exposure"]) / c["exposure"],
                                   abs(s["R"] - c["released"]) / max(c["released"], 1e-300),
                                   abs(s["A"] - c["final"]) / c["final"])
            exposures = [s["U" if hypothesis == "burden" else "R"] for s in states]
            continuous.append({"surgery_keep": q, "hypothesis": hypothesis,
                               "best_position": int(np.argmin(exposures)),
                               "continuous_criterion_surgery_first": q < 1 / (1 + log(2)) if hypothesis == "burden" else True})
    checks["continuous_schedules"] = continuous
    checks["max_continuous_closed_form_relative_error"] = closed_error

    # 4. Committed injury after complete antigen removal, and the autonomous contrast.
    regimes = {}
    for label, autonomous in (("antigen_driven", False), ("autonomous_cns", True)):
        start = initial(burden=10.0, effector=10.0)
        held = scales("burden", autonomous=autonomous)
        at_removal = final(run(network, start, [Phase("established", 5, SOURCE, rate_scales=held)]))
        removed = {**at_removal, "A": 0.0}
        trajectory = network.simulate(removed, list(np.arange(0, 20.0001, 1.0)), max_step_s=STEP,
                                      rate_scales=held)
        f_series = trajectory.concentrations[:, SPECIES.index("F")]
        row = {"marker_ratio_t10_to_t2": float(f_series[10] / f_series[2]),
               "marker_ratio_t20_to_t2": float(f_series[20] / f_series[2]),
               "remaining_neurons_at_end": trajectory.final["N"]}
        if not autonomous:
            bound = float(RATES["hazard"] / RATES["decay"]) * at_removal["E"] * (1 - exp(-20))
            row["committed_hazard_bound"] = bound
            row["committed_hazard_observed"] = trajectory.final["H"] - at_removal["H"]
            row["committed_error"] = abs(row["committed_hazard_observed"] - bound)
        regimes[label] = row
    checks["regimes_after_removal"] = regimes

    # 5. No-rise theorem in continuous time, and the release signature.
    signature = {}
    turnover = 0.1
    for hypothesis in ("burden", "release"):
        steady = turnover * 100.0  # alpha*d*A0/delta with alpha = delta = 1
        state = initial(effector=steady)
        phases = [Phase("chemo-1", 1, SOURCE, rate_scales=scales(hypothesis, chemo=log(2), presentation=turnover)),
                  Phase("hold", 1, SOURCE, rate_scales=scales(hypothesis, presentation=turnover)),
                  Phase("surgery", 2, SOURCE, retain={"A": 0.5}, rate_scales=scales(hypothesis, presentation=turnover))]
        series = []
        for result in run_protocol(network, state, phases, max_step_s=STEP):
            traj = network.simulate(result.after, list(np.linspace(0, result.phase.duration_s, 21)),
                                    max_step_s=STEP, rate_scales=result.phase.rate_scales)
            series.extend(traj.concentrations[:, SPECIES.index("E")].tolist())
        diffs = np.diff(series)
        signature[hypothesis] = {"max_rise": float(diffs.max()),
                                 "never_rises": bool(diffs.max() <= 1e-12),
                                 "rises_during_first_chemo": bool(series[20] > series[0] + 1e-9)}
    checks["steady_state_signature"] = signature

    tolerances = {"identity": 1e-10, "drift": 1e-10, "closed_form_relative": 1e-8, "survival": 1e-8,
                  "committed": 1e-8, "asma": 1e-9}
    passed = (all(found.values())
              and drift <= tolerances["drift"]
              and all(r["identity_error"] <= tolerances["identity"] for r in discrete)
              and all(r["survival_exp_minus_h_error"] <= tolerances["survival"] for r in discrete)
              and closed_error <= tolerances["closed_form_relative"]
              and all(r["best_position"] == (0 if r["continuous_criterion_surgery_first"] else 6) for r in continuous)
              and regimes["antigen_driven"]["committed_error"] <= tolerances["committed"]
              and regimes["autonomous_cns"]["marker_ratio_t20_to_t2"] > 10 * regimes["antigen_driven"]["marker_ratio_t20_to_t2"]
              and signature["burden"]["never_rises"] and signature["release"]["rises_during_first_chemo"]
              and (not asma_errors or max(asma_errors) <= tolerances["asma"]))
    result = {
        "status": "pass" if passed else "fail",
        "scope": "synthetic_mathematics_and_software_only_not_biological_validation",
        "as_of": "2026-11-05",
        "authorities": ["exact kinetic invariants (rational nullspace)", "closed-form integrals",
                        "Project Asma exact rational module (optional bridge)"],
        "integration": {"method": "fixed-step RK4", "max_step": STEP},
        "tolerances": tolerances,
        "checks": checks,
        "virtual_human_checkout": revision(ROOT),
        "source_sha256": {"vh_reactions_module": sha256((ROOT / "vh/graph/reactions.py").read_bytes()).hexdigest(),
                          "validation_script": sha256(Path(__file__).read_bytes()).hexdigest()},
        "limitations": ["invented rates, burdens and units", "no physiological conversion",
                        "tumor growth, resistance and order-dependent surgery omitted",
                        "no anti-Yo, NfL, patient or treatment-benefit claim"],
    }
    if asma is not None:
        result["asma_bridge"] = {"asma_checkout": revision(asma_root),
                                 "asma_module_sha256": sha256(asma_path.read_bytes()).hexdigest(),
                                 "comparisons": len(asma_errors),
                                 "max_absolute_error": max(asma_errors)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asma-root", type=Path, help="trusted local Project Asma checkout (optional bridge)")
    args = parser.parse_args()
    outcome = validate(args.asma_root.resolve() if args.asma_root else None)
    print(json.dumps(outcome, indent=2, sort_keys=True))
    raise SystemExit(0 if outcome["status"] == "pass" else 1)
