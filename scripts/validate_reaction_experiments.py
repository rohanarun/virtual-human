"""Reproduce synthetic reaction/intervention validation; no fitted parameters.

Run with PYTHONPATH=. python scripts/validate_reaction_experiments.py [--output PATH]
Exact solutions and independent hand-written DOP853 equations are the referees.
"""
import argparse
import json
import platform
from math import exp
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from vh.graph.reactions import Phase, Reaction, ReactionNetwork, Species, rate_unit, run_protocol

SOURCE = ("Artificial mathematical fixture, all rates/states selected for numerical verification",)


def make_network(ids, reactions):
    return ReactionNetwork([Species(s, SOURCE) for s in ids],
        [Reaction(name, left, right, rate, rate_unit(sum(left.values())), SOURCE)
         for name, left, right, rate in reactions],
        context={"system": "synthetic", "compartment": "fixed well-mixed volume"}, provenance=SOURCE)


def run():
    linear = make_network(["A", "B"], [("forward", {"A": 1}, {"B": 1}, 0.7)])
    times = np.linspace(0, 5, 21)
    trace = linear.simulate({"A": 2, "B": 0}, times, max_step_s=0.01,
                            sensitivity_reactions=["forward"])
    a = 2*np.exp(-0.7*times)
    da = -0.7*times*a
    linear_error = float(np.max(np.abs(trace.concentrations-np.column_stack((a, 2-a)))))
    linear_sensitivity_error = float(np.max(np.abs(trace.sensitivities[:, :, 0]-np.column_stack((da, -da)))))
    dimer = make_network(["A", "D"], [("dimerize", {"A": 2}, {"D": 1}, 0.3)])
    dtimes = np.linspace(0, 4, 17)
    dtrace = dimer.simulate({"A": 2, "D": 0}, dtimes, max_step_s=0.005,
                           sensitivity_reactions=["dimerize"])
    dexact = 2/(1+1.2*dtimes)
    ds_exact = -2*1.2*dtimes/(1+1.2*dtimes)**2
    dimer_error = float(np.max(np.abs(dtrace.concentrations[:, 0]-dexact)))
    dimer_sensitivity_error = float(np.max(np.abs(dtrace.sensitivities[:, 0, 0]-ds_exact)))
    pool_drift = float(np.max(np.abs(dtrace.concentrations @ [1, 2]-2)))
    binding = make_network(["A", "B", "C"], [
        ("on", {"A": 1, "B": 1}, {"C": 1}, 1.1),
        ("off", {"C": 1}, {"A": 1, "B": 1}, 0.2)])
    btimes = np.linspace(0, 5, 26)
    btrace = binding.simulate({"A": 1.2, "B": 0.8, "C": 0}, btimes, max_step_s=0.005,
                              sensitivity_reactions=["on", "off"])

    def referee(on=1.1, off=0.2):
        def rhs(t, state):
            v = on*state[0]*state[1]-off*state[2]
            return [-v, -v, v]
        solution = solve_ivp(rhs, [0, 5], [1.2, 0.8, 0], method="DOP853",
                             t_eval=btimes, rtol=2e-13, atol=2e-14)
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution.y.T

    binding_error = float(np.max(np.abs(btrace.concentrations-referee())))
    sensitivity_errors = []
    for epsilon in (1e-3, 1e-4):
        errors = []
        for j in range(2):
            plus, minus = [1.1, 0.2], [1.1, 0.2]
            plus[j] *= 1+epsilon
            minus[j] *= 1-epsilon
            reference = (referee(*plus)-referee(*minus))/(2*epsilon)
            errors.append(float(np.max(np.abs(btrace.sensitivities[:, :, j]-reference))))
        sensitivity_errors.append({"relative_difference_step": epsilon, "max_abs_error_mol_per_L": max(errors)})

    convergence = []
    for step in (0.2, 0.1, 0.05):
        result = linear.simulate({"A": 2, "B": 0}, [0, 2], max_step_s=step)
        convergence.append({"max_step_s": step, "abs_error_mol_per_L": abs(result.final["A"]-2*exp(-1.4))})

    def order_test(model, initial):
        retention = {s: 0.4 for s in model.ids}
        before = run_protocol(model, initial, [Phase("retention_then_flow", 3, SOURCE, retain=retention)], max_step_s=0.01)
        after = run_protocol(model, initial, [Phase("flow", 3, SOURCE),
                             Phase("retention", 0, SOURCE, retain=retention)], max_step_s=0.01)
        left, right = before[-1].trajectory.final, after[-1].trajectory.final
        return {"retention_then_flow_mol_per_L": left, "flow_then_retention_mol_per_L": right,
                "max_abs_difference_mol_per_L": max(abs(left[s]-right[s]) for s in model.ids)}

    linear_order = order_test(linear, {"A": 2, "B": 0})
    nonlinear_order = order_test(dimer, {"A": 2, "D": 0})
    jump = run_protocol(linear, {"A": 2, "B": 0}, [
        Phase("paused", 2, SOURCE, rate_scales={"forward": 0}),
        Phase("external_exchange", 1, SOURCE, retain={"A": 0.5}, add={"B": 0.3})], max_step_s=0.01)
    metrics = {
        "linear_trajectory_max_abs_error_mol_per_L": linear_error,
        "linear_sensitivity_max_abs_error_mol_per_L": linear_sensitivity_error,
        "dimer_trajectory_max_abs_error_mol_per_L": dimer_error,
        "dimer_sensitivity_max_abs_error_mol_per_L": dimer_sensitivity_error,
        "dimer_conserved_monomers_max_drift_mol_per_L": pool_drift,
        "nonlinear_binding_vs_DOP853_max_abs_error_mol_per_L": binding_error,
    }
    thresholds = {name: 2e-10 for name in metrics}
    thresholds["dimer_conserved_monomers_max_drift_mol_per_L"] = 1e-14
    for name, value in metrics.items():
        if value > thresholds[name]:
            raise AssertionError(f"{name}: {value} exceeds {thresholds[name]}")
    if sensitivity_errors[-1]["max_abs_error_mol_per_L"] > 2e-9:
        raise AssertionError("binding sensitivity mismatch")
    if linear_order["max_abs_difference_mol_per_L"] > 1e-14:
        raise AssertionError("linear scalar-retention commutativity failed")
    if nonlinear_order["max_abs_difference_mol_per_L"] < 0.1:
        raise AssertionError("nonlinear counterexample failed")
    return {
        "schema_version": 1,
        "scope": "Synthetic mathematical verification only; no biological parameter fit or validation",
        "provenance": SOURCE,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "units": {"concentration": "mol/L", "time": "s", "sensitivity_parameter": "dimensionless local effective-rate multiplier"},
        "metrics": metrics, "acceptance_thresholds": thresholds,
        "binding_sensitivity_vs_independent_central_difference": sensitivity_errors,
        "rk4_step_halving": convergence,
        "exact_dimer_pool_residuals": {"correct_A_plus_2D": {k: str(v) for k, v in dimer.pool_residuals({"A": 1, "D": 2}).items()},
                                        "incorrect_A_plus_D": {k: str(v) for k, v in dimer.pool_residuals({"A": 1, "D": 1}).items()}},
        "linear_scalar_retention_order": linear_order,
        "nonlinear_scalar_retention_order": nonlinear_order,
        "external_exchange": {"before_total_mol_per_L": sum(jump[1].before.values()),
                               "after_total_mol_per_L": sum(jump[1].after.values()),
                               "final_total_mol_per_L": sum(jump[1].trajectory.final.values()),
                               "expected_after_total_mol_per_L": 1.3},
        "limits": ["No SBML importer or Track E completion", "No clinical predictions or parameter estimation",
                   "Finite-step explicit RK4; reject negative states; choose converged step for each model",
                   "Sensitivity holds initial state fixed and does not propagate through protocol jumps",
                   "Conservation laws are structural; elemental composition is not inferred"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(run(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
