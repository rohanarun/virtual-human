"""Independent exact kinetics, mathematical boundaries and malformed inputs."""
from fractions import Fraction
from math import exp

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vh.graph.core import Graph
from vh.graph.reactions import Phase, Reaction, ReactionNetwork, Species, rate_unit, run_protocol

SOURCE = ("Artificial mathematical fixture; not a biological measurement",)


def reaction(name, left, right, rate):
    return Reaction(name, left, right, rate, rate_unit(sum(left.values())), SOURCE)


def network(ids, reactions):
    return ReactionNetwork([Species(s, SOURCE) for s in ids], reactions,
                           context={"system": "synthetic", "compartment": "constant-volume"},
                           provenance=SOURCE)


def conversion(k=0.7):
    return network(["A", "B"], [reaction("forward", {"A": 1}, {"B": 1}, k)])


def dimer():
    return network(["A", "D"], [reaction("dimerize", {"A": 2}, {"D": 1}, 0.3)])


def test_irreversible_exact_trajectory_and_rate_scale_sensitivity():
    times = np.linspace(0, 5, 21)
    trace = conversion().simulate({"B": 0, "A": 2}, times, max_step_s=0.01,
                                  sensitivity_reactions=["forward"])
    a = 2 * np.exp(-0.7*times)
    expected = np.column_stack((a, 2-a))
    da = -0.7*times*a
    np.testing.assert_allclose(trace.concentrations, expected, atol=2e-11, rtol=0)
    np.testing.assert_allclose(trace.sensitivities[:, :, 0], np.column_stack((da, -da)), atol=1e-10, rtol=0)
    assert trace.species == ("A", "B")  # Mapping order never chooses state order.


def test_reversible_equilibrium_and_transient_against_exact_solution():
    model = network(["A", "B"], [reaction("f", {"A": 1}, {"B": 1}, 0.7),
                                  reaction("r", {"B": 1}, {"A": 1}, 0.3)])
    times = np.linspace(0, 10, 21)
    trace = model.simulate({"A": 2, "B": 1}, times, max_step_s=0.01)
    a = 0.9+1.1*np.exp(-times)
    np.testing.assert_allclose(trace.concentrations[:, 0], a, atol=4e-11, rtol=0)
    steady = model.simulate({"A": 0.9, "B": 2.1}, [0, 10], max_step_s=0.05)
    np.testing.assert_allclose(steady.final["A"], 0.9, atol=1e-14)


def test_nonlinear_dimer_exact_kinetics_sensitivity_and_monomer_conservation():
    times = np.linspace(0, 4, 17)
    trace = dimer().simulate({"A": 2, "D": 0}, times, max_step_s=0.005,
                             sensitivity_reactions=["dimerize"])
    a = 2/(1+1.2*times)
    da = -2*1.2*times/(1+1.2*times)**2
    np.testing.assert_allclose(trace.concentrations[:, 0], a, atol=4e-11, rtol=0)
    np.testing.assert_allclose(trace.sensitivities[:, 0, 0], da, atol=2e-10, rtol=0)
    np.testing.assert_allclose(trace.concentrations @ [1, 2], 2, atol=1e-14, rtol=0)
    assert dimer().pool_residuals({"A": 1, "D": 2}) == {"dimerize": 0}
    assert dimer().pool_residuals({"A": 1, "D": 1}) == {"dimerize": -1}
    assert dimer().conservation_basis() == ({"A": Fraction(1, 2), "D": Fraction(1)},)


def test_binding_nonlinear_trajectory_and_sensitivities_vs_independent_referee():
    model = network(["A", "B", "C"], [reaction("on", {"A": 1, "B": 1}, {"C": 1}, 1.1),
                                      reaction("off", {"C": 1}, {"A": 1, "B": 1}, 0.2)])
    times = np.linspace(0, 5, 26)
    trace = model.simulate({"A": 1.2, "B": 0.8, "C": 0}, times, max_step_s=0.005,
                           sensitivity_reactions=["on", "off"])

    # Referee equations deliberately do not reuse model stoichiometry or flux code.
    def referee(on=1.1, off=0.2):
        def rhs(t, x):
            binding = on*x[0]*x[1]-off*x[2]
            return [-binding, -binding, binding]
        result = solve_ivp(rhs, [0, 5], [1.2, 0.8, 0], t_eval=times,
                           method="DOP853", rtol=2e-13, atol=2e-14)
        assert result.success
        return result.y.T

    np.testing.assert_allclose(trace.concentrations, referee(), atol=5e-11, rtol=0)
    epsilon = 1e-4
    for j in range(2):
        plus, minus = [1.1, 0.2], [1.1, 0.2]
        plus[j] *= 1+epsilon
        minus[j] *= 1-epsilon
        derivative = (referee(*plus)-referee(*minus))/(2*epsilon)
        np.testing.assert_allclose(trace.sensitivities[:, :, j], derivative, atol=2e-9, rtol=0)
    for pool in model.conservation_basis():
        assert set(model.pool_residuals(pool).values()) == {0}


def test_rk4_convergence_has_fourth_order_behavior():
    exact = 2*exp(-0.7*2)
    errors = [abs(conversion().simulate({"A": 2, "B": 0}, [0, 2], max_step_s=h).final["A"]-exact)
              for h in (0.2, 0.1, 0.05)]
    assert 14 < errors[0]/errors[1] < 18
    assert 14 < errors[1]/errors[2] < 18


def test_zero_rate_zero_concentration_and_zero_order_source():
    inert = conversion(0).simulate({"A": 2, "B": 0}, [0, 1], max_step_s=0.1,
                                   sensitivity_reactions=["forward"])
    assert inert.final == {"A": 2, "B": 0}
    assert not inert.sensitivities.any()
    model = network(["A"], [reaction("source", {}, {"A": 1}, 0.4)])
    trace = model.simulate({"A": 0}, [0, 3], max_step_s=0.01, sensitivity_reactions=["source"])
    np.testing.assert_allclose(trace.final["A"], 1.2, atol=1e-14)
    np.testing.assert_allclose(trace.sensitivities[-1, 0, 0], 1.2, atol=1e-14)
    assert model.conservation_basis() == ()


def test_no_reactions_and_full_rank_stoichiometry():
    model = network(["A", "B"], [])
    assert len(model.conservation_basis()) == 2
    assert model.simulate({"A": 1, "B": 2}, [0, 1], max_step_s=1).final == {"A": 1, "B": 2}
    open_model = network(["A", "B"], [reaction("a", {"A": 1}, {}, 1), reaction("b", {"B": 1}, {}, 1)])
    assert open_model.conservation_basis() == ()


def test_graph_export_preserves_catalyst_and_stoichiometry_with_colliding_ids():
    model = network(["r", "B", "E"], [reaction("r", {"r": 2, "E": 1}, {"B": 1, "E": 1}, 0.1)])
    graph = Graph.from_dict(model.to_graph().to_dict())
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 4
    assert graph.out_edges("species:r")[0].attrs["stoichiometry"] == 2
    assert graph.node("species:r").attrs["network_provenance"] == list(SOURCE)
    assert graph.node("reaction:r").attrs["context"] == dict(model.context)
    assert {e.type for e in graph.in_edges("species:E")+graph.out_edges("species:E")} == {"reactant", "product"}
    assert model.pool_residuals({"E": 1}) == {"r": 0}


def test_protocol_records_external_balance_and_resets_rate_scales():
    model = conversion()
    phases = [Phase("paused", 2, SOURCE, rate_scales={"forward": 0}),
              Phase("remove-add", 1, SOURCE, retain={"A": 0.5}, add={"B": 0.3})]
    result = run_protocol(model, {"A": 2, "B": 0}, phases, max_step_s=0.01)
    assert result[0].trajectory.final == {"A": 2, "B": 0}
    assert result[1].before == {"A": 2, "B": 0}
    assert result[1].after == {"A": 1, "B": 0.3}
    assert result[1].start_s == 2
    assert sum(result[1].trajectory.final.values()) == pytest.approx(1.3, abs=1e-14, rel=0)
    assert result[1].trajectory.final["A"] == pytest.approx(exp(-0.7), abs=1e-10, rel=0)


def test_scalar_retention_commutes_with_linear_flow_but_not_nonlinear_flow():
    def ordered(model, initial):
        retained = {s: 0.4 for s in model.ids}
        before = run_protocol(model, initial, [Phase("retain", 3, SOURCE, retain=retained)], max_step_s=0.01)
        after = run_protocol(model, initial, [Phase("evolve", 3, SOURCE),
                                              Phase("retain", 0, SOURCE, retain=retained)], max_step_s=0.01)
        return before[-1].trajectory.final, after[-1].trajectory.final
    left, right = ordered(conversion(), {"A": 2, "B": 0})
    np.testing.assert_allclose(list(left.values()), list(right.values()), atol=1e-14, rtol=0)
    left, right = ordered(dimer(), {"A": 2, "D": 0})
    assert left["A"] == pytest.approx(0.8/(1+0.6*0.8*3), abs=1e-9, rel=0)
    assert right["A"] == pytest.approx(0.4*2/(1+0.6*2*3), abs=1e-9, rel=0)
    assert abs(left["A"]-right["A"]) > 0.1


def test_selective_retention_need_not_commute_even_in_linear_network():
    model = conversion()
    before = run_protocol(model, {"A": 2, "B": 0}, [Phase("retain", 1, SOURCE, retain={"A": 0.5})], max_step_s=0.01)
    after = run_protocol(model, {"A": 2, "B": 0}, [Phase("evolve", 1, SOURCE),
                         Phase("retain", 0, SOURCE, retain={"A": 0.5})], max_step_s=0.01)
    assert before[-1].trajectory.final["B"] == pytest.approx(1-exp(-0.7), abs=1e-10, rel=0)
    assert after[-1].trajectory.final["B"] == pytest.approx(2*(1-exp(-0.7)), abs=1e-10, rel=0)


@pytest.mark.parametrize("initial", [{"A": 1}, {"A": 1, "B": 0, "C": 2}, {"A": -1, "B": 0},
                                    {"A": float("nan"), "B": 0}, {"A": True, "B": 0}])
def test_bad_initial_state_rejected(initial):
    with pytest.raises(ValueError):
        conversion().simulate(initial, [0, 1], max_step_s=0.01)


@pytest.mark.parametrize("times", [[], [1], [0, 0], [0, -1], [0, float("inf")], [0, True]])
def test_bad_times_rejected(times):
    with pytest.raises(ValueError):
        conversion().simulate({"A": 1, "B": 0}, times, max_step_s=0.01)


@pytest.mark.parametrize("kwargs", [{"max_step_s": 0}, {"max_step_s": -1}, {"max_step_s": 1e-320},
    {"max_step_s": 0.01, "max_steps": 2}, {"max_step_s": 0.01, "max_steps": True},
    {"max_step_s": 0.01, "rate_scales": {"wrong": 1}},
    {"max_step_s": 0.01, "rate_scales": {"forward": -1}},
    {"max_step_s": 0.01, "sensitivity_reactions": ["wrong"]},
    {"max_step_s": 0.01, "sensitivity_reactions": ["forward", "forward"]}])
def test_bad_solver_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        conversion().simulate({"A": 1, "B": 0}, [0, 1], **kwargs)


def test_unstable_step_rejected_without_clipping():
    with pytest.raises(ValueError, match="negative integration"):
        conversion(10).simulate({"A": 1, "B": 0}, [0, 1], max_step_s=1)


def test_nonfinite_effective_rate_rejected_even_without_time_evolution():
    with pytest.raises(ValueError, match="effective rate"):
        conversion(1e300).simulate({"A": 1, "B": 0}, [0], max_step_s=1, rate_scales={"forward": 1e300})


@pytest.mark.parametrize("left", [{"A": 0}, {"A": -1}, {"A": 0.5}, {"A": True}])
def test_invalid_stoichiometry_rejected(left):
    with pytest.raises(ValueError):
        Reaction("r", left, {"B": 1}, 1, "1/s", SOURCE)


def test_invalid_network_and_rate_semantics_rejected():
    with pytest.raises(ValueError, match="unit"):
        Reaction("r", {"A": 2}, {}, 1, "1/s", SOURCE)
    with pytest.raises(ValueError, match="unique"):
        network(["A", "A"], [])
    r = reaction("r", {"A": 1}, {}, 1)
    with pytest.raises(ValueError, match="unique"):
        network(["A"], [r, r])
    with pytest.raises(ValueError, match="unknown"):
        network(["B"], [r])
    with pytest.raises(ValueError):
        Species("A", ())
    with pytest.raises(ValueError):
        Species("A", "not a tuple")
    with pytest.raises(ValueError):
        conversion().pool_residuals({"A": 0.5})
    with pytest.raises(ValueError):
        conversion().pool_residuals({"wrong": 1})


def test_phase_validation_and_defensive_copies():
    retain = {"A": 0.5}
    phase = Phase("phase", 0, SOURCE, retain=retain)
    retain["A"] = 0
    assert phase.retain["A"] == 0.5
    with pytest.raises(TypeError):
        phase.retain["A"] = 0
    for kwargs in ({"retain": {"A": 1.1}}, {"add": {"A": -1}}, {"rate_scales": {"r": float("nan")}}):
        with pytest.raises(ValueError):
            Phase("bad", 1, SOURCE, **kwargs)
    for phases in ([], [phase, phase], [Phase("wrong", 0, SOURCE, retain={"C": 1})],
                   [Phase("wrong", 0, SOURCE, rate_scales={"C": 1})]):
        with pytest.raises(ValueError):
            run_protocol(conversion(), {"A": 1, "B": 0}, phases, max_step_s=0.1)


def test_tiny_time_interval_does_not_underflow_step_count():
    result = conversion().simulate({"A": 1, "B": 0}, [0, 1e-300], max_step_s=1e100)
    assert result.final["A"] == 1
    assert result.final["B"] == pytest.approx(0.7e-300, rel=1e-14, abs=0)


def test_protocol_time_overflow_rejected_before_running():
    with pytest.raises(ValueError, match="cumulative protocol time"):
        run_protocol(conversion(), {"A": 1, "B": 0}, [
            Phase("one", 1e308, SOURCE), Phase("two", 1e308, SOURCE)], max_step_s=1e308)


def test_empty_generator_provenance_rejected():
    with pytest.raises(ValueError, match="provenance"):
        Species("A", (item for item in []))
