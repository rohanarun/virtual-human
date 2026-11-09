"""Exposure experiment on the engine versus closed forms; no Asma dependency."""
from fractions import Fraction
import importlib.util
from math import exp, log
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exposure_script", ROOT / "scripts/validate_paraneoplastic_exposure.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)
NETWORK = script.build_network()


def test_exposure_identity_is_exact_for_both_hypotheses():
    for hypothesis in ("burden", "release"):
        for chemo in (0.0, 1.0):
            residuals = NETWORK.kinetic_residuals(script.invariant_weights(hypothesis),
                                                  rate_scales=script.scales(hypothesis, chemo=chemo),
                                                  exact_rates=script.RATES)
            assert all(r == 0 for r in residuals.values())
    # The pure release hypothesis is not burden-driven: its identity fails with presentation on.
    mixed = NETWORK.kinetic_residuals(script.invariant_weights("release"),
                                      rate_scales=script.scales("release", presentation=1.0),
                                      exact_rates=script.RATES)
    assert any(r != 0 for r in mixed.values())


def test_discrete_order_comparison_matches_exact_exposure():
    early, late = script.sweep(0.1, 0.5)[0], script.sweep(0.1, 0.5)[3]
    b0 = script.discrete_schedule(NETWORK, "burden", early)
    b3 = script.discrete_schedule(NETWORK, "burden", late)
    assert abs(b0["A"] - 5 / 32) < 1e-12 and abs(b3["A"] - 5 / 32) < 1e-12
    assert abs(b0["U"] - 635 / 32) < 1e-9 and abs(b3["U"] - 2875 / 32) < 1e-9
    predicted = 0.01 * (b3["U"] - b0["U"]) - 0.01 * (b3["E"] - b0["E"])
    assert abs((b3["H"] - b0["H"]) - predicted) < 1e-12
    r0 = script.discrete_schedule(NETWORK, "release", early)
    r3 = script.discrete_schedule(NETWORK, "release", late)
    assert abs(r3["R"] - r0["R"] - 78.75) < 1e-9
    assert abs(r0["N"] - exp(-r0["H"])) < 1e-11


def test_continuous_schedule_matches_closed_form():
    ops = script.sweep(0.75, 0.5)[2]
    for hypothesis in ("burden", "release"):
        state = script.continuous_schedule(NETWORK, hypothesis, ops)
        closed = script.continuous_closed_form(ops)
        assert abs(state["U"] - closed["exposure"]) / closed["exposure"] < 1e-9
        assert abs(state["R"] - closed["released"]) / closed["released"] < 1e-9
        assert abs(state["R"] - (100 - closed["final"] - closed["removed"])) < 1e-8
    # Continuous chemotherapy shifts the burden-driven threshold to q < 1/(1 + c).
    assert 0.5 < 1 / (1 + log(2)) < 0.75


def test_committed_injury_after_complete_removal():
    held = script.scales("burden")
    state = script.initial(burden=0.0, effector=3.0)
    out = NETWORK.simulate(state, [0, 10], max_step_s=script.STEP, rate_scales=held).final
    assert abs(out["H"] - 0.01 * 3.0 * (1 - exp(-10))) < 1e-10


def test_no_rise_from_steady_state_and_release_rise():
    for hypothesis, expect_rise in (("burden", False), ("release", True)):
        state = script.initial(effector=10.0)
        traj = NETWORK.simulate(state, [0, 0.25, 0.5, 0.75, 1.0], max_step_s=script.STEP,
                                rate_scales=script.scales(hypothesis, chemo=log(2), presentation=0.1))
        effector = traj.concentrations[:, script.SPECIES.index("E")]
        assert (effector[-1] > effector[0] + 1e-9) == expect_rise
        if not expect_rise:
            assert all(b <= a + 1e-12 for a, b in zip(effector, effector[1:]))
