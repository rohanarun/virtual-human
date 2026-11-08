"""Rate-dependent linear invariants: exact algebra, numerical drift and inputs."""
from fractions import Fraction
from math import exp

import numpy as np
import pytest

from vh.graph.reactions import Phase, Reaction, ReactionNetwork, Species, rate_unit, run_protocol

SOURCE = ("Artificial mathematical fixture; not a biological measurement",)


def reaction(name, left, right, rate):
    return Reaction(name, left, right, rate, rate_unit(sum(left.values())), SOURCE)


def network(ids, reactions):
    return ReactionNetwork([Species(s, SOURCE) for s in ids], reactions,
                           context={"system": "synthetic", "compartment": "constant-volume"},
                           provenance=SOURCE)


def span_rank(vectors, ids):
    return np.linalg.matrix_rank(np.array([[float(v.get(s, 0)) for s in ids] for v in vectors])) if vectors else 0


def clearance():
    # Bolus C eliminated at k; U accumulates AUC; X accumulates the amount eliminated.
    return network(["C", "X", "U"], [reaction("eliminate", {"C": 1}, {"X": 1}, 0.25),
                                     reaction("auc", {"C": 1}, {"C": 1, "U": 1}, 1.0)])


def test_stoichiometric_pools_are_kinetic_invariants_for_any_rates():
    for rates in ((0.3, 0.7), (2.0, 0.0), (0.125, 5.0)):
        model = network(["A", "B", "D"], [reaction("dimerize", {"A": 2}, {"D": 1}, rates[0]),
                                          reaction("convert", {"A": 1}, {"B": 1}, rates[1])])
        kinetic = model.kinetic_invariant_basis()
        for pool in model.conservation_basis():
            assert all(r == 0 for r in model.kinetic_residuals(pool).values())
        assert span_rank(kinetic + model.conservation_basis(), model.ids) == span_rank(kinetic, model.ids)


def test_distinct_monomials_add_no_invariant():
    model = network(["A", "B", "C"], [reaction("ab", {"A": 1}, {"B": 1}, 0.4),
                                      reaction("bc", {"B": 1}, {"C": 1}, 0.9)])
    assert span_rank(model.kinetic_invariant_basis(), model.ids) == span_rank(model.conservation_basis(), model.ids) == 1


def test_clearance_identity_found_exactly_and_holds_numerically():
    model = clearance()
    exact = {"eliminate": Fraction(1, 4), "auc": 1}
    basis = model.kinetic_invariant_basis(exact_rates=exact)
    identity = {"X": 1, "U": Fraction(-1, 4)}  # eliminated amount = k * AUC
    assert all(r == 0 for r in model.kinetic_residuals(identity, exact_rates=exact).values())
    assert span_rank(basis + (identity,), model.ids) == span_rank(basis, model.ids) == 2
    final = model.simulate({"C": 8.0, "X": 0.0, "U": 0.0}, [0, 3, 40], max_step_s=0.01).final
    assert abs(final["X"] - 0.25 * final["U"]) < 1e-12
    assert abs(final["U"] - 8.0 / 0.25 * (1 - exp(-0.25 * 40))) < 1e-8  # dose / k


def test_exposure_identity_and_rate_scale_dependence():
    model = network(["A", "E", "H", "U"], [
        reaction("present", {"A": 1}, {"A": 1, "E": 1}, 0.5),
        reaction("exposure", {"A": 1}, {"A": 1, "U": 1}, 1.0),
        reaction("decay", {"E": 1}, {}, 0.25),
        reaction("hazard", {"E": 1}, {"E": 1, "H": 1}, 0.01)])
    exact = {"present": Fraction(1, 2), "decay": Fraction(1, 4), "hazard": Fraction(1, 100)}
    identity = {"H": 1, "E": Fraction(1, 25), "U": Fraction(-1, 50)}
    assert all(r == 0 for r in model.kinetic_residuals(identity, exact_rates=exact).values())
    # Doubling presentation changes the rate-dependent coefficient, not the pool A.
    doubled = model.kinetic_residuals(identity, rate_scales={"present": 2}, exact_rates=exact)
    assert doubled["A"] == Fraction(1, 25) * Fraction(1, 2)
    fixed = {"H": 1, "E": Fraction(1, 25), "U": Fraction(-1, 25)}
    assert all(r == 0 for r in model.kinetic_residuals(fixed, rate_scales={"present": 2}, exact_rates=exact).values())
    # Retaining A is an external event that leaves the weighted sum unchanged.
    phases = [Phase("hold", 3, SOURCE), Phase("cut", 4, SOURCE, retain={"A": 0.1})]
    out = run_protocol(model, {"A": 10.0, "E": 1.0, "H": 0.0, "U": 0.0}, phases, max_step_s=0.01)
    value = lambda s: s["H"] + s["E"] / 25 - s["U"] / 50
    assert abs(value(out[-1].trajectory.final) - value({"H": 0, "E": 1.0, "U": 0})) < 1e-13


def test_float_rates_are_converted_exactly():
    model = network(["A", "U", "X"], [reaction("loss", {"A": 1}, {"X": 1}, 0.1),
                                      reaction("auc", {"A": 1}, {"A": 1, "U": 1}, 1.0)])
    assert all(r == 0 for r in model.kinetic_residuals({"X": 1, "U": -Fraction(0.1)}).values())
    assert model.kinetic_residuals({"X": 1, "U": Fraction(-1, 10)})["A"] != 0
    exact = {"loss": Fraction(1, 10)}
    assert all(r == 0 for r in model.kinetic_residuals({"X": 1, "U": Fraction(-1, 10)}, exact_rates=exact).values())


@pytest.mark.parametrize("exact", [{"missing": 1}, {"eliminate": Fraction(1, 3)}, {"eliminate": True},
                                   {"eliminate": -Fraction(1, 4)}, {"eliminate": 0.25}])
def test_invalid_exact_rates_rejected(exact):
    with pytest.raises(ValueError):
        clearance().kinetic_invariant_basis(exact_rates=exact)


def test_invalid_weights_and_scales_rejected():
    model = clearance()
    for weights in ({}, {"Z": 1}, {"X": 0.5}, {"X": True}):
        with pytest.raises(ValueError):
            model.kinetic_residuals(weights)
    with pytest.raises(ValueError):
        model.kinetic_invariant_basis(rate_scales={"unknown": 1})
    with pytest.raises(ValueError):
        model.kinetic_invariant_basis(rate_scales={"auc": -1})
