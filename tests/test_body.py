"""Whole-body engine tests: conservation, exact analytic limits, real-time API, bridge."""

import numpy as np
import pytest

from vh.body import MIDAZOLAM, Dose, Ionization, PBPKModel, RealTimeSimulator, reference_individual
from vh.body.calibration import clearance_overrides
from vh.body.compound import Compound
from vh.body.partition import kpu_all, tissue_data
from vh.body.physiology import TISSUES, make_individual, sample_population
from vh.bridge import binding_free_energy_from_kd, kd_from_binding_free_energy, kinetic_occupancy


@pytest.fixture(scope="module")
def mdz():
    return PBPKModel(MIDAZOLAM, reference_individual("male"))


def test_blood_flows_balance():
    ind = reference_individual("male")
    systemic = sum(ind.flows[t] for t in TISSUES if t not in ("lung", "liver", "gut", "spleen", "pancreas"))
    assert systemic + ind.liver_flow_total == pytest.approx(ind.cardiac_output, rel=1e-12)


def test_mass_is_conserved(mdz):
    sim = mdz.simulate([Dose(0, 5.0, "iv", 0.1), Dose(2.0, 10.0, "oral")], t_end_h=48, dt_h=0.1)
    total = sim["state"].sum(axis=1)
    assert total[-1] == pytest.approx(15.0, rel=1e-10)
    assert np.all(sim["state"] >= -1e-12)


def test_iv_auc_equals_dose_over_clearance(mdz):
    """For a linear model AUC(0,inf) = Dose / CL exactly; checks the whole matrix assembly."""
    dose = 3.65
    sim = mdz.simulate([Dose(0, dose, "iv")], t_end_h=200, dt_h=0.005)
    auc = np.trapezoid(sim["plasma"], sim["time_h"])
    cl = mdz.p.cl_total_plasma_l_h
    # renal elimination is taken from kidney tissue, so allow a tiny difference
    assert auc == pytest.approx(dose / cl, rel=0.01)


def test_clearance_calibration_matches_integrated_compartment_model(mdz):
    systemic, apparent_oral = 22.2, 84.78  # L/h, feasible reference targets
    overrides = clearance_overrides(mdz, systemic, apparent_oral)
    model = PBPKModel(MIDAZOLAM, mdz.ind, **overrides)
    # Integrate the transient state to infinity directly from the matrix,
    # independent of the clearance algebra used by the calibration.
    names = [name for name in model.names if not name.startswith("elim_") and name != "feces"]
    transient = [model.ix[name] for name in names]
    A = model.M[np.ix_(transient, transient)]
    for site, clearance in (("venous", systemic), ("stomach", apparent_oral)):
        unit_dose = np.zeros(len(transient))
        unit_dose[names.index(site)] = 1.0
        integral = np.linalg.solve(-A, unit_dose)
        auc = integral[names.index("venous")] / (model.V["venous"] * model.p.bp)
        assert 1.0 / auc == pytest.approx(clearance, rel=1e-12)


def test_steady_state_infusion(mdz):
    rate = 1.0  # mg/h
    sim = mdz.simulate([Dose(0, rate * 100, "iv", 100.0)], t_end_h=99, dt_h=1.0)
    css = sim["plasma"][-1]
    assert css == pytest.approx(rate / mdz.p.cl_total_plasma_l_h, rel=0.01)


def test_oral_bioavailability_matches_components(mdz):
    sim = mdz.simulate([Dose(0, 10.0, "oral")], t_end_h=200, dt_h=0.01)
    sim_iv = mdz.simulate([Dose(0, 10.0, "iv")], t_end_h=200, dt_h=0.01)
    f_sim = np.trapezoid(sim["plasma"], sim["time_h"]) / np.trapezoid(sim_iv["plasma"], sim_iv["time_h"])
    p = mdz.p
    f_hep = 1.0 - p.cl_hep_blood_l_h / mdz.ind.liver_flow_total
    assert f_sim == pytest.approx(p.fa * p.fg * f_hep, rel=0.02)


def test_realtime_matches_batch(mdz):
    rt = RealTimeSimulator(mdz)
    rt.dose(5.0, "iv", duration_h=0.25)
    obs = None
    for _ in range(40):          # irregular steps, total 3 h
        obs = rt.advance(0.075)
    batch = mdz.simulate([Dose(0, 5.0, "iv", 0.25)], t_end_h=3.0, dt_h=0.075)
    assert obs["plasma"] == pytest.approx(batch["plasma"][-1], rel=1e-9)


def test_neutral_hydrophilic_kpu_is_water_plus_protein_terms():
    c = Compound("hydrophilic", 100.0, -5.0, Ionization("neutral"), fu=1.0)
    kpu = kpu_all(c.logp, c.ionization, c.fu, 1.0, 0.45)
    d = tissue_data()["tissues"]["muscle"]
    expected = d["f_ew"] + d["f_iw"] + 0.7 * d["f_np"]    # P -> 0, no protein binding at fu = 1
    assert kpu["muscle"] == pytest.approx(expected, rel=1e-3)


def test_population_variability_spreads_exposure():
    pop = sample_population(20, seed=1)
    aucs = []
    for ind in pop:
        m = PBPKModel(MIDAZOLAM, ind)
        s = m.simulate([Dose(0, 2.0, "iv", 0.05)], t_end_h=24, dt_h=0.1)
        aucs.append(np.trapezoid(s["plasma"], s["time_h"]))
    cv = np.std(aucs) / np.mean(aucs)
    assert 0.1 < cv < 1.0


def test_obese_individual_has_larger_volume():
    lean = PBPKModel(MIDAZOLAM, make_individual("female", 37, 61))
    obese = PBPKModel(MIDAZOLAM, make_individual("female", 36, 136))
    assert obese.p.vss_l > 1.5 * lean.p.vss_l


def test_bridge_kd_free_energy_roundtrip_and_kinetics():
    dg = binding_free_energy_from_kd(1.8)
    assert kd_from_binding_free_energy(dg) == pytest.approx(1.8, rel=1e-12)
    t = np.linspace(0, 1, 101)
    occ = kinetic_occupancy(t, np.full_like(t, 1.8), 1.8, koff_per_h=60.0)
    assert occ[-1] == pytest.approx(0.5, abs=1e-6)
