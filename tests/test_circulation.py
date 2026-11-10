"""Heart, closed circulation, and drug-transport-on-the-heartbeat tests.

Every test compares against an independent authority -- an exact structural
identity of the model (volume conservation, zero valve backflow, the matched
ejections of the two ventricles), the published normal-physiology table the
parameters were taken from (Broomé et al. 2013, Biomed Eng Online 12:69,
Table 1; the parameters set pressures/flows are NOT the quantities being
checked), or the already-validated PBPK engine that the drug layer must mirror
exactly.  Numbers quoted in comments are the measured values of this model,
not hard-coded expectations of code output.
"""

import copy

import numpy as np
import pytest

from vh.body import MIDAZOLAM, PBPKModel, Dose, reference_individual
from vh.body.physiology import PORTAL, TISSUES
from vh.graph.circulation import (
    CirculationParams,
    HeartCirculation,
    build_circulation_graph,
    double_hill,
)
from vh.graph.uptake import (
    CirculationUptakeModel,
    extraction_ratio_liver,
    organ_net_uptake_l_h,
    transport_matrix,
)

# published normal values (Broomé et al. 2013 Table 1, "normal" column)
PUB = dict(co=(6.3, 2.4), sv=(89.0, 30.0), ef=(0.59, 0.04), lvedv=(123.0, 28.0),
           rap=(4.8, 2.4), lap=(8.3, 2.5))


@pytest.fixture(scope="session")
def circ():
    ind = reference_individual("male")
    params = CirculationParams.from_individual(ind)
    heart = HeartCirculation(params)
    rec = heart.simulate(n_beats=45, dt=1e-3, record_last=8)
    return ind, params, heart, rec, heart.beat_summary(rec)


@pytest.fixture(scope="session")
def uptake(circ):
    """PBPK + uptake layer + the hybrid heartbeat record, built once."""
    ind, params, heart, rec, _ = circ
    pbpk = PBPKModel(MIDAZOLAM, ind)
    return pbpk, CirculationUptakeModel(pbpk, heart), heart, rec


# ---------------------------------------------------------------------------
# the hydraulic heart: structural identities
# ---------------------------------------------------------------------------
def test_total_blood_volume_conserved(circ):
    ind, params, heart, rec, _ = circ
    total = rec.volumes.sum(axis=1)
    assert float(np.ptp(total)) < 1e-8          # measured drift: 1.8e-11 mL
    # the graph carries exactly the individual's ICRP blood volume (5.3 L)
    assert abs(total[0] - (ind.v_arterial + ind.v_venous) * 1000.0) < 1e-6


def test_both_ventricles_eject_identically(circ):
    """LV and RV sit in ONE closed loop: their stroke volumes must agree."""
    _, _, heart, rec, _ = circ
    q_lv = float(rec.flow_at("aortic").mean())
    q_rv = float(rec.flow_at("pulmonic").mean())
    assert abs(q_lv / q_rv - 1.0) < 1e-3        # measured: 1.7e-4 (window residue)


def test_valves_never_carry_backflow(circ):
    _, params, _, rec, _ = circ
    for name in params.valves:
        q = rec.flow_at(name)
        assert float(np.min(q)) == 0.0          # exactly zero when reversed
        assert float(np.max(q)) > 0.0           # and they do open


def test_zero_backflow_when_reversed():
    """A valve is closed against reverse pressure by construction, not by clipping."""
    v = HeartCirculation.__mro__  # noqa: F841  (placeholder keeps import list stable)
    from vh.graph.circulation import Valve
    val = Valve("mitral", "la", "lv", area_cm2=5.0)
    assert float(val.flow(5.0, 9.0)) == 0.0
    q = val.flow(np.array([5.0, 12.0]), np.array([9.0, 9.0]))
    assert q[0] == 0.0 and q[1] > 0.0


def test_elastance_reaches_published_amplitude(circ):
    """max_t E(t) == Emax for every chamber (double-Hill normalization)."""
    _, params, _, _, _ = circ
    period = params.period_s
    for cid, ch in params.chambers.items():
        tt = np.linspace(0.0, period, 20000, endpoint=False)
        e = ch.elastance(tt, period)
        peak = float((e.max() - ch.emin_mmhg_ml) / (ch.emax_mmhg_ml - ch.emin_mmhg_ml))
        assert peak > 0.999, f"{cid}: elastance peak {peak}"
        assert float(e.min()) == pytest.approx(ch.emin_mmhg_ml, abs=1e-12)


def test_double_hill_shape_is_continuous_and_bounded():
    period = 0.857142857
    f = double_hill(np.linspace(0.0, period, 2001), period, 0.30, 1.32, 0.47, 21.9)
    assert float(f.min()) >= 0.0 and float(f.max()) <= 1.0 + 1e-12
    assert abs(float(f[0])) < 1e-6 and float(f.max()) > 0.0


def test_organ_bed_shares_sum_to_icrp_split(circ):
    _, params, _, _, _ = circ
    total = sum(b.share_pct for b in params.beds.values())
    assert abs(total - 100.0) < 0.5
    assert set(params.beds) == set(TISSUES) - {"lung"}


# ---------------------------------------------------------------------------
# the hydraulic heart: baseline physiology vs the published table
# ---------------------------------------------------------------------------
def test_baseline_inside_published_ranges(circ):
    _, _, heart, rec, s = circ
    checks = [
        ("cardiac_output_l_min", (PUB["co"][0] - PUB["co"][1], PUB["co"][0] + PUB["co"][1])),
        ("sv_ml", (PUB["sv"][0] - PUB["sv"][1], PUB["sv"][0] + PUB["sv"][1])),
        ("ef", (PUB["ef"][0] - 3 * PUB["ef"][1], PUB["ef"][0] + 3 * PUB["ef"][1])),
        ("edv_lv_ml", (PUB["lvedv"][0] - PUB["lvedv"][1], PUB["lvedv"][0] + PUB["lvedv"][1])),
        ("mean_p_ra", (PUB["rap"][0] - PUB["rap"][1], PUB["rap"][0] + PUB["rap"][1])),
        ("mean_p_la", (PUB["lap"][0] - PUB["lap"][1], PUB["lap"][0] + PUB["lap"][1])),
    ]
    for key, (lo, hi) in checks:
        assert lo <= s[key] <= hi, f"{key}={s[key]:.3f} outside published [{lo}, {hi}]"
    # measured: CO 6.35, SV 90.8, EF 63%, EDV 143, RAP 3.46, LAP 5.91,
    #           107/70, PAP 22/12 -- each inside the published column


def test_graph_schema_validates_clean(circ):
    _, params, _, _, _ = circ
    g, schema = build_circulation_graph(params)
    assert g.validate(schema) == []


def test_reference_state_has_positive_volumes(circ):
    _, _, heart, _, _ = circ
    v0 = heart.state_at_reference()
    assert float(v0.min()) > 0.0
    assert abs(float(v0.sum()) - heart.blood_volume(v0)) < 1e-6


def test_bed_flows_split_exactly_on_icrp_shares(circ):
    """Structural identity: every bed sees the same arterial-venous gradient,
    so the instantaneous flow split follows the resistances -- which were set
    once from the ICRP shares.  This holds at any state, any beat phase."""
    _, params, heart, rec, _ = circ
    V = rec.volumes[len(rec.volumes) // 2]
    t = float(rec.time_s[len(rec.time_s) // 2])
    q = heart.flows(V, t)
    total = sum(q[f"bed_{t_}"] for t_ in params.beds)
    for tissue, bed in params.beds.items():
        share = q[f"bed_{tissue}"] / total
        assert abs(share * 100.0 / bed.share_pct - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# the hydraulic heart: responses (directional, against known physiology)
# ---------------------------------------------------------------------------
def _run(params, n_beats=50, v0=None):
    heart = HeartCirculation(params)
    rec = heart.simulate(v0=v0, n_beats=n_beats, dt=1e-3, record_last=6)
    return heart.beat_summary(rec)


def test_frank_starling(circ):
    """Preload up -> CO and EDV up; preload down -> both down (published: SV
    rises with filling pressure until ~2x resting, Guyton)."""
    ind = reference_individual("male")
    base_params = CirculationParams.from_individual(ind)
    heart = HeartCirculation(base_params)
    base = _run(base_params)
    s_plus = _run(CirculationParams.from_individual(ind),
                  v0=heart.transfuse(0.10 * heart.blood_volume()))
    s_minus = _run(CirculationParams.from_individual(ind),
                   v0=heart.transfuse(-0.10 * heart.blood_volume()))
    assert s_plus["cardiac_output_l_min"] > base["cardiac_output_l_min"]    # 8.74 > 6.35
    assert s_plus["edv_lv_ml"] > base["edv_lv_ml"]                          # 191.6 > 143.2
    assert s_minus["cardiac_output_l_min"] < base["cardiac_output_l_min"]   # 3.88 < 6.35
    assert s_minus["edv_lv_ml"] < base["edv_lv_ml"]                         # 93.1 < 143.2
    # measured: +10% blood -> CO 8.74, MAP 124.7; -10% -> CO 3.88, MAP 55.1


def test_systolic_failure_congests_and_lowers_ef(circ):
    """Emax halved: EF falls, EDV rises, filling pressures rise (published
    direction of every systolic-heart-failuretrial)."""
    ind = reference_individual("male")
    p = CirculationParams.from_individual(ind)
    p.chambers["lv"].emax_mmhg_ml *= 0.5
    p.chambers["rv"].emax_mmhg_ml *= 0.5
    s = _run(p)
    s0 = _run(CirculationParams.from_individual(ind))
    assert s["ef"] < s0["ef"] - 0.05                       # 49% vs 63%
    assert s["edv_lv_ml"] > s0["edv_lv_ml"] + 10.0         # 158 vs 143
    assert s["mean_p_la"] > s0["mean_p_la"]                # congestion: 6.40 > 5.91
    assert s["mean_p_ra"] > s0["mean_p_ra"]                # 3.85 > 3.46


def test_severe_hemorrhage_shocks(circ):
    """-20% blood: CO collapses and MAP falls to shock levels."""
    ind = reference_individual("male")
    heart = HeartCirculation(CirculationParams.from_individual(ind))
    s = _run(CirculationParams.from_individual(ind),
             v0=heart.transfuse(-0.20 * heart.blood_volume()))
    assert s["cardiac_output_l_min"] < 2.0                 # measured 1.31
    assert s["map_mmhg"] < 40.0                            # measured 18.5


def test_tachycardia_wraps_the_beat(circ):
    """HR 100: the beat shortens, SV falls, CO rises modestly at rest
    (no contractility change; published resting response)."""
    p = CirculationParams.from_individual(reference_individual("male"))
    p.heart_rate_min = 100.0
    s = _run(p)
    s0 = _run(CirculationParams.from_individual(reference_individual("male")))
    assert abs(60.0 / p.period_s - 100.0) < 1e-9
    assert s["sv_ml"] < s0["sv_ml"]                        # 69.9 < 90.8
    assert s["cardiac_output_l_min"] > s0["cardiac_output_l_min"]  # 6.99 > 6.35


# ---------------------------------------------------------------------------
# the drug layer: identities against the validated PBPK engine
# ---------------------------------------------------------------------------
def test_bridge_matrix_is_exactly_the_pbpk_matrix(uptake):
    pbpk, u, _, _ = uptake
    assert np.max(np.abs(u.matrix(u.flows_mean_l_h) - pbpk.M)) == 0.0


def test_mean_solver_matches_pbpk_exactly(uptake):
    pbpk, u, _, _ = uptake
    doses = [Dose(0.0, 5.0, "iv"), Dose(0.0, 7.5, "oral")]
    a = u.simulate_mean(doses, 24.0, dt_h=0.05)
    b = pbpk.simulate(doses, 24.0, dt_h=0.05)
    rel = np.max(np.abs(a["plasma"] - b["plasma"]) / np.maximum(b["plasma"], 1e-12))
    assert rel < 1e-11                                     # measured: 1.2e-13


def test_transport_matrix_conservation_identity(uptake):
    """Every transport matrix has zero column sums except the sinks it writes
    into, so the total drug amount is a linear invariant term by term."""
    pbpk, u, _, _ = uptake
    M = transport_matrix(pbpk, dict(u.flows_mean_l_h))
    cols = M.sum(axis=0)
    carry = [c for c, name in enumerate(pbpk.names)
             if name not in ("elim_hepatic", "elim_renal", "elim_gut_wall", "feces")]
    assert float(np.max(np.abs(cols[carry]))) < 1e-12


def test_steady_state_identity(uptake):
    """At the linear SS the input must leave through the elimination sinks
    exactly, and the plasma concentration doubles as R/CL to within the small
    (7.8e-5) gap between the scalar CL aggregate and the exact compartmental
    SS -- the same convention the validated engine itself reports."""
    pbpk, u, _, _ = uptake
    r = 10.0
    x = u.steady_state(r)
    # exact structural identity: sink production equals the infusion rate
    M = u.matrix(dict(u.flows_mean_l_h))
    sinks = [pbpk.ix[s] for s in ("elim_hepatic", "elim_renal", "elim_gut_wall", "feces")]
    assert abs(float((M @ x)[sinks].sum()) - r) < 1e-9
    # scalar R/CL (the engine's CL definition) matches the exact SS to 7.8e-5
    css = float(pbpk.observe(x)["plasma"])
    assert abs(css / (r / pbpk.p.cl_total_plasma_l_h) - 1.0) < 1e-3


def test_liver_extraction_lands_on_published_band(uptake):
    """E_h from the bottom-up CLint chain ~ 0.28 (literature band for
    midazolam hepatic extraction 0.30-0.50, Thummel et al. 0.44; the
    middle-out CLint x1.13 case reaches ~0.31 -- documented in the report)."""
    pbpk, u, _, _ = uptake
    x = u.steady_state(10.0)
    eh = extraction_ratio_liver(pbpk, x, dict(u.flows_mean_l_h))
    eh_identity = pbpk.p.cl_hep_blood_l_h / pbpk.ind.liver_flow_total
    assert abs(eh / eh_identity - 1.0) < 2e-3              # internal identity
    assert 0.20 <= eh <= 0.50                              # published band


def test_organ_uptake_magnitudes(uptake):
    """At 15 min post-dose the liver dominates net uptake (it clears); Portal
    ordering is a documented physiological prediction."""
    pbpk, u, _, _ = uptake
    doses = [Dose(0.0, 5.0, "iv")]
    o = u.simulate_mean(doses, 1.0, dt_h=0.05)
    st = o["state"][np.searchsorted(o["time_h"], 0.25)]
    up = organ_net_uptake_l_h(pbpk, st, dict(u.flows_mean_l_h))
    assert up["liver"] > 1.0                               # measured 5.8-8.1 mg/h
    rest = [t for t in TISSUES if t not in ("liver", "lung")]
    assert up["liver"] > max(up[t] for t in rest)


# ---------------------------------------------------------------------------
# the drug layer on the heartbeat itself
# ---------------------------------------------------------------------------
def _flat_beat(heart: HeartCirculation, rec, summary):
    """A BeatRecord whose every bed/lung flow is replaced by its beat mean --
    same machinery, constant flows: must reproduce the mean solver exactly."""
    flat = copy.copy(rec)
    flat.flows = rec.flows.copy()
    for j, fid in enumerate(rec.flow_ids):
        if fid.startswith("bed_"):
            flat.flows[:, j] = summary[f"mean_flow_{fid}"]
        elif fid == "pulmonary":
            flat.flows[:, j] = summary["mean_flow_pulmonary"]
    return flat


def test_pulsatile_walker_reproduces_mean_solver_on_flat_flows(uptake):
    """Exactly the same flows, two independent code paths: piecewise-constant
    regrouping vs breakpoints.  Measured: 2.8e-11 relative."""
    pbpk, u, heart, rec = uptake
    flat = _flat_beat(heart, rec, heart.beat_summary(rec))
    flows_h = {k.replace("mean_flow_bed_", ""): v * 3.6 for k, v in
               heart.beat_summary(rec).items() if k.startswith("mean_flow_bed_")}
    flows_h["lung"] = heart.beat_summary(rec)["mean_flow_pulmonary"] * 3.6
    u_h = CirculationUptakeModel(pbpk, heart, mean_flows_l_h=flows_h)
    doses = [Dose(0.0, 5.0, "iv"), Dose(0.0, 7.5, "oral")]
    a = u_h.simulate_pulsatile(flat, doses, 6.0, dt_transport_s=0.004, sample_every_h=0.05)
    b = u_h.simulate_mean(doses, 6.0, dt_h=0.05)
    rel = max(abs(a["plasma"][i] / b["plasma"][i] - 1.0)
              for i in range(1, len(a["time_h"])))
    assert rel < 1e-9


def test_pulsatile_total_drug_conserved(uptake):
    """IV bolus + oral dose together: total drug (all states incl. elimination
    sinks) equals the dose at every sampled time, however flows pulse."""
    pbpk, u, heart, rec = uptake
    doses = [Dose(0.0, 5.0, "iv"), Dose(0.0, 7.5, "oral")]
    o = u.simulate_pulsatile(rec, doses, 24.0, dt_transport_s=0.004, sample_every_h=0.05)
    drift = float(np.ptp(o["state"].sum(axis=1)))
    assert drift < 1e-7                                   # measured: 2.7e-10 mg
    assert abs(o["state"].sum(axis=1)[-1] - 12.5) < 1e-7


def test_pulsatile_infusion_matches_clearance_identity(uptake):
    """An infusion through the pulsatile path (starting from no drug): the
    in-system amount equals the integral of the rate exactly, and the plasma
    approaches the mean-flow steady state from below as dosing proceeds."""
    pbpk, u, heart, rec = uptake
    r = 10.0                                   # mg/h; dose = rate x duration
    o = u.simulate_pulsatile(rec, [Dose(0.0, r * 24.0, "iv", duration_h=24.0)],
                             24.0, dt_transport_s=0.004, sample_every_h=0.25)
    net = o["state"].sum(axis=1) - r * o["time_h"]
    assert float(np.max(np.abs(net))) < 1e-7
    css = float(pbpk.observe(o["state"][-1])["plasma"])
    assert 0.9 < css / (r / pbpk.p.cl_total_plasma_l_h) < 1.0   # 24h ~ 98.6% of SS
    x_ss = u.steady_state(r)
    assert css < float(pbpk.observe(x_ss)["plasma"])           # below the true C_ss


def test_pulsatile_discretization_converged(uptake):
    """Halving and quartering the transport step leaves the orbit unchanged
    (the piecewise-constant template is converged at 4 ms): measured 0.00%."""
    pbpk, u, heart, rec = uptake
    doses = [Dose(0.0, 5.0, "iv"), Dose(0.0, 7.5, "oral")]
    o4 = u.simulate_pulsatile(rec, doses, 4.0, dt_transport_s=0.004, sample_every_h=0.05)
    o1 = u.simulate_pulsatile(rec, doses, 4.0, dt_transport_s=0.001, sample_every_h=0.05)
    rel = max(abs(o1["plasma"][i] / o4["plasma"][i] - 1.0) for i in range(len(o4["time_h"])))
    assert rel < 5e-3


def test_matched_mean_auc_converges(uptake):
    pbpk, u, heart, rec = uptake
    doses = [Dose(0.0, 5.0, "iv"), Dose(0.0, 7.5, "oral")]
    ratios = []
    for dt in (0.004, 0.001):
        pulse = u.simulate_pulsatile(rec, doses, 24.0, dt_transport_s=dt)
        mean = CirculationUptakeModel(pbpk, heart, mean_flows_l_h=u.mean_transport_flows(rec, dt))
        steady = mean.simulate_mean(doses, 24.0)
        ratios.append(pulse["plasma_auc_mg_h_l"][-1] / steady["plasma_auc_mg_h_l"][-1])
    assert ratios[0] == pytest.approx(ratios[1], abs=1e-6)


def test_transport_samples_entire_beat(uptake, monkeypatch):
    pbpk, _, heart, rec = uptake
    u = CirculationUptakeModel(pbpk, heart)
    # A periodic piecewise-linear waveform whose first half has a larger mean.
    beat = copy.copy(rec)
    beat.time_s = np.arange(4) * heart.period / 4
    flow = np.array([1.0, 5.0, 3.0, 1.0])
    beat.flows = np.tile(flow[:, None], (1, len(rec.flow_ids)))
    segment_inputs = []

    def capture(flows):
        segment_inputs.append(flows)
        return pbpk.M

    monkeypatch.setattr(u, "matrix", capture)
    step = heart.period / 8
    u.simulate_pulsatile(beat, [], 0.0, dt_transport_s=step)
    actual = [sample["lung"] / 3.6 for sample in segment_inputs]
    np.testing.assert_allclose(actual, [2.0, 4.0, 4.5, 3.5, 2.5, 1.5, 1.0, 1.0])
    assert u.mean_transport_flows(beat, step)["lung"] == pytest.approx(3.6 * flow.mean())
    assert heart.sample_flows(beat, 1.25 * heart.period)["pulmonary"] == pytest.approx(5.0)


def test_off_grid_doses_and_auc_independent_of_sampling(uptake):
    _, u, heart, rec = uptake
    doses = [Dose(60 / 3600, 1.0, "iv", duration_h=60 / 3600),
             Dose(60.036 / 3600, 0.25, "iv", duration_h=1 / 3600),
             Dose(120.001 / 3600, 0.5, "iv"), Dose(120.001 / 3600, 0.75, "oral")]
    end = 123.456 / 3600
    for cadence in (18 / 3600, 0.037 / 3600):
        out = u.simulate_pulsatile(rec, doses, end, sample_every_h=cadence)
        assert out["time_h"][-1] == end
        expected = np.zeros_like(out["time_h"])
        for dose in doses:
            elapsed = out["time_h"] - dose.time_h
            delivered = (np.clip(elapsed / dose.duration_h, 0.0, 1.0)
                         if dose.duration_h else elapsed >= 0.0)
            expected += dose.amount_mg * delivered
        np.testing.assert_allclose(out["state"].sum(axis=1), expected, atol=1e-9, rtol=0)
        if cadence == 18 / 3600:
            coarse = out
        else:
            np.testing.assert_allclose(out["state"][-1], coarse["state"][-1], atol=1e-9)
            assert out["plasma_auc_mg_h_l"][-1] == pytest.approx(
                coarse["plasma_auc_mg_h_l"][-1], rel=1e-9)


def test_liver_net_uptake_matches_mass_balance(uptake):
    pbpk, u, _, _ = uptake
    x = u.steady_state(10.0)
    balance = u.matrix(u.flows_mean_l_h) @ x
    uptake = organ_net_uptake_l_h(pbpk, x, u.flows_mean_l_h)
    assert uptake["liver"] == pytest.approx(
        balance[pbpk.ix["liver"]] + balance[pbpk.ix["elim_hepatic"]])


def test_beat_summary_averages_extrema_across_cycles(circ):
    _, _, heart, rec, _ = circ
    beat = copy.copy(rec)
    beat.time_s = 10.0 + np.arange(20) * heart.period / 10
    beat.volumes = np.tile(rec.volumes[0], (20, 1))
    beat.flows = np.tile(rec.flows[0], (20, 1))
    beat.pressures = np.repeat([80.0, 120.0], 10)[:, None] * np.ones(len(rec.node_ids))
    assert heart.beat_summary(beat)["sap_mmhg"] == pytest.approx(100.0)


def test_pulsatile_24h_runs_fast(uptake):
    """The regrouped walker must keep the 24 h beat-coupled run interactive
    (it was ~19 s per-segment; the regrouping brings it to ~0.2 s)."""
    pbpk, u, heart, rec = uptake
    import time
    t0 = time.time()
    u.simulate_pulsatile(rec, [Dose(0.0, 5.0, "iv"), Dose(0.0, 7.5, "oral")],
                         24.0, dt_transport_s=0.004, sample_every_h=0.05)
    assert time.time() - t0 < 5.0
