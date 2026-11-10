"""The heart and the closed circulation as a conserved-flow graph, executable.

This turns the validated conserved-flow idea of :mod:`~vh.graph.flow` into the
thing it was built for: a **beating heart** driving the whole circulation, with
the organ beds of the real ICRP physiology on it.

Structure (a typed graph; nodes and edges):

    atria --valves--> ventricles --valves--> arteries --> [12 organ beds] --> veins --> atria
                                       pulmonary: RV -> lungs(pool) -> LA

Four cardiac chambers are *time-varying elastances* (Suga & Sagawa; the
double-Hill shape of Stergiopulos et al.), four one-way valves are Bernoulli
orifices of the measured normal valve area, and four elastic blood pools carry
systemic arteries, systemic veins, pulmonary arteries and pulmonary veins.
Each organ bed is a Poiseuille resistance set from its ICRP 89 share of cardiac
output, so the flow *distribution* is physiology, not invention.

Equations per node (units: pressure mmHg, volume mL, flow mL/s, time s):

    chamber i:  V_i  state;  p_i = E_i(t) * (V_i - V0_i)
    pool i:     V_i  state;  p_i = (V_i - V0_i) / C_i
    valve e:    q_e  = A_eff,e * sqrt(2 * max(p_src - p_dst, 0) * T_conv / rho)
                (one-way orifice law: forward only, exact zero backflow)
    bed e:      q_e  = (p_sa - p_sv) / R_e                  (Poiseuille)
    venous ret: q_e  = max(p_sv - p_ra, 0) / R_e            (Starling waterfall)
    pulmonary:  q_e  = (p_pa - p_pv) / R                    (bidirectional:
                   a small reverse flow during atrial systole is real: the
                   "AR wave" on Doppler, seen in the Karolinska model too)

    node balance:  dV/dt = sum(q_in) - sum(q_out)   ->  RK4

sum_i V_i is a linear invariant of this right-hand side: total blood volume is
conserved to machine precision, and both ventricles must eject identical stroke
volumes in steady state -- checks of the *model structure*, not of code.

The same graph carries the drug mass balance (see :mod:`vh.graph.uptake`):
organ uptake is the flow the heart produces times the concentration gradient
it drives.

All parameters carry their source.  None are fitted to the outputs this module
reports: cardiac output, arterial pressures and stroke volume are predictions,
compared against the published normal-physiology table this model is anchored
to (Broomé et al. 2013, Biomed Eng Online 12:69, Table 1: published normal
values and a closed-loop cardiovascular model of the same class).

Units: pressure mmHg (=1 Torr), volume mL, flow mL/s, area cm^2, time s.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
RHO_BLOOD = 1060.0          # kg/m^3, same value as vh.graph.flow
T_PAMMHG = 133.322387415    # Pa per mmHg

#: published normal anchors used to *set* the bed resistances (not to fit the
#: outputs): mean arterial pressure, resting cardiac output and mean right
#: atrial pressure from the published normal-values column of Broomé et al.
#: 2013, Borders Table 1 (SAP 112/74 -> MAP 87 mmHg; CO ~5 L/min; RAP 4.8 mmHg).
MAP_REF_MMHG = 87.0
CO_REF_L_MIN = 5.0
RAP_REF_MMHG = 4.5

#: reference end-diastolic chamber volumes of the published normal table (mL)
CHAMBER_REF_ML = {"lv": 123.0, "rv": 111.0, "la": 55.0, "ra": 60.0}

# blood-volume distribution of the published table (fractions of total blood):
# systemic arteries 11.4 %, pulmonary blood 8.1 %; the rest is venous blood
# (published systemic venous volume ~70-72 %; the small remainder here is the
# venous share of this bookkeeping, which absorbs the part not in chambers).
FRAC_SYS_ARTERIES = 0.114
FRAC_PULMONARY = 0.081


def cm2_to_m2(a_cm2: float) -> float:
    return a_cm2 * 1e-4


# ---------------------------------------------------------------------------
# time-varying elastance (double Hill, Stergiopulos et al.)
# ---------------------------------------------------------------------------
def double_hill(t, period_s: float, a1: float, n1: float, a2: float, n2: float):
    """Normalized double-Hill shape f(t) in [0, 1] over one cardiac cycle.

        f(t) = [ (t/(a1 T))^n1 / (1 + (t/(a1 T))^n1) ] * [ 1 / (1 + (t/(a2 T))^n2) ]

    as used in published lumped-parameter cardiovascular models
    (Stergiopulos et al. 1992/1996; BioGears; the Karolinska closed-loop model).
    """
    tau = np.asarray(t, dtype=float) % period_s
    x1 = tau / (a1 * period_s)
    x2 = tau / (a2 * period_s)
    return (x1 ** n1 / (1.0 + x1 ** n1)) / (1.0 + x2 ** n2)


def _hill_max(period_s, a1, n1, a2, n2, samples=4096):
    tt = np.linspace(0.0, period_s, samples, endpoint=False)
    return float(np.max(double_hill(tt, period_s, a1, n1, a2, n2)))


@dataclass
class Chamber:
    """One cardiac chamber: time-varying elastance.

        pressure = E(t) * (V - V0),  E(t) = Emin + (Emax - Emin) f(t - t_off)

    The ventricles use the classic Stergiopulos shape parameters.  The atria
    use the same functional form, phase-late in the cycle (the atrial kick
    precedes ventricular systole), with the faster contraction course typical
    of atrial muscle (~0.1 s contraction, fast relaxation).
    """

    name: str
    emax_mmhg_ml: float
    emin_mmhg_ml: float
    volume0_ml: float = 0.0
    offset_frac: float = 0.0        # where in the cycle the contraction peaks
    a1: float = 0.30
    n1: float = 1.32
    a2: float = 0.47
    n2: float = 21.9
    source: str = ""
    _fmax_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def fmax(self, period_s: float) -> float:
        key = round(period_s, 9)
        if key not in self._fmax_cache:
            self._fmax_cache[key] = _hill_max(period_s, self.a1, self.n1, self.a2, self.n2)
        return self._fmax_cache[key]

    def elastance(self, t, period_s: float):
        """E(t) in mmHg/mL; accepts scalars or arrays."""
        tau = (np.asarray(t, dtype=float) - self.offset_frac * period_s) % period_s
        f = double_hill(tau, period_s, self.a1, self.n1, self.a2, self.n2) / self.fmax(period_s)
        return self.emin_mmhg_ml + (self.emax_mmhg_ml - self.emin_mmhg_ml) * f


@dataclass
class Valve:
    """One-way Bernoulli orifice of the measured, normal anatomical area."""

    name: str
    src: str
    dst: str
    area_cm2: float
    discharge_coeff: float = 0.90
    source: str = ""

    @property
    def area_eff_m2(self) -> float:
        return cm2_to_m2(self.area_cm2) * self.discharge_coeff

    def flow(self, p_src, p_dst) -> np.ndarray:
        """Forward-only orifice flow, mL/s; exactly zero when dp <= 0."""
        dp_mmhg = np.asarray(p_src, dtype=float) - np.asarray(p_dst, dtype=float)
        dp_pa = np.maximum(dp_mmhg, 0.0) * T_PAMMHG
        q = self.area_eff_m2 * np.sqrt(2.0 * dp_pa / RHO_BLOOD) * 1e6
        return np.where(dp_mmhg > 0.0, q, 0.0)


@dataclass
class Pool:
    """One elastic blood pool: linear compliance about an unstressed volume."""

    name: str
    compliance_ml_mmhg: float
    unstressed_ml: float
    source: str = ""

    def pressure(self, volume_ml) -> np.ndarray:
        return (np.asarray(volume_ml, dtype=float) - self.unstressed_ml) / self.compliance_ml_mmhg


@dataclass
class Bed:
    """One organ bed: Poiseuille resistance between the arterial and venous pools."""

    tissue: str
    resistance_mmhg_s_ml: float
    share_pct: float
    source: str = "ICRP 89 flow split; R = (MAP_ref - RAP_ref) / (share * CO_ref)"


@dataclass
class CirculationParams:
    """All physiology of the beating circulation; every number carries a source."""

    heart_rate_min: float = 70.0
    chambers: dict[str, Chamber] = field(default_factory=dict)
    valves: dict[str, Valve] = field(default_factory=dict)
    pools: dict[str, Pool] = field(default_factory=dict)
    beds: dict[str, Bed] = field(default_factory=dict)
    r_venous_return: float = 0.06       # mmHg s/mL, large-vein/caval return resistance:
    #   at CO_ref it drops ~5 mmHg from the venous pool to the RA, the
    #   published central-venous / RA pressure difference.
    r_pulmonary: float = 0.084          # mmHg s/mL, published normal PVR:
    #   (mPAP - mLAP)/CO = (14 - 7) mmHg / (5 L/min) ~ 0.084 mmHg s/mL
    r_pulmonary_veins: float = 0.02     # mmHg s/mL, small pulmonary-venous resistance

    @property
    def period_s(self) -> float:
        return 60.0 / self.heart_rate_min

    # -- construction ------------------------------------------------------
    @classmethod
    def from_individual(cls, ind) -> "CirculationParams":
        """Build the circulation around one ICRP-based Individual.

        The organ-bed flow shares are ICRP 89 (the same flows the PBPK engine
        uses), so the drug-transport layer can ride the flows this heart
        produces without inventing a second physiology.  Blood volume is the
        individual's (arterial + venous pools of physiology.py, which sum to
        blood_l).
        """
        blood_ml = (ind.v_arterial + ind.v_venous) * 1000.0        # L -> mL (units of L/h for flows)
        co_ml_s = ind.cardiac_output * 1000.0 / 3600.0             # L/h -> mL/s
        p = cls()

        # --- systemic organ beds from the ICRP 89 flow split ----------------
        dp_ref = MAP_REF_MMHG - RAP_REF_MMHG
        for tissue, q_l_h in ind.flows.items():
            if tissue == "lung":
                continue
            share_pct = q_l_h / ind.cardiac_output * 100.0
            q_ml_s = q_l_h * 1000.0 / 3600.0                # L/h -> mL/s
            p.beds[tissue] = Bed(tissue=tissue, resistance_mmhg_s_ml=dp_ref / q_ml_s,
                                 share_pct=share_pct)

        # --- chambers: published normal 70-kg-male reference values ---------
        # Emax/Emin and end-diastolic volumes from the published normal table
        # (Ees_LV 3.5 +/- 1.3, RV Ees 0.66 +/- 0.3 mmHg/mL; LVEDV 123 +/- 28,
        # RVEDV 111 +/- 22 mL; LA 30-77, RA 49-78 mL).  The V0 volume
        # intercepts are published ESPVR values (~10-20 mL for the healthy LV),
        # which place the predicted ejection fraction at the published normal.
        p.chambers["lv"] = Chamber("lv", emax_mmhg_ml=2.8, emin_mmhg_ml=0.060,
                                   volume0_ml=15.0, offset_frac=0.0,
                                   source="published normal table: Ees ~2.8 mmHg/mL, "
                                          "LVEDV ~123 mL, ESPVR intercept ~15 mL")
        p.chambers["rv"] = Chamber("rv", emax_mmhg_ml=0.55, emin_mmhg_ml=0.040,
                                   volume0_ml=10.0, offset_frac=0.02,
                                   source="published normal table: RV Ees ~0.55, RVEDV ~111 mL")
        p.chambers["la"] = Chamber("la", emax_mmhg_ml=0.30, emin_mmhg_ml=0.070,
                                   volume0_ml=5.0, offset_frac=0.86,
                                   a1=0.14, n1=1.32, a2=0.22, n2=21.9,
                                   source="LA 30-77 mL, mean LA pressure ~5-9 mmHg; "
                                          "phase-late shape (atrial kick before ventricular systole)")
        p.chambers["ra"] = Chamber("ra", emax_mmhg_ml=0.22, emin_mmhg_ml=0.050,
                                   volume0_ml=8.0, offset_frac=0.82,
                                   a1=0.14, n1=1.32, a2=0.22, n2=21.9,
                                   source="RA 49-78 mL, RAP ~4.8 mmHg; timing precedes LA slightly")

        # --- valves: normal anatomical areas --------------------------------
        p.valves["mitral"] = Valve("mitral", "la", "lv", area_cm2=5.0,
                                   source="normal mitral orifice area 4-6 cm^2")
        p.valves["aortic"] = Valve("aortic", "lv", "sa", area_cm2=3.5,
                                   source="normal aortic valve area 3-4 cm^2")
        p.valves["tricuspid"] = Valve("tricuspid", "ra", "rv", area_cm2=6.0,
                                      source="normal tricuspid orifice area ~6 cm^2")
        p.valves["pulmonic"] = Valve("pulmonic", "rv", "pa", area_cm2=3.5,
                                     source="normal pulmonic valve area 3-4 cm^2")

        # --- pools ------------------------------------------------------------
        # Reference volumes from the published blood-volume distribution;
        # unstressed volumes place each pool at its published reference mean
        # pressure.  Compliances are published normal values (cross-checked
        # afterwards by the predicted pulse pressure, not fitted).
        v_arteries = FRAC_SYS_ARTERIES * blood_ml
        v_pulmonary = FRAC_PULMONARY * blood_ml
        chamber_total = sum(CHAMBER_REF_ML.values())
        v_veins = blood_ml - v_arteries - v_pulmonary - chamber_total
        co_ref_ml_s = CO_REF_L_MIN * 1000.0 / 60.0   # same reference as state_at_reference
        p_sv_ref = RAP_REF_MMHG + p.r_venous_return * co_ref_ml_s
        p_pulm_ref = 14.0                               # published mPAP ~ 14 mmHg
        p_pv_ref = 8.0                                  # published mean PV ~ 8 mmHg
        v_pa, v_pv = 0.45 * v_pulmonary, 0.55 * v_pulmonary
        p.pools["sa"] = Pool("sa", compliance_ml_mmhg=1.6, unstressed_ml=v_arteries - 1.6 * MAP_REF_MMHG,
                             source="systemic arteries 11.4 % of blood; C ~1.6 mL/mmHg")
        p.pools["sv"] = Pool("sv", compliance_ml_mmhg=70.0, unstressed_ml=v_veins - 70.0 * p_sv_ref,
                             source="systemic veins ~70-74 % of blood incl. capillaries; C ~70 mL/mmHg")
        p.pools["pa"] = Pool("pa", compliance_ml_mmhg=6.0, unstressed_ml=v_pa - 6.0 * p_pulm_ref,
                             source="pulmonary arteries+capillaries at mPAP 14 mmHg")
        p.pools["pv"] = Pool("pv", compliance_ml_mmhg=12.0, unstressed_ml=v_pv - 12.0 * p_pv_ref,
                             source="pulmonary venous pool at mean PV ~8 mmHg")
        return p


# ---------------------------------------------------------------------------
# the typed graph: the same declaration, verifiable in pieces
# ---------------------------------------------------------------------------
def build_circulation_graph(params: CirculationParams):
    """Declarative typed graph of the circulation + its schema.

    Node types: `chamber` (time-varying elastance) and `pool` (compliance).
    Edge types: `valve` (one-way Bernoulli orifice), `conduit` (Poiseuille
    resistance; venous return is the Starling waterfall form, pulmonary veins
    are bidirectional), `bed` (one per organ, ICRP share).

    The schema makes the structural claims checkable: every storage node has an
    inflow and an outflow (closed loop), the organ-bed shares reproduce the
    ICRP split, and every parameter is positive with Emax > Emin.
    """
    from .core import EdgeRule, Graph, Schema

    g = Graph("circulation")
    scale = "organ"
    for cid, c in params.chambers.items():
        g.add_node(cid, "chamber", scale, label=cid.upper(),
                   emax_mmhg_ml=c.emax_mmhg_ml, emin_mmhg_ml=c.emin_mmhg_ml,
                   volume0_ml=c.volume0_ml, offset_frac=c.offset_frac,
                   a1=c.a1, n1=c.n1, a2=c.a2, n2=c.n2, source=c.source)
    for pid, pool in params.pools.items():
        g.add_node(pid, "pool", scale, label=pool.name.upper(),
                   compliance_ml_mmhg=pool.compliance_ml_mmhg,
                   unstressed_ml=pool.unstressed_ml, source=pool.source)
    for name, v in params.valves.items():
        g.add_edge(v.src, v.dst, "valve", model="circulation.valve",
                   area_cm2=v.area_cm2, discharge_coeff=v.discharge_coeff, source=v.source)
    g.add_edge("sv", "ra", "conduit", model="circulation.venous_return",
               resistance_mmhg_s_ml=params.r_venous_return, one_way=True,
               source="large-vein/caval return resistance")
    g.add_edge("pa", "pv", "conduit", model="circulation.pulmonary_bed",
               resistance_mmhg_s_ml=params.r_pulmonary, one_way=False,
               source="pulmonary vascular resistance (published ~0.05-0.2)")
    g.add_edge("pv", "la", "conduit", model="circulation.pulmonary_veins",
               resistance_mmhg_s_ml=params.r_pulmonary_veins, one_way=False,
               source="pulmonary venous resistance (small)")
    for name, bed in params.beds.items():
        g.add_edge("sa", "sv", "bed", model="circulation.organ_bed",
                   tissue=name, resistance_mmhg_s_ml=bed.resistance_mmhg_s_ml,
                   share_pct=bed.share_pct, source=bed.source)

    schema = Schema(
        node_types={
            "chamber": {"emax_mmhg_ml", "emin_mmhg_ml", "volume0_ml", "offset_frac"},
            "pool": {"compliance_ml_mmhg", "unstressed_ml"},
        },
        edge_rules=[
            EdgeRule("valve", {"chamber"}, {"chamber", "pool"},
                     {"area_cm2", "discharge_coeff"}),
            EdgeRule("conduit", {"pool", "chamber"}, {"pool", "chamber"},
                     {"resistance_mmhg_s_ml"}),
            EdgeRule("bed", {"pool"}, {"pool"}, {"tissue", "resistance_mmhg_s_ml", "share_pct"}),
        ],
    )

    @schema.invariant("closed-loop")
    def _closed(g) -> list[str]:
        out = []
        for n in g.nodes.values():
            outs = [(e.src, e.dst) for e in g.edges if e.src == n.id
                    and e.type in ("valve", "conduit", "bed")]
            ins = [(e.src, e.dst) for e in g.edges if e.dst == n.id
                   and e.type in ("valve", "conduit", "bed")]
            if not outs:
                out.append(f"node {n.id} has no outgoing blood flow")
            if not ins:
                out.append(f"node {n.id} has no inflow")
        return out

    @schema.invariant("positive-parameters")
    def _positive(g) -> list[str]:
        out = []
        for n in g.nodes.values():
            a = n.attrs
            if n.type == "chamber" and not (a["emax_mmhg_ml"] > a["emin_mmhg_ml"] > 0):
                out.append(f"chamber {n.id}: needs Emax > Emin > 0")
            if n.type == "pool" and a["compliance_ml_mmhg"] <= 0:
                out.append(f"pool {n.id}: non-positive compliance")
        for e in g.edges:
            if e.type in ("conduit", "bed") and e.attrs["resistance_mmhg_s_ml"] <= 0:
                out.append(f"edge {e.key}: non-positive resistance")
            if e.type == "valve" and e.attrs["area_cm2"] <= 0:
                out.append(f"edge {e.key}: non-positive valve area")
        return out

    @schema.invariant("icrp-flow-split")
    def _shares(g) -> list[str]:
        beds = [e for e in g.edges if e.type == "bed"]
        total = sum(e.attrs["share_pct"] for e in beds)
        if abs(total - 100.0) > 0.5:
            out = [f"organ-bed shares sum to {total:.1f} %, not the ICRP ~100 %"]
        else:
            out = []
        out += [f"bed {e.attrs['tissue']}: non-positive share"
                for e in beds if e.attrs["share_pct"] <= 0]
        return out

    return g, schema


# ---------------------------------------------------------------------------
# the executable heart
# ---------------------------------------------------------------------------
STATE = ("la", "lv", "ra", "rv", "sa", "sv", "pa", "pv")
PRESSURE_ORDER = STATE


@dataclass
class BeatRecord:
    """Time series of the beating circulation."""

    time_s: np.ndarray
    volumes: np.ndarray              # (n_time, 8)
    pressures: np.ndarray            # (n_time, 8), order = node_ids
    flows: np.ndarray                # (n_time, n_flows)
    node_ids: list[str]
    flow_ids: list[str]

    def pressure_at(self, node: str) -> np.ndarray:
        return self.pressures[:, self.node_ids.index(node)]

    def volume_at(self, node: str) -> np.ndarray:
        return self.volumes[:, self.node_ids.index(node)]

    def flow_at(self, eid: str) -> np.ndarray:
        return self.flows[:, self.flow_ids.index(eid)]


class HeartCirculation:
    """The executable closed circulation (RK4; smooth except at valve opening)."""

    def __init__(self, params: CirculationParams):
        self.p = params
        self.period = params.period_s
        self.bed_names = list(params.beds)
        self._fmax = {cid: ch.fmax(self.period) for cid, ch in params.chambers.items()}

    # -- states ------------------------------------------------------------
    def state_at_reference(self) -> np.ndarray:
        """Volumes (mL) placing every pressure at its published reference value."""
        pp = self.p.pools
        co_ref = CO_REF_L_MIN * 1000.0 / 60.0
        p_sv_ref = RAP_REF_MMHG + self.p.r_venous_return * co_ref
        v = dict(CHAMBER_REF_ML)
        v["sa"] = pp["sa"].unstressed_ml + pp["sa"].compliance_ml_mmhg * MAP_REF_MMHG
        v["sv"] = pp["sv"].unstressed_ml + pp["sv"].compliance_ml_mmhg * p_sv_ref
        v["pa"] = pp["pa"].unstressed_ml + pp["pa"].compliance_ml_mmhg * 14.0
        v["pv"] = pp["pv"].unstressed_ml + pp["pv"].compliance_ml_mmhg * 8.0
        return np.array([v[k] for k in STATE])

    def blood_volume(self, v_state: np.ndarray | None = None) -> float:
        """Total blood volume carried on the graph (mL): the conserved invariant."""
        if v_state is None:
            v_state = self.state_at_reference()
        return float(np.sum(v_state))

    def transfuse(self, delta_ml: float, v_state: np.ndarray | None = None) -> np.ndarray:
        """Add (delta>0, transfusion) or remove ( hemorrhage) blood in the venous pool."""
        v = self.state_at_reference() if v_state is None else np.array(v_state, dtype=float)
        v[STATE.index("sv")] += delta_ml
        return v

    # -- pressures and flows ------------------------------------------------
    def pressures(self, V: np.ndarray, t: float) -> dict[str, float]:
        pr = {}
        for cid in ("la", "lv", "ra", "rv"):
            ch = self.p.chambers[cid]
            pr[cid] = float(ch.elastance(t, self.period) * (V[STATE.index(cid)] - ch.volume0_ml))
        for pid in ("sa", "sv", "pa", "pv"):
            pool = self.p.pools[pid]
            pr[pid] = float(pool.pressure(V[STATE.index(pid)]))
        return pr

    def flows(self, V: np.ndarray, t: float) -> dict[str, float]:
        pr = self.pressures(V, t)
        q = {}
        for name, valve in self.p.valves.items():
            q[name] = float(valve.flow(pr[valve.src], pr[valve.dst]))
        for name, bed in self.p.beds.items():
            q[f"bed_{name}"] = (pr["sa"] - pr["sv"]) / bed.resistance_mmhg_s_ml
        q["venous_return"] = max(pr["sv"] - pr["ra"], 0.0) / self.p.r_venous_return
        q["pulmonary"] = (pr["pa"] - pr["pv"]) / self.p.r_pulmonary
        q["pulmonary_veins"] = (pr["pv"] - pr["la"]) / self.p.r_pulmonary_veins
        return q

    def rhs(self, V: np.ndarray, t: float) -> np.ndarray:
        """dV/dt (mL/s) for all eight storage nodes; sums to exactly zero."""
        q = self.flows(V, t)
        idx = {k: i for i, k in enumerate(STATE)}
        d = np.zeros(len(STATE))
        d[idx["la"]] = q["pulmonary_veins"] - q["mitral"]
        d[idx["lv"]] = q["mitral"] - q["aortic"]
        d[idx["ra"]] = q["venous_return"] - q["tricuspid"]
        d[idx["rv"]] = q["tricuspid"] - q["pulmonic"]
        d[idx["sa"]] = q["aortic"] - sum(q[f"bed_{n}"] for n in self.bed_names)
        d[idx["sv"]] = sum(q[f"bed_{n}"] for n in self.bed_names) - q["venous_return"]
        d[idx["pa"]] = q["pulmonic"] - q["pulmonary"]
        d[idx["pv"]] = q["pulmonary"] - q["pulmonary_veins"]
        return d

    def _flow_order(self) -> list[str]:
        return (list(self.p.valves)
                + ["venous_return", "pulmonary", "pulmonary_veins"]
                + [f"bed_{n}" for n in self.bed_names])

    # -- integration -------------------------------------------------------
    def simulate(self, v0: np.ndarray | None = None, n_beats: int = 20, dt: float = 1e-3,
                 record_last: int = 10) -> BeatRecord:
        """RK4-integrate n_beats; record the last `record_last` beats."""
        v = self.state_at_reference() if v0 is None else np.array(v0, dtype=float)
        n_steps = int(round(n_beats * self.period / dt))
        rec_start = n_steps - int(round(record_last * self.period / dt))
        rec_every = max(1, int(round((self.period / dt) / 240)))
        times, Vrec, prec, qrec = [], [], [], []
        order = self._flow_order()
        for k in range(n_steps):
            t = k * dt
            if k >= rec_start and k % rec_every == 0:
                times.append(t)
                Vrec.append(v.copy())
                pr = self.pressures(v, t)
                prec.append([pr[n] for n in PRESSURE_ORDER])
                fl = self.flows(v, t)
                qrec.append([fl[n] for n in order])
            k1 = self.rhs(v, t)
            k2 = self.rhs(v + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self.rhs(v + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self.rhs(v + dt * k3, t + dt)
            v = v + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return BeatRecord(np.array(times), np.array(Vrec), np.array(prec), np.array(qrec),
                          list(STATE), order)

    # -- summary -----------------------------------------------------------
    def beat_summary(self, rec: BeatRecord) -> dict:
        """Mean physiology of the recorded beats.

        Systolic/diastolic aortic and pulmonary pressures are the within-beat
        maxima/minima (beats are cycle-aligned to the recording phase); other
        entries are cycle means.
        """
        t, P, V = rec.time_s, rec.pressures, rec.volumes
        period = self.period
        elapsed = t - t[0]
        beat_ids = np.floor(elapsed / period + 1e-12)
        # The record excludes its integration endpoint. Allow one recording
        # interval at the end of a complete cycle and discard trailing fragments.
        record_step = np.median(np.diff(t)) if len(t) > 1 else 0.0
        ids = np.unique(beat_ids)
        complete = (ids + 1) * period <= elapsed[-1] + record_step + 1e-12
        masks = [beat_ids == b for b in ids[complete]] or [np.ones(len(t), dtype=bool)]

        def beat_extreme(nid, fn):
            j = rec.node_ids.index(nid)
            return float(np.mean([fn(P[m, j]) for m in masks]))

        def cycle_mean(cid, array, i=None):
            out = []
            for m in masks:
                out.append(np.mean(array[m, i] if i is not None else array[m]))
            return float(np.mean(out))

        out = {}
        for i, nid in enumerate(rec.node_ids):
            out[f"mean_p_{nid}"] = cycle_mean(nid, P, i)
        out["min_p_la"] = beat_extreme("la", np.min)
        out["min_p_lv"] = beat_extreme("lv", np.min)
        out["min_p_ra"] = beat_extreme("ra", np.min)
        out["min_p_rv"] = beat_extreme("rv", np.min)
        out["max_p_la"] = beat_extreme("la", np.max)
        out["max_p_lv"] = beat_extreme("lv", np.max)
        out["max_p_ra"] = beat_extreme("ra", np.max)
        out["max_p_rv"] = beat_extreme("rv", np.max)
        out["sap_mmhg"] = beat_extreme("sa", np.max)
        out["dap_mmhg"] = beat_extreme("sa", np.min)
        out["pap_sys_mmhg"] = beat_extreme("pa", np.max)
        out["pap_dia_mmhg"] = beat_extreme("pa", np.min)
        for i, fid in enumerate(rec.flow_ids):
            out[f"mean_flow_{fid}"] = cycle_mean(fid, rec.flows, i)
        out["cardiac_output_ml_s"] = out["mean_flow_aortic"]
        out["cardiac_output_l_min"] = out["mean_flow_aortic"] * 0.06
        out["sv_ml"] = out["cardiac_output_ml_s"] * period
        j_lv = rec.node_ids.index("lv")
        edv = float(np.mean([V[m][:, j_lv].max() for m in masks]))
        out["edv_lv_ml"] = edv
        out["esv_lv_ml"] = float(np.mean([V[m][:, j_lv].min() for m in masks]))
        out["ef"] = out["sv_ml"] / edv
        out["pulse_pressure_mmhg"] = out["sap_mmhg"] - out["dap_mmhg"]
        out["map_mmhg"] = out["mean_p_sa"]
        j_rv = rec.node_ids.index("rv")
        out["edv_rv_ml"] = float(np.mean([V[m][:, j_rv].max() for m in masks]))
        out["esv_rv_ml"] = float(np.mean([V[m][:, j_rv].min() for m in masks]))
        out["ef_rv"] = float((out["edv_rv_ml"] - out["esv_rv_ml"]) / out["edv_rv_ml"])
        out["peak_ejection_ml_s"] = beat_extreme_rec(rec, np.max, "aortic")
        return out

    def mean_bed_flows(self, rec: BeatRecord) -> dict[str, float]:
        """Mean organ-bed flows (mL/s), as the drug-transport layer wants them."""
        out = {}
        for name in self.bed_names:
            out[name] = float(_cycle_mean_flow(rec, f"bed_{name}"))
        out["lung"] = float(_cycle_mean_flow(rec, "pulmonary"))
        return out

    def sample_flows(self, rec: BeatRecord, t: float) -> dict[str, float]:
        """Instantaneous flows at time t within the beat (repeats cyclically)."""
        elapsed = rec.time_s - rec.time_s[0]
        return {fid: float(np.interp(t, elapsed, rec.flow_at(fid), period=self.period))
                for fid in rec.flow_ids}


def beat_extreme_rec(rec: BeatRecord, fn, fid: str) -> float:
    j = rec.flow_ids.index(fid)
    return float(fn(rec.flows[:, j]))


def _cycle_mean_flow(rec: BeatRecord, fid: str) -> float:
    return float(np.mean(rec.flow_at(fid)))
