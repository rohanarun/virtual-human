"""Whole-body physiologically based pharmacokinetic (PBPK) engine.

Perfusion-limited organs connected by blood flow, with
  - liver elimination: well-stirred model, CLint scaled from in vitro enzyme
    kinetics (IVIVE, microsomal protein per gram liver) or given directly;
  - renal elimination: glomerular filtration of unbound drug;
  - oral absorption: stomach + 7-compartment small-intestine transit
    (CAT model, Yu & Amidon 1999) with first-pass gut-wall metabolism
    (Qgut model, Yang et al. 2007).

The model is linear in drug amount (valid when concentrations stay well
below the enzymes' Km, true for clinical doses of most drugs), so it is
solved EXACTLY with matrix exponentials: any time step, no integration
error, and fast enough to stream a virtual patient in real time.

Units: amount mg, volume L, flow L/h, concentration mg/L, time h.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from .compound import Compound
from .partition import kp_all, predict_bp
from .physiology import PORTAL, TISSUES, Individual, reference_individual

N_TRANSIT = 7
SI_TRANSIT_H = 3.32        # small-intestinal transit time (Yu & Amidon 1999)
GASTRIC_HALF_LIFE_H = 0.25  # fasted gastric emptying half-life (~15 min)
SI_RADIUS_CM = 1.75         # small-intestine radius for ka = 2 Peff / R
Q_VILLI_L_H = 18.0          # villous blood flow (Yang 2007)
SI_AREA_CM2 = 6600.0        # absorptive surface 0.66 m^2 (Yang 2007)
HEPATIC_CYP3A4_PMOL_MG = 137.0   # CYP3A4 in liver microsomes (Simcyp default, Rowland Yeo 2004)
GUT_CYP3A4_NMOL = 70.5           # total small-intestinal CYP3A4 (Paine 1997; Yang 2007)

SYSTEMIC = tuple(t for t in TISSUES if t not in PORTAL and t not in ("lung", "liver"))


@dataclass
class Dose:
    time_h: float
    amount_mg: float
    route: str = "iv"            # "iv" or "oral"
    duration_h: float = 0.0      # IV infusion duration (0 = bolus)


@dataclass
class DerivedParams:
    kp: dict
    bp: float
    fu: float
    clint_hep_l_h: float
    cl_hep_blood_l_h: float      # well-stirred prediction (for reporting)
    cl_renal_l_h: float          # plasma
    ka_h: float
    fa: float
    fg: float
    vss_l: float
    q_gut_l_h: float = 0.0       # Qgut model "flow" term
    clint_gut_l_h: float = 0.0   # enterocyte intrinsic clearance
    sources: dict = field(default_factory=dict)

    @property
    def cl_total_plasma_l_h(self):
        return self.cl_hep_blood_l_h * self.bp + self.cl_renal_l_h

    @property
    def effective_half_life_h(self):
        """ln2 * Vss / CL = ln2 * mean residence time.

        This equals the elimination half-life only for a one-compartment
        model. For this multi-compartment model the terminal half-life --
        the one a clinical terminal slope measures -- is longer and is set
        by the slowest eigenvalue of the system matrix; see
        PBPKModel.terminal_half_life_h().
        """
        return np.log(2) * self.vss_l / self.cl_total_plasma_l_h

    @property
    def half_life_h(self):
        """Backward-compatible alias for effective_half_life_h (ln2*Vss/CL).

        Not the model's terminal half-life; prefer effective_half_life_h
        for this quantity and PBPKModel.terminal_half_life_h() for the
        terminal slope.
        """
        return self.effective_half_life_h


class PBPKModel:
    """Builds the linear system dx/dt = M x + u(t) for one compound in one individual."""

    def __init__(self, compound: Compound, individual: Individual | None = None, **overrides):
        self.compound = c = compound
        self.ind = ind = individual or reference_individual()
        self.overrides = overrides
        fu = overrides.get("fu", c.fu)
        bp = overrides.get("bp", c.bp)
        if bp is None:
            bp = predict_bp(c.logp, c.ionization, fu, ind.hematocrit)
        kp = kp_all(c.logp, c.ionization, fu, bp, ind.hematocrit)
        kp.update(overrides.get("kp", {}))

        # --- hepatic intrinsic clearance (unbound, L/h) ---------------------
        if overrides.get("clint_hepatic_l_h") is not None:
            clint = overrides["clint_hepatic_l_h"]
        elif c.clint_hepatic_l_h is not None:
            clint = c.clint_hepatic_l_h * ind.enzyme_scale.get("hepatic", 1.0)
        else:
            liver_g = ind.volumes["liver"] * 1000.0
            clint = sum(
                e.clint_ul_min_mg * ind.mppgl * liver_g * 60e-6 * ind.enzyme_scale.get(e.enzyme, 1.0)
                for e in c.enzymes
            )
        clint *= overrides.get("clint_scale", 1.0)
        q_h = ind.liver_flow_total
        fub = fu / bp
        cl_h_blood = q_h * fub * clint / (q_h + fub * clint)
        cl_r = c.renal_filtration * fu * ind.gfr

        # --- absorption -------------------------------------------------------
        peff = overrides.get("peff_1e4_cm_s", c.peff_1e4_cm_s)
        if peff:
            ka = 2.0 * peff * 1e-4 / SI_RADIUS_CM * 3600.0
            cl_perm = peff * 1e-4 * SI_AREA_CM2 * 3.6          # cm^3/s -> L/h
            q_gut = Q_VILLI_L_H * cl_perm / (Q_VILLI_L_H + cl_perm)
            clint_gut = sum(
                e.clint_ul_min_mg / HEPATIC_CYP3A4_PMOL_MG * GUT_CYP3A4_NMOL * 1000.0 * 60e-6
                * ind.enzyme_scale.get(e.enzyme, 1.0)
                for e in c.enzymes if e.enzyme in c.gut_enzymes
            ) * overrides.get("clint_scale", 1.0)
            fg = q_gut / (q_gut + clint_gut)
        else:
            ka, fg, q_gut, clint_gut = 0.0, 1.0, 0.0, 0.0
        fg = overrides.get("fg", fg)
        kt = N_TRANSIT / SI_TRANSIT_H
        fa = 1.0 - (1.0 + ka / kt) ** (-N_TRANSIT) if ka > 0 else 0.0

        v = ind.volumes
        vss = float(sum(kp[t] * v[t] for t in TISSUES) + (ind.v_arterial + ind.v_venous) * bp)
        self.p = DerivedParams(kp=kp, bp=bp, fu=fu, clint_hep_l_h=clint, cl_hep_blood_l_h=cl_h_blood,
                               cl_renal_l_h=cl_r, ka_h=ka, fa=fa, fg=fg, vss_l=vss,
                               q_gut_l_h=q_gut, clint_gut_l_h=clint_gut)
        self._build_matrix(kt)

    # ------------------------------------------------------------------------
    def _build_matrix(self, kt):
        ind, p = self.ind, self.p
        names = list(TISSUES) + ["arterial", "venous", "stomach"] + [f"si{i + 1}" for i in range(N_TRANSIT)] + \
                ["elim_hepatic", "elim_renal", "elim_gut_wall", "feces"]
        ix = {n: i for i, n in enumerate(names)}
        n = len(names)
        M = np.zeros((n, n))
        V = dict(ind.volumes)
        V["arterial"], V["venous"] = ind.v_arterial, ind.v_venous
        Q, bp, kp = ind.flows, p.bp, p.kp
        out_k = {t: bp / kp[t] / V[t] for t in TISSUES}       # (venous blood conc)/(amount)

        def flow(src, dst, rate):           # first-order transfer src -> dst, rate in 1/h
            M[ix[src], ix[src]] -= rate
            M[ix[dst], ix[src]] += rate

        q_co = ind.cardiac_output
        flow("venous", "lung", q_co / V["venous"])
        flow("lung", "arterial", q_co * out_k["lung"])
        for t in TISSUES:
            if t == "lung":
                continue
            qin = Q[t]                                       # arterial supply (hepatic artery for liver)
            flow("arterial", t, qin / V["arterial"])
        for t in SYSTEMIC:
            flow(t, "venous", Q[t] * out_k[t])
        for t in PORTAL:
            flow(t, "liver", Q[t] * out_k[t])
        flow("liver", "venous", ind.liver_flow_total * out_k["liver"])
        # elimination
        flow("liver", "elim_hepatic", p.clint_hep_l_h * p.fu / (kp["liver"] * V["liver"]))
        flow("kidney", "elim_renal", p.cl_renal_l_h / (kp["kidney"] * V["kidney"]))
        # oral absorption: stomach -> si1..si7 -> feces; absorbed -> gut wall (Fg) -> gut tissue -> portal
        flow("stomach", "si1", np.log(2) / GASTRIC_HALF_LIFE_H)
        for i in range(1, N_TRANSIT + 1):
            seg = f"si{i}"
            nxt = f"si{i + 1}" if i < N_TRANSIT else "feces"
            flow(seg, nxt, kt)
            if p.ka_h > 0:
                flow(seg, "gut", p.ka_h * p.fg)
                flow(seg, "elim_gut_wall", p.ka_h * (1.0 - p.fg))
        self.M, self.names, self.ix, self.V = M, names, ix, V
        self._cache = {}

    # ------------------------------------------------------------------------
    def propagator(self, h: float, infusion_mg_h: float = 0.0):
        """Exact one-step map: x(t+h) = E x(t) + f (f = effect of a constant IV infusion)."""
        key = (round(h, 12), round(infusion_mg_h, 12))
        if key not in self._cache:
            n = self.M.shape[0]
            A = np.zeros((n + 1, n + 1))
            A[:n, :n] = self.M
            A[self.ix["venous"], n] = infusion_mg_h
            E = expm(A * h)
            self._cache[key] = (E[:n, :n], E[:n, n])
            if len(self._cache) > 256:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def observe(self, x) -> dict:
        """Concentrations (mg/L) from the state vector (amounts, mg)."""
        x = np.asarray(x)
        ix, V, p = self.ix, self.V, self.p
        out = {"plasma": x[..., ix["venous"]] / V["venous"] / p.bp,
               "arterial_plasma": x[..., ix["arterial"]] / V["arterial"] / p.bp}
        for t in TISSUES:
            out[t] = x[..., ix[t]] / V[t]
        out["brain_unbound"] = out["brain"] * p.fu / p.kp["brain"]
        for k in ("elim_hepatic", "elim_renal", "elim_gut_wall", "feces"):
            out[k + "_mg"] = x[..., ix[k]]
        return out

    def simulate(self, doses: list[Dose], t_end_h: float, dt_h: float = 0.05) -> dict:
        """Simulate on a uniform grid, handling dose and infusion events exactly."""
        n = self.M.shape[0]
        times = np.arange(0.0, t_end_h + 1e-9, dt_h)
        events = sorted({0.0, t_end_h, *[d.time_h for d in doses],
                         *[d.time_h + d.duration_h for d in doses if d.route == "iv" and d.duration_h > 0]})
        x = np.zeros(n)
        t = 0.0
        states = np.zeros((len(times), n))

        def infusion_rate(tt):
            return sum(d.amount_mg / d.duration_h for d in doses
                       if d.route == "iv" and d.duration_h > 0 and d.time_h <= tt + 1e-12 < d.time_h + d.duration_h)

        def apply_boluses(tt, x):
            for d in doses:
                if abs(d.time_h - tt) < 1e-12:
                    if d.route == "oral":
                        x[self.ix["stomach"]] += d.amount_mg
                    elif d.duration_h <= 0:
                        x[self.ix["venous"]] += d.amount_mg
            return x

        k_out = 0
        x = apply_boluses(0.0, x)
        states[0] = x
        k_out = 1
        breakpoints = sorted(set(events) | set(times.tolist()))
        for t_next in breakpoints[1:]:
            h = t_next - t
            if h <= 1e-12:
                continue
            E, f = self.propagator(h, infusion_rate(t))
            x = E @ x + f
            t = t_next
            x = apply_boluses(t, x)
            while k_out < len(times) and abs(times[k_out] - t) < 1e-9:
                states[k_out] = x
                k_out += 1
        obs = self.observe(states)
        obs["time_h"] = times
        obs["state"] = states
        return obs

    def mass_balance(self, x) -> float:
        return float(np.sum(x))

    # Absorbing accumulator states: drug only flows in, so they carry the
    # matrix's zero eigenvalues and must be excluded from decay analysis.
    SINKS = ("elim_hepatic", "elim_renal", "elim_gut_wall", "feces")

    def terminal_half_life_h(self) -> float:
        """Half-life of the model's slowest decay mode, ln2 / -max Re(eig).

        Computed from the eigenvalues of M restricted to the non-sink
        states. This is the half-life of the terminal log-linear phase of
        the plasma curve -- the quantity clinical terminal half-lives
        measure. Returns inf if the non-sink system has no decaying mode
        (e.g. all clearances zero).
        """
        keep = [i for i, n in enumerate(self.names) if n not in self.SINKS]
        lam = np.linalg.eigvals(self.M[np.ix_(keep, keep)])
        slowest = float(np.max(lam.real))
        return float(np.log(2) / -slowest) if slowest < 0 else float("inf")

    def summary(self) -> str:
        p = self.p
        return (f"{self.compound.name} in {self.ind.label}: Vss {p.vss_l:.0f} L ({p.vss_l / self.ind.body_mass:.2f} L/kg), "
                f"B:P {p.bp:.2f}, fu {p.fu:.3f}, CLint,hep {p.clint_hep_l_h:.0f} L/h -> CL_plasma "
                f"{p.cl_total_plasma_l_h:.1f} L/h (hepatic blood {p.cl_hep_blood_l_h:.1f}, renal {p.cl_renal_l_h:.2f}), "
                f"t1/2,terminal ~{self.terminal_half_life_h():.1f} h (effective ln2*Vss/CL ~{p.effective_half_life_h:.1f} h) | "
                f"oral: ka {p.ka_h:.2f}/h, Fa {p.fa:.2f}, Fg {p.fg:.2f}, "
                f"Fh {1 - p.cl_hep_blood_l_h / self.ind.liver_flow_total:.2f}")


class RealTimeSimulator:
    """Steps a virtual patient forward in arbitrary increments with live dosing.

    This is the interface the browser demo mirrors: dose() at any moment,
    advance(dt) by any amount (exact), read observe() for organ concentrations.
    """

    def __init__(self, model: PBPKModel):
        self.model = model
        self.x = np.zeros(model.M.shape[0])
        self.t = 0.0
        self.infusions = []        # (end_time, rate)

    def dose(self, amount_mg: float, route: str = "iv", duration_h: float = 0.0):
        if route == "oral":
            self.x[self.model.ix["stomach"]] += amount_mg
        elif duration_h > 0:
            self.infusions.append((self.t + duration_h, amount_mg / duration_h))
        else:
            self.x[self.model.ix["venous"]] += amount_mg

    def advance(self, dt_h: float):
        t_end = self.t + dt_h
        while self.t < t_end - 1e-12:
            active = [(e, r) for e, r in self.infusions if e > self.t + 1e-12]
            self.infusions = active
            next_stop = min([t_end] + [e for e, _ in active])
            rate = sum(r for _, r in active)
            E, f = self.model.propagator(next_stop - self.t, rate)
            self.x = E @ self.x + f
            self.t = next_stop
        return self.observe()

    def observe(self) -> dict:
        o = {k: float(v) for k, v in self.model.observe(self.x).items()}
        o["time_h"] = self.t
        return o
