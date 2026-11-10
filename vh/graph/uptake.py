"""Drug transport riding the beating circulation: organ uptake on the graph.

The validated PBPK engine (:mod:`vh.body.pbpk`) treats organ blood flow as
constant.  That is exact for whole-body exposure on hours, but the heart beats,
and the question "how does the medicine reach an organ" has a beat-by-beat
answer the body-level engine cannot see.  This module couples the two without
inventing a second mass balance:

  - the **state**, volumes, partition coefficients, clearance rates and the oral
    absorption chain are exactly those of :class:`vh.body.pbpk.PBPKModel`, so at
    the ICRP reference flows the matrix built here is *identical* to the
    validated one (checked to 1e-12 in the tests);
  - the **flows** become the organ-bed flows the :class:`~vh.graph.circulation`
    heart actually produces: constant means for the mean-stream model, the real
    pulsatile q(t) for the beat-coupled model;
  - because every edge is a rate `src -> dst`, the drug state matrix has
    *exactly zero column sums* at every instant: total drug (including what has
    been metabolized/excreted, which are states here) is conserved to machine
    precision by each exact matrix-exponential step, however the heartbeat
    modulates the flows.

Two authorities structure the validation:

  1. at ICRP flows the transport matrix must equal the validated PBPK matrix
     (implementation-level cross-check), and both solvers must agree;
  2. the pulsatile run must converge to the mean-stream run as the transport
     step shrinks, and the difference between the two quantifies how much the
     heartbeat actually modulates organ uptake.

Units: amount mg, volume L, flow L/h, concentration mg/L, time h
(circulation-record flows are mL/s and converted once, at the interface).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from vh.body.physiology import PORTAL, TISSUES

from .circulation import BeatRecord, HeartCirculation

LH_PER_ML_S = 3.6               # 1 mL/s = 3.6 L/h
SINKS = ("elim_hepatic", "elim_renal", "elim_gut_wall", "feces")


@dataclass
class _BeatTransport:
    """Precomputed exact per-segment algebra for one recorded cardiac cycle.

    The cycle is tiled into ``n_seg`` equal segments; segment ``k`` has the
    exact one-step exponential ``E[k]`` and unit-rate infusion plateau ``f[k]``.
    The affine products let any span of segments advance in O(1) matmuls:

    ``S[k], F[k]``  suffix pair: applying segments k..n_seg-1 (completes the
    beat) gives ``x' = S[k] @ x + rate * F[k]``.
    ``L[k], Lf[k]``  prefix pair: applying segments 0..k-1 from the beat start
    gives ``x' = L[k] @ x + rate * Lf[k]``.
    ``P_aug``  full-beat propagator ``[[S[0], F[0]], [0, 1]]`` with a unit-rate
    column, so ``matrix_power(P_aug, m)`` advances m whole beats exactly
    (the affine part scales linearly with the rate).
    """

    n: int
    n_seg: int
    dt_seg_s: float
    dt_h: float
    E: list
    f: list
    S: list
    F: list
    L: list
    Lf: list
    P_aug: np.ndarray


# ---------------------------------------------------------------------------
# the matrix: the validated PBPK structure with replaceable flows
# ---------------------------------------------------------------------------
def transport_matrix(model, flows_l_h: dict[str, float]) -> np.ndarray:
    """The drug state matrix for a given set of organ flows (L/h).

    Same state vector, volumes, partition coefficients, hepatic/renal
    elimination and oral absorption chain as :class:`vh.body.pbpk.PBPKModel`;
    only blood flows are supplied externally (so they can come from the beating
    circulation).  At the ICRP reference flows this returns exactly the
    validated matrix.
    """
    return transport_matrix_full(
        names=model.names, ix=model.ix, V=model.V, kp=model.p.kp, bp=model.p.bp,
        fu=model.p.fu, clint_l_h=model.p.clint_hep_l_h, cl_renal_l_h=model.p.cl_renal_l_h,
        ka_h=model.p.ka_h, fg=model.p.fg, flows_l_h=flows_l_h,
    )


def transport_matrix_full(names, ix, V, kp, bp, fu, clint_l_h, cl_renal_l_h,
                          ka_h, fg, flows_l_h: dict[str, float]) -> np.ndarray:
    """As :func:`transport_matrix`, given the raw model parameters."""
    n = len(names)
    M = np.zeros((n, n))
    out_k = {t: bp / kp[t] / V[t] for t in TISSUES}       # (venous-equivalent)/amount

    def flow(src, dst, rate):          # first-order transfer src -> dst, 1/h
        M[ix[src], ix[src]] -= rate
        M[ix[dst], ix[src]] += rate

    q_lung = flows_l_h["lung"]
    liver_total = flows_l_h["liver"] + sum(flows_l_h[t] for t in PORTAL)
    flow("venous", "lung", q_lung / V["venous"])
    flow("lung", "arterial", q_lung * out_k["lung"])
    for t in TISSUES:
        if t == "lung":
            continue
        flow("arterial", t, flows_l_h[t] / V["arterial"])
    for t in TISSUES:
        if t in ("lung", "liver"):
            continue
        dest = "liver" if t in PORTAL else "venous"
        flow(t, dest, flows_l_h[t] * out_k[t])
    flow("liver", "venous", liver_total * out_k["liver"])
    # elimination
    flow("liver", "elim_hepatic", clint_l_h * fu / (kp["liver"] * V["liver"]))
    flow("kidney", "elim_renal", cl_renal_l_h / (kp["kidney"] * V["kidney"]))
    # oral absorption chain (constant rates: they are tissue properties)
    from vh.body import pbpk as _pbpk
    kt = _pbpk.N_TRANSIT / _pbpk.SI_TRANSIT_H
    flow("stomach", "si1", np.log(2) / _pbpk.GASTRIC_HALF_LIFE_H)
    for i in range(1, _pbpk.N_TRANSIT + 1):
        seg = f"si{i}"
        nxt = f"si{i + 1}" if i < _pbpk.N_TRANSIT else "feces"
        flow(seg, nxt, kt)
        if ka_h > 0:
            flow(seg, "gut", ka_h * fg)
            flow(seg, "elim_gut_wall", ka_h * (1.0 - fg))
    return M


# ---------------------------------------------------------------------------
# organ uptake diagnostics on a state vector
# ---------------------------------------------------------------------------
def organ_net_uptake_l_h(model, x, flows_l_h: dict[str, float]) -> dict[str, float]:
    """Net organ uptake, mg/h, per tissue: Q (C_in - C_out) at this instant.

    Positive = net uptake from blood into tissue.  For the liver the inflow is
    the hepatic artery plus the three portal outflows.
    """
    xb = np.asarray(x, dtype=float)
    ix, V, p = model.ix, model.V, model.p
    out_k = {t: p.bp / p.kp[t] / V[t] for t in TISSUES}
    c_arterial = xb[ix["arterial"]] / V["arterial"]        # total blood conc, mg/L
    uptake = {}
    for t in TISSUES:
        if t == "lung":
            inflow = flows_l_h["lung"]
            amount_in = (xb[ix["venous"]] / V["venous"]) * inflow
        elif t == "liver":
            q_in = flows_l_h["liver"]
            amount_in = (c_arterial * flows_l_h["liver"]
                         + sum(out_k[g] * xb[ix[g]] * flows_l_h[g] for g in PORTAL))
        else:
            q_in = flows_l_h[t]
            amount_in = c_arterial * flows_l_h[t]
        amount_out = out_k[t] * xb[ix[t]] * (flows_l_h[t] if t != "lung" else flows_l_h["lung"])
        uptake[t] = float(amount_in - amount_out)
    return uptake


def extraction_ratio_liver(model, x, flows_l_h: dict[str, float]) -> float:
    """Instantaneous hepatic extraction ratio: fraction removed of liver net inflow."""
    ix, V, p = model.ix, model.V, model.p
    out_liver = p.bp / p.kp["liver"] / V["liver"]
    # inflow (mg/h): hepatic-artery blood + portal organ outflows
    inflow = (x[ix["arterial"]] / V["arterial"]) * flows_l_h["liver"] \
        + sum((p.bp / p.kp[g] / V[g]) * x[ix[g]] * flows_l_h[g] for g in PORTAL)
    outflow = out_liver * x[ix["liver"]] * (flows_l_h["liver"] + sum(flows_l_h[g] for g in PORTAL))
    if inflow <= 0:
        return 0.0
    return float(1.0 - outflow / inflow)


# ---------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------
class CirculationUptakeModel:
    """The drug state on the beating circulation.

    ``scale`` multiplies the individual's ICRP organ flows (``1.0`` -> exactly
    the validated PBPK physiology, the bridge to the body engine).  Flows may
    also be given explicitly from the heartbeat mean bed flows.
    """

    def __init__(self, pbpk, heart: HeartCirculation | None = None,
                 scale: float = 1.0, mean_flows_l_h: dict[str, float] | None = None):
        self.pbpk = pbpk
        self.heart = heart
        self.scale = scale
        if mean_flows_l_h is not None:
            self.flows_mean_l_h = dict(mean_flows_l_h)
        elif scale == 1.0:
            self.flows_mean_l_h = dict(pbpk.ind.flows)
        else:
            self.flows_mean_l_h = {t: q * scale for t, q in pbpk.ind.flows.items()}

    # -- matrix ------------------------------------------------------------
    def matrix(self, flows_l_h: dict[str, float]) -> np.ndarray:
        return transport_matrix(self.pbpk, flows_l_h)

    # -- exact steady state (mean flows) -----------------------------------
    def steady_state(self, infusion_mg_h: float) -> np.ndarray:
        """Analytic steady state under constant flow and infusion (M x = -u).

        Solved on the drug-cycling sub-block; the sink states (eliminated
        drug) grow linearly forever, so they are excluded and filled from the
        rates of the cycling block.
        """
        M = self.matrix(self.flows_mean_l_h)
        ix = self.pbpk.ix
        n = M.shape[0]
        cycling = np.array([i for i in range(n) if self.pbpk.names[i] not in SINKS])
        sink = np.array([i for i in range(n) if self.pbpk.names[i] in SINKS])
        A = M[np.ix_(cycling, cycling)]
        u = np.zeros(len(cycling))
        u[cycling.tolist().index(ix["venous"])] = infusion_mg_h
        xc = np.linalg.solve(A, -u)
        x = np.zeros(n)
        x[cycling] = xc
        # sinks accumulate at their production rate at steady state
        prod = M[:, cycling] @ xc
        x[sink] = np.maximum(prod[sink], 0.0)
        return x

    # -- simulate (mean flows, exact per step as pbpk) -----------------------
    def simulate_mean(self, doses, t_end_h: float, dt_h: float = 0.05) -> dict:
        """Exact stream of the PBPK state at constant mean flows.

        Same event logic as :meth:`vh.body.pbpk.PBPKModel.simulate`; output has
        the same keys as its ``observe``.
        """
        M = self.matrix(self.flows_mean_l_h)
        n = M.shape[0]
        ix = self.pbpk.ix
        times = np.arange(0.0, t_end_h + 1e-9, dt_h)
        events = sorted({0.0, t_end_h,
                         *[d.time_h for d in doses],
                         *[d.time_h + d.duration_h for d in doses
                           if d.route == "iv" and d.duration_h > 0]})
        x = np.zeros(n)
        t = 0.0
        states = np.zeros((len(times), n))

        def rate_at(tt):
            return sum(d.amount_mg / d.duration_h for d in doses
                       if d.route == "iv" and d.duration_h > 0
                       and d.time_h <= tt + 1e-12 < d.time_h + d.duration_h)

        def bolus_at(tt, xx):
            for d in doses:
                if abs(d.time_h - tt) < 1e-12:
                    if d.route == "oral":
                        xx[ix["stomach"]] += d.amount_mg
                    elif d.duration_h <= 0:
                        xx[ix["venous"]] += d.amount_mg
            return xx

        x = bolus_at(0.0, x)
        states[0] = x
        k = 1

        def propagator(h, rate):
            nn = M.shape[0]
            aug = np.zeros((nn + 1, nn + 1))
            aug[:nn, :nn] = M
            aug[ix["venous"], nn] = rate
            E = expm(aug * h)
            return E[:nn, :nn], E[:nn, nn]

        breakpoints = sorted(set(events) | set(times.tolist()))
        for t_next in breakpoints[1:]:
            h = t_next - t
            if h <= 1e-12:
                continue
            E, f = propagator(h, rate_at(t))
            x = E @ x + f
            t = t_next
            x = bolus_at(t, x)
            while k < len(times) and abs(times[k] - t) < 1e-9:
                states[k] = x
                k += 1
        obs = self.pbpk.observe(states)
        obs["time_h"] = times
        obs["state"] = states
        return obs

    def _beat_transport(self, beat: BeatRecord, dt_transport_s: float) -> _BeatTransport:
        """Precompute the exact per-segment exponentials of one cardiac cycle.

        The cycle is tiled into ``ceil(T/dt)`` equal segments; flows are the
        recorded beat's bed/lung flows sampled at segment centres (linear
        interpolation across the wrap), held piecewise-constant.  For each
        segment the exact one-step exponential ``E_k`` is formed, plus the
        unit-rate infusion plateau ``f_k``; then the affine suffix/prefix
        products of :class:`_BeatTransport`.  This is bookkeeping only: every
        later advance reuses these same exact exponentials.
        """
        n = self.pbpk.M.shape[0]
        ix = self.pbpk.ix
        t_rec = beat.time_s
        phase_u = (t_rec - t_rec[0]) % self.heart.period
        order = np.argsort(phase_u)
        ph = phase_u[order]
        n_seg = max(8, int(np.ceil(self.heart.period / dt_transport_s)))
        dt_seg = self.heart.period / n_seg
        centres = 0.5 * (np.arange(n_seg) + 0.5) * dt_seg
        base = {"lung": beat.flow_at("pulmonary")[order]}
        for b in self.heart.bed_names:
            base[b] = beat.flow_at(f"bed_{b}")[order]
        wrapped = np.r_[ph, self.heart.period]
        q_l_h = {k: np.interp(centres, wrapped, np.r_[v, v[0]]) * LH_PER_ML_S
                 for k, v in base.items()}
        dt_h = dt_seg / 3600.0
        E, f = [], []
        aug = np.zeros((n + 1, n + 1))
        aug[n, n] = 1.0
        # scipy's expm scales-and-squares with enormous probe powers while
        # picking the Pade structure; on this platform numpy then raises
        # spurious matmul RuntimeWarnings (divide/overflow/invalid) -- they
        # fire for plain pbpk.simulate as well and the results are unaffected
        # (verified exact to 1e-15).  The flags are deferred: numpy reports
        # them at the first following matmul, so the suppression covers this
        # whole setup body.  Scoped to these loops, not global.
        with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore",
                                                    divide="ignore"):
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for k in range(n_seg):
                M = self.matrix({t: float(q_l_h[t][k]) for t in q_l_h})
                aug[:n, :n] = M
                aug[ix["venous"], n] = 1.0
                A = expm(aug * dt_h)
                E.append(A[:n, :n])
                f.append(A[:n, n])
            # suffix pairs: applying segments k .. n_seg-1 completes the beat
            S = [None] * (n_seg + 1)
            F = [None] * (n_seg + 1)
            S[n_seg], F[n_seg] = np.eye(n), np.zeros(n)
            for k in range(n_seg - 1, -1, -1):
                S[k] = S[k + 1] @ E[k]
                F[k] = S[k + 1] @ f[k] + F[k + 1]
            # prefix pairs: applying segments 0 .. k-1 from the beat start
            L, Lf = [np.eye(n)], [np.zeros(n)]
            for k in range(n_seg):
                L.append(E[k] @ L[-1])
                Lf.append(E[k] @ Lf[-1] + f[k])
        # full-beat propagator, unit-rate column (affine part scales with rate)
        P_aug = np.zeros((n + 1, n + 1))
        P_aug[:n, :n] = S[0]
        P_aug[:n, n] = F[0]
        P_aug[n, n] = 1.0
        return _BeatTransport(n=n, n_seg=n_seg, dt_seg_s=dt_seg, dt_h=dt_h,
                              E=E, f=f, S=S, F=F, L=L, Lf=Lf, P_aug=P_aug)

    @staticmethod
    def _advance_segments(x, seg0: int, m: int, rate: float, bt: _BeatTransport):
        """Advance the drug state by exactly m segments at one constant rate.

        Whole beats regroup into a matrix power of the full-beat propagator
        (exact regrouping of the same per-segment exponentials; only the
        floating-point association order changes).  Partial beats use the
        suffix/prefix affine pairs; a span fully inside one beat steps
        segment by segment.
        """
        if m <= 0:
            return x, seg0
        n_seg, n = bt.n_seg, bt.n
        if seg0 + m <= n_seg:                 # entirely within one beat
            for k in range(seg0, seg0 + m):
                x = bt.E[k] @ x + rate * bt.f[k]
            return x, seg0 + m
        q = n_seg - seg0                       # complete the current beat
        if q:
            x = bt.S[seg0] @ x + rate * bt.F[seg0]
            m -= q
        full, rem = divmod(m, n_seg)
        if full:
            with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore",
                                                        divide="ignore"):
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                mp = np.linalg.matrix_power(bt.P_aug, full)
            x = mp[:n, :n] @ x + rate * mp[:n, n]
        if rem:
            x = bt.L[rem] @ x + rate * bt.Lf[rem]
        return x, rem

    def simulate_pulsatile(self, beat: BeatRecord, doses, t_end_h: float,
                           dt_transport_s: float = 0.004,
                           sample_every_h: float = 0.01) -> dict:
        """Exact transport with the beat's own flows: q_i(t) piecewise-constant.

        The cardiac cycle is tiled into equal segments (< ``dt_transport_s``);
        on each segment the drug state advances by the exact matrix exponential
        of that segment's transport matrix, so total drug mass (with the
        eliminated amounts tracked as states) is conserved to machine precision
        however the heartbeat modulates the flows.  Long runs regroup whole
        beats into matrix powers of the full-beat propagator (an exact
        regrouping of the same exponentials).  Boluses land on the segment
        grid, within one segment (~4 ms) of their nominal time.
        """
        bt = self._beat_transport(beat, dt_transport_s)
        ix = self.pbpk.ix
        iv = [d for d in doses if d.route == "iv" and d.duration_h > 0]

        def rate_at(t):
            return sum(d.amount_mg / d.duration_h for d in iv
                       if d.time_h <= t < d.time_h + d.duration_h)

        if sample_every_h is None:
            sample_every_h = bt.dt_h
        times = np.arange(0.0, t_end_h + 1e-12, sample_every_h)
        if times[-1] < t_end_h - 1e-9:
            times = np.append(times, t_end_h)
        boluses = sorted((d.time_h, d.amount_mg, d.route) for d in doses
                         if (d.route == "oral" or d.duration_h <= 0)
                         and d.time_h > 1e-12)
        events = sorted({e for d in iv for e in (d.time_h, d.time_h + d.duration_h)}
                        | {b[0] for b in boluses})
        bounds = np.unique(np.r_[0.0, t_end_h, times,
                                 np.asarray(events, dtype=float)])
        bounds = bounds[bounds <= t_end_h + 1e-12]

        x = np.zeros(bt.n)
        for d in doses:
            if abs(d.time_h) < 1e-12:
                if d.route == "oral":
                    x[ix["stomach"]] += d.amount_mg
                elif d.duration_h <= 0:
                    x[ix["venous"]] += d.amount_mg
        states = np.zeros((len(times), bt.n))
        states[0] = x
        seg0, t_cur, k, bi = 0, 0.0, 1, 0
        for b in bounds[1:]:
            b = min(float(b), t_end_h)
            m = int(round((b - t_cur) / bt.dt_h))
            if m > 0:
                x, seg0 = self._advance_segments(x, seg0, m, rate_at(t_cur), bt)
                t_cur += m * bt.dt_h
            while bi < len(boluses) and boluses[bi][0] <= t_cur + bt.dt_h * 0.5:
                _, amount, route = boluses[bi]
                x[ix["stomach" if route == "oral" else "venous"]] += amount
                bi += 1
            while k < len(times) and times[k] <= t_cur + bt.dt_h + 1e-12:
                states[k] = x
                k += 1
        obs = self.pbpk.observe(states)
        obs["time_h"] = times
        obs["state"] = states
        return obs
