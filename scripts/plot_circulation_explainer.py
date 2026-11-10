"""Generate plain-language explainer charts for the beating circulation.

    python scripts/plot_circulation_explainer.py

Writes:
  docs/img/circulation_explained_loop.png
  docs/img/circulation_explained_beat.png
  docs/img/circulation_explained_starling.png
  docs/img/circulation_explained_drug.png
  docs/img/circulation_explained_scorecard.png

These are for humans, not for the validation itself.  The numbers behind them
come from the same model and the same report the validation page uses:
`docs/validation_circulation.json` (sweeps, baseline z-scores) and a live
45-beat simulation (waveforms, flow pulsatility, drug profiles).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # runnable as `python scripts/plot_circulation_explainer.py`
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

from vh.body import MIDAZOLAM, PBPKModel, Dose, reference_individual  # noqa: E402
from vh.graph.circulation import CirculationParams, HeartCirculation  # noqa: E402
from vh.graph.uptake import CirculationUptakeModel  # noqa: E402

IMG = ROOT / "docs" / "img"
DATA = json.loads((ROOT / "docs" / "validation_circulation.json").read_text())

INK = "#33415C"
RED = "#C2373A"
BLUE = "#2B44C4"
GREEN = "#2E8B57"
AMBER = "#E6A700"
GRAY = "#6B7280"
PINK = "#F6D5D5"
LIGHTBLUE = "#DCE8F5"
ORANGE = "#E08A2E"


# ---------------------------------------------------------------------------
# shared: one live reference simulation
# ---------------------------------------------------------------------------
def build_case():
    ind = reference_individual("male")
    params = CirculationParams.from_individual(ind)
    heart = HeartCirculation(params)
    rec = heart.simulate(n_beats=45, dt=1e-3, record_last=8)
    s = heart.beat_summary(rec)
    pbpk = PBPKModel(MIDAZOLAM, ind)
    u = CirculationUptakeModel(pbpk, heart)
    flows_h = u.mean_transport_flows(rec)
    u_mean = CirculationUptakeModel(pbpk, heart, mean_flows_l_h=flows_h)
    return {"ind": ind, "params": params, "heart": heart, "rec": rec, "s": s,
            "pbpk": pbpk, "u": u, "u_mean": u_mean,
            "doses": [Dose(0.0, 5.0, "iv"), Dose(0.0, 7.5, "oral")]}


def settled_beat(rec, heart):
    """One complete recorded beat, re-ordered so mitral closure is at t = 0."""
    t = rec.time_s
    beat = np.floor((t - t[0]) / heart.period)
    m = beat == np.unique(beat)[-2]
    tt = (t[m] - t[m][0]) * 1e3
    mit = rec.flow_at("mitral")[m]
    shift = tt[np.where(mit > 1e-9)[0][-1]]          # mitral closure
    tau = (tt - shift) % (heart.period * 1e3)
    order = np.argsort(tau)
    tau = tau[order]
    return {
        "tau": tau,
        "p_lv": rec.pressure_at("lv")[m][order],
        "p_ao": rec.pressure_at("sa")[m][order],
        "p_la": rec.pressure_at("la")[m][order],
        "v_lv": rec.volume_at("lv")[m][order],
        "q_aortic": rec.flow_at("aortic")[m][order],
        "q_mitral": rec.flow_at("mitral")[m][order],
    }


# ---------------------------------------------------------------------------
# 1. The loop, with real pressures and the real flow split
# ---------------------------------------------------------------------------
def loop_chart(c):
    s, params = c["s"], c["params"]
    fig = plt.figure(figsize=(12.8, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("One closed loop, two pumps in series", fontsize=13, loc="left",
                 color=INK, pad=2)

    def chamber(x, y, name, lo, hi, side):
        color = PINK if side == "left" else LIGHTBLUE
        edge = RED if side == "left" else BLUE
        ax.add_patch(FancyBboxPatch((x - 0.075, y - 0.052), 0.15, 0.104,
                                    boxstyle="round,pad=0.006,rounding_size=0.02",
                                    facecolor=color, edgecolor=edge, lw=1.6, zorder=3))
        ax.text(x, y + 0.012, name, ha="center", va="center", fontsize=10,
                weight="bold", color=edge, zorder=4)
        ax.text(x, y - 0.022, f"{lo:.0f}-{hi:.0f}", ha="center", va="center",
                fontsize=7.6, color=INK, zorder=4)

    def pool(x, y, name, value, side, bold=False):
        color = PINK if side == "left" else LIGHTBLUE
        edge = RED if side == "left" else BLUE
        ax.add_patch(FancyBboxPatch((x - 0.071, y - 0.046), 0.142, 0.092,
                                    boxstyle="round,pad=0.006,rounding_size=0.02",
                                    facecolor=color, edgecolor=edge, lw=1.4, zorder=3))
        ax.text(x, y + 0.012, name, ha="center", va="center", fontsize=9.5,
                weight="bold", color=edge, zorder=4)
        ax.text(x, y - 0.020, value, ha="center", va="center", fontsize=8.0,
                color=INK, zorder=4)

    def organ(x, y, w, h, name, sub, face, edge):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.008,rounding_size=0.02",
                                    facecolor=face, edgecolor=edge, lw=1.6, zorder=3))
        ax.text(x, y + 0.014, name, ha="center", va="center", fontsize=9.5,
                weight="bold", color=edge, zorder=4)
        ax.text(x, y - 0.020, sub, ha="center", va="center", fontsize=7.8,
                color=INK, zorder=4)

    def arrow(p1, p2, color=INK, rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                     lw=2.0, color=color, shrinkA=24, shrinkB=24,
                                     connectionstyle=f"arc3,rad={rad}", zorder=2))

    # positions
    lungs = (0.50, 0.88)
    pa, pv = (0.33, 0.70), (0.67, 0.70)
    ra, la = (0.13, 0.60), (0.87, 0.60)
    rv, lv = (0.13, 0.40), (0.87, 0.40)
    sv, sa = (0.27, 0.18), (0.73, 0.18)
    beds = (0.50, 0.085)

    organ(*lungs, 0.30, 0.100, "LUNGS", f"{s['cardiac_output_l_min']:.1f} L/min",
          "#EAF3EA", GREEN)
    pool(*pa, "PA", f"{s['mean_p_pa']:.1f} mmHg", "right")
    pool(*pv, "PV", f"{s['mean_p_pv']:.1f} mmHg", "left")
    chamber(*ra, "RA", s["min_p_ra"], s["max_p_ra"], "right")
    chamber(*la, "LA", s["min_p_la"], s["max_p_la"], "left")
    chamber(*rv, "RV", s["min_p_rv"], s["max_p_rv"], "right")
    chamber(*lv, "LV", s["min_p_lv"], s["max_p_lv"], "left")
    pool(*sv, "SV", f"{s['mean_p_sv']:.1f} mmHg", "right")
    pool(*sa, "SA", f"{s['mean_p_sa']:.1f} mmHg", "left")
    organ(*beds, 0.30, 0.085, "12 ORGAN BEDS", f"{s['cardiac_output_l_min']:.1f} L/min",
          "#FBEBDD", ORANGE)

    # flow arrows
    arrow(ra, rv, BLUE)
    arrow(rv, pa, BLUE)
    arrow(pa, lungs, BLUE)
    arrow(lungs, pv, RED)
    arrow(pv, la, RED)
    arrow(la, lv, RED)
    arrow(lv, sa, RED)
    arrow(sa, beds, RED)
    arrow(beds, sv, BLUE)
    arrow(sv, ra, BLUE, rad=-0.55)

    ax.text(0.50, 0.545, "pulmonary\ncircuit", ha="center", va="center",
            fontsize=8.2, color=GRAY)
    ax.text(0.50, 0.315, "systemic\ncircuit", ha="center", va="center",
            fontsize=8.2, color=GRAY)
    ax.text(0.045, 0.955, "deoxygenated", fontsize=8, color=BLUE, style="italic")
    ax.text(0.955, 0.955, "oxygenated", fontsize=8, color=RED, style="italic",
            ha="right")
    ax.text(0.5, -0.015,
            "The four arrows leaving a chamber cross a one-way valve.  Chamber "
            "labels are min-max mmHg; pool labels are mean mmHg.",
            ha="center", va="top", fontsize=7.8, color=GRAY)

    # -- right: where the output goes -------------------------------------
    bx = fig.add_subplot(gs[0, 1])
    items = sorted(((k, v.share_pct) for k, v in params.beds.items()),
                   key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = []
    for k in names:
        if k == "liver":
            colors.append(RED)
        elif k == "gut":
            colors.append(ORANGE)
        elif k == "kidney":
            colors.append(BLUE)
        else:
            colors.append("#9AA7B8")
    ys = np.arange(len(names))
    bx.barh(ys, vals, color=colors, height=0.68, alpha=0.92)
    for y, v in zip(ys, vals):
        bx.text(v + 0.35, y, f"{v:.1f}%", va="center", fontsize=8, color=INK)
    bx.set_yticks(ys)
    bx.set_yticklabels([n.capitalize() for n in names], fontsize=8.5)
    bx.set_xlim(0, 22)
    bx.set_xlabel("share of cardiac output (ICRP 89, %)", fontsize=9)
    bx.set_title("Where the output goes", fontsize=12, loc="left", color=INK)
    bx.grid(axis="x", alpha=0.22)
    bx.text(0.98, 0.03,
            "liver + gut + kidney = 40.5% of flow",
            transform=bx.transAxes, ha="right", va="bottom", fontsize=8,
            color=INK, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor="#CBD5E1", alpha=0.9))
    bx.annotate("main midazolam\nclearance organ", xy=(6.5, names.index("liver")),
                xytext=(12.5, names.index("liver") - 2.6), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    fig.savefig(IMG / "circulation_explained_loop.png", dpi=150,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. One heartbeat, phase by phase
# ---------------------------------------------------------------------------
def beat_chart(c):
    rec, heart = c["rec"], c["heart"]
    b = settled_beat(rec, heart)
    tau, T = b["tau"], heart.period * 1e3

    aor, mit = b["q_aortic"], b["q_mitral"]
    a_open, a_close = tau[aor > 1e-9][0], tau[aor > 1e-9][-1]
    m_open, m_close = tau[mit > 1e-9][0], tau[mit > 1e-9][-1]
    phases = [
        (0.0, a_open, "IC", "#EFEFEF"),
        (a_open, a_close, "ejection", "#F8DCDC"),
        (a_close, m_open, "IR", "#EFEFEF"),
        (m_open, m_close, "filling", "#DCE8F5"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.4), constrained_layout=True)
    axp, axv, axq, axl = axs.flat
    for ax in (axp, axv):
        for x0, x1, label, col in phases:
            ax.axvspan(x0, x1, color=col, alpha=0.9, lw=0, zorder=0)

    axp.plot(tau, b["p_lv"], color=RED, lw=2.0, label="LV pressure")
    axp.plot(tau, b["p_ao"], color="#7A1F1F", lw=1.5, ls="--", label="aortic pressure")
    axp.plot(tau, b["p_la"], color=BLUE, lw=1.3, label="LA pressure")
    axp.set_xlim(0, T)
    axp.set_ylim(0, 125)
    axp.set_ylabel("pressure (mmHg)")
    axp.set_xlabel("time from mitral closure (ms)")
    axp.set_title("One beat: pressure, volume, and the valves that sequence them",
                  fontsize=12.5, loc="left", color=INK)
    handles, labels = axp.get_legend_handles_labels()
    handles.append(Patch(facecolor="#EFEFEF", edgecolor="#CBD5E1",
                         label="both valves closed"))
    axp.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")
    axp.grid(alpha=0.25)

    axv.plot(tau, b["v_lv"], color=RED, lw=2.0)
    axv.fill_between(tau, 0, b["v_lv"], color=RED, alpha=0.08)
    edv, esv = b["v_lv"].max(), b["v_lv"].min()
    axv.axhline(edv, color=INK, lw=0.8, ls=":")
    axv.axhline(esv, color=INK, lw=0.8, ls=":")
    axv.text(T * 0.02, edv + 3, f"EDV {edv:.0f} mL", fontsize=8.5, color=INK)
    axv.text(T * 0.02, esv - 9, f"ESV {esv:.0f} mL", fontsize=8.5, color=INK)
    x_sv = tau[np.argmax(b["v_lv"])]
    axv.annotate("", xy=(x_sv, edv), xytext=(x_sv, esv),
                 arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.8))
    axv.text(x_sv + 12, (edv + esv) / 2, f"stroke volume\n{edv - esv:.0f} mL",
             fontsize=8.5, color=GREEN, va="center")
    axv.set_xlim(0, T)
    axv.set_ylim(0, 170)
    axv.set_ylabel("LV volume (mL)")
    axv.set_xlabel("time from mitral closure (ms)")
    axv.set_title("The ventricle empties by about two-thirds", fontsize=11,
                  loc="left", color=INK)
    axv.grid(alpha=0.25)

    axq.plot(tau, aor, color="#7A1F1F", lw=1.8, label="aortic valve")
    axq.plot(tau, mit, color=BLUE, lw=1.6, label="mitral valve")
    axq.axhline(0, color=INK, lw=0.8)
    axq.set_xlim(0, T)
    axq.set_ylim(-8, max(aor.max(), mit.max()) * 1.15)
    axq.set_ylabel("valve flow (mL/s)")
    axq.set_xlabel("time from mitral closure (ms)")
    axq.set_title("Valves are one-way: closed means exactly zero", fontsize=11,
                  loc="left", color=INK)
    axq.legend(fontsize=8, frameon=False)
    axq.grid(alpha=0.25)

    # PV loop, coloured by phase.  The two isovolumic phases are separate
    # vertical segments at ESV and EDV; plotting them as one line would draw a
    # false diagonal between them.
    fill = mit > 1e-9
    eject = aor > 1e-9
    iso_c = tau < a_open
    iso_r = (tau > a_close) & (tau < m_open)
    axl.plot(b["v_lv"][iso_c], b["p_lv"][iso_c], color="#9AA7B8", lw=1.6,
             ls="--", label="isovolumic")
    axl.plot(b["v_lv"][iso_r], b["p_lv"][iso_r], color="#9AA7B8", lw=1.6, ls="--")
    axl.plot(b["v_lv"][fill], b["p_lv"][fill], color=BLUE, lw=2.2, label="filling")
    axl.plot(b["v_lv"][eject], b["p_lv"][eject], color=RED, lw=2.2, label="ejection")
    axl.scatter([edv, esv], [b["p_lv"][np.argmax(b["v_lv"])],
                             b["p_lv"][np.argmin(b["v_lv"])]],
                color=INK, s=22, zorder=4)
    work_j = abs(np.trapezoid(b["p_lv"], b["v_lv"])) * 0.133322e-3
    axl.text(0.03, 0.96,
             f"area = stroke work\n{work_j:.2f} J per beat",
             transform=axl.transAxes, va="top", fontsize=8.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#CBD5E1", alpha=0.9))
    axl.set_xlim(esv - 15, edv + 15)
    axl.set_ylim(0, 125)
    axl.set_xlabel("LV volume (mL)")
    axl.set_ylabel("LV pressure (mmHg)")
    axl.set_title("The pressure-volume loop", fontsize=11, loc="left", color=INK)
    axl.legend(fontsize=8, frameon=True, framealpha=0.92, loc="center",
               edgecolor="#CBD5E1")
    axl.grid(alpha=0.25)

    fig.savefig(IMG / "circulation_explained_beat.png", dpi=150,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Why the heart pumps what it does: fill, squeeze, rate
# ---------------------------------------------------------------------------
def starling_chart(c):
    ind = c["ind"]
    sweep = DATA["sweeps"]["_sweep"]       # [volume %, CO, SV, EF, EDV, LAP]
    esweep = DATA["sweeps"]["_esweep"]     # [Emax x, EF, EDV, LAP, CO, SV]

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.2, 4.6),
                                     constrained_layout=True)

    # -- preload -> output -------------------------------------------------
    # Keep the normal operating range (>= -10 %); the -20 % hemorrhage point
    # collapses to EDV 39 mL and is documented in the validation report.
    rows = [r for r in sweep if r[0] >= -10]
    edv = np.array([r[4] for r in rows])
    co = np.array([r[1] for r in rows])
    delta = np.array([r[0] for r in rows])
    ax.plot(edv, co, color=RED, lw=2.0, zorder=2)
    sc = ax.scatter(edv, co, c=delta, cmap="coolwarm", s=70, edgecolor="white",
                    linewidth=1.2, zorder=3)
    base = next(r for r in rows if r[0] == 0)
    ax.scatter([base[4]], [base[1]], facecolors="none", edgecolors=INK,
               s=180, linewidth=1.6, zorder=4)
    ax.annotate("normal resting point\n143 mL → 6.36 L/min",
                xy=(base[4], base[1]), xytext=(120, 7.8), fontsize=8,
                color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.annotate("+10% blood volume\nmore stretch, more output",
                xy=(edv[-1], co[-1]), xytext=(148, 8.3), fontsize=8,
                color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.annotate("-10%", xy=(edv[0], co[0]), xytext=(edv[0] + 4, co[0] + 0.6),
                fontsize=8, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    ax.set_xlabel("left-ventricular end-diastolic volume (mL)")
    ax.set_ylabel("cardiac output (L/min)")
    ax.set_title("More filling → more output", fontsize=11.5, loc="left",
                 color=INK)
    ax.grid(alpha=0.25)
    ax.set_xlim(85, 205)
    ax.set_ylim(3.4, 9.3)

    # -- contractility -----------------------------------------------------
    scale = np.array([r[0] for r in esweep])
    ef = np.array([r[1] for r in esweep])
    edv2 = np.array([r[2] for r in esweep])
    bx.plot(scale, ef * 100, color=RED, lw=2.0, marker="o", ms=5,
            label="ejection fraction")
    bx.set_xlabel("contractility (Emax multiplier)")
    bx.set_ylabel("ejection fraction (%)", color=RED)
    bx.tick_params(axis="y", labelcolor=RED)
    bx.set_ylim(30, 72)
    bx.set_title("A weaker squeeze dilates the ventricle", fontsize=11.5,
                 loc="left", color=INK)
    bx2 = bx.twinx()
    bx2.plot(scale, edv2, color=BLUE, lw=1.8, ls="--", marker="s", ms=4,
             label="end-diastolic volume")
    bx2.set_ylabel("LV end-diastolic volume (mL)", color=BLUE)
    bx2.tick_params(axis="y", labelcolor=BLUE)
    bx2.set_ylim(130, 180)
    bx.annotate("half strength:\nEF 49%, EDV 159 mL", xy=(0.5, 48.9),
                xytext=(0.62, 36), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    bx.grid(alpha=0.25)

    # -- heart rate --------------------------------------------------------
    hrs, cos, svs = [], [], []
    for hr in (55, 70, 85, 100):
        pp = CirculationParams.from_individual(ind)
        pp.heart_rate_min = hr
        hh = HeartCirculation(pp)
        rr = hh.simulate(n_beats=30, dt=1e-3, record_last=5)
        ss = hh.beat_summary(rr)
        hrs.append(hr)
        cos.append(ss["cardiac_output_l_min"])
        svs.append(ss["sv_ml"])
    cx.plot(hrs, cos, color=RED, lw=2.0, marker="o", ms=5, label="cardiac output")
    cx.set_xlabel("heart rate (beats/min)")
    cx.set_ylabel("cardiac output (L/min)", color=RED)
    cx.tick_params(axis="y", labelcolor=RED)
    cx.set_ylim(5.3, 7.5)
    cx.set_title("Faster rate: smaller stroke, similar output", fontsize=11.5,
                 loc="left", color=INK)
    cx2 = cx.twinx()
    cx2.plot(hrs, svs, color=BLUE, lw=1.8, ls="--", marker="s", ms=4,
             label="stroke volume")
    cx2.set_ylabel("stroke volume (mL)", color=BLUE)
    cx2.tick_params(axis="y", labelcolor=BLUE)
    cx2.set_ylim(60, 115)
    cx.annotate("100 bpm:\nSV 70 mL, CO 6.99", xy=(99, 6.99), xytext=(76, 7.22),
                fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    cx.grid(alpha=0.25)
    cx.set_xlim(50, 105)

    fig.text(0.5, -0.035,
             "Open-loop model: no baroreflex and no venoconstriction.  These curves "
             "show mechanism and direction, not the compensated response of an intact animal.",
             ha="center", va="top", fontsize=8.2, color=GRAY)
    fig.savefig(IMG / "circulation_explained_starling.png", dpi=150,
                bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. The drug on the beat: what the waveform changes
# ---------------------------------------------------------------------------
def drug_chart(c):
    heart, rec, u, u_mean = c["heart"], c["rec"], c["u"], c["u_mean"]
    doses = c["doses"]
    op = u.simulate_pulsatile(rec, doses, 24.0, dt_transport_s=0.004,
                              sample_every_h=0.05)
    om = u_mean.simulate_mean(doses, 24.0, dt_h=0.05)
    t = op["time_h"]
    cp, cm = op["plasma"], om["plasma"]
    cum_p, cum_m = op["plasma_auc_mg_h_l"], om["plasma_auc_mg_h_l"]
    effect = cum_p[-1] / cum_m[-1] - 1.0

    fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.2), constrained_layout=True)
    ax, bx, cx, dx = axs.flat

    # -- 24 h profiles -----------------------------------------------------
    ax.semilogy(t, cm, color=BLUE, lw=1.8, label="mean-flow engine")
    ax.semilogy(t, cp, color=RED, lw=1.4, alpha=0.9,
                label="beat-coupled (this layer)")
    ax.fill_between(t, cm, cp, color=RED, alpha=0.10)
    ax.set_xlim(0, 24)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("plasma midazolam (mg/L, log)")
    ax.set_title("Same mean flows, different waveform", fontsize=11.5,
                 loc="left", color=INK)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")
    ax.text(0.97, 0.95, f"24 h AUC: {effect:+.4%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=RED, weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#E5B5B5", alpha=0.95))

    # -- cumulative AUC effect --------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(cum_m > 0, cum_p / cum_m - 1.0, np.nan)
    bx.plot(t, 100 * ratio, color=RED, lw=2.0)
    bx.axhline(100 * effect, color=INK, lw=0.9, ls=":")
    bx.annotate(f"final {effect:+.4%}", (t[-1], 100 * effect),
                xytext=(-8, 6), textcoords="offset points", ha="right",
                color=INK, fontsize=8.5)
    bx.axhline(0, color=GRAY, lw=0.8)
    bx.set_xlim(0, 24)
    bx.set_xlabel("time (h)")
    bx.set_ylabel("cumulative AUC difference (%)")
    bx.set_title("When the difference appears", fontsize=11.5, loc="left",
                 color=INK)
    bx.grid(alpha=0.25)

    # -- how the pulse is smoothed around the loop ------------------------
    beat = settled_beat(rec, heart)
    T = heart.period
    q_bed = np.zeros_like(beat["tau"])
    t_rec = rec.time_s
    m = np.floor((t_rec - t_rec[0]) / T) == np.unique(
        np.floor((t_rec - t_rec[0]) / T))[-2]
    for tissue in c["params"].beds:
        q_bed += rec.flow_at(f"bed_{tissue}")[m]
    # align bed sum to the same beat ordering used above
    tt = (t_rec[m] - t_rec[m][0]) * 1e3
    mit = rec.flow_at("mitral")[m]
    shift = tt[np.where(mit > 1e-9)[0][-1]]
    tau = (tt - shift) % (T * 1e3)
    order = np.argsort(tau)
    q_bed = q_bed[order]

    # Peak / cycle mean: the closed phase (zero flow) counts, so this is the
    # pulsatility the organ actually sees.
    q_pul = rec.flow_at("pulmonic")[m][order]
    q_ven = rec.flow_at("venous_return")[m][order]

    # Peak / *cycle* mean: the closed phase (zero flow) is part of the cycle,
    # so this is the pulsatility the organ actually sees.
    labels = ["aortic\n(ejection)", "pulmonic\n(ejection)", "organ beds\n(sum)",
              "venous\nreturn"]
    values = [
        beat["q_aortic"].max() / beat["q_aortic"].mean(),
        q_pul.max() / q_pul.mean(),
        q_bed.max() / q_bed.mean(),
        q_ven.max() / q_ven.mean(),
    ]
    colors = [RED, "#7A1F1F", ORANGE, BLUE]
    ys = np.arange(len(labels))[::-1]
    cx.barh(ys, values, color=colors, height=0.62, alpha=0.9)
    for y, v in zip(ys, values):
        cx.text(v + 0.12, y, f"{v:.2f}x", va="center", fontsize=9, color=INK)
    cx.axvline(1.0, color=GRAY, lw=0.9, ls=":")
    cx.set_yticks(ys)
    cx.set_yticklabels(labels, fontsize=8.5)
    cx.set_xlim(0, 8.0)
    cx.set_xlabel("peak flow / cycle-mean flow (-)")
    cx.set_title("The elastic pools smooth the pulse", fontsize=11.5,
                 loc="left", color=INK)
    cx.grid(axis="x", alpha=0.25)
    cx.text(0.97, 0.05,
            "7x at the valve → 1.2x at the beds",
            transform=cx.transAxes, ha="right", va="bottom", fontsize=8.2,
            color=INK, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor="#CBD5E1", alpha=0.9))

    # -- within-beat ripple at the sampling vein --------------------------
    fine = u.simulate_pulsatile(rec, [Dose(0.0, 5.0, "iv")], 3.5 / 3600.0,
                                dt_transport_s=0.004,
                                sample_every_h=1e-3 / 3600.0)
    sec = fine["time_h"] * 3600.0
    cyc = (sec >= 2.0 * T) & (sec < 3.0 * T)
    swing = float(fine["plasma"][cyc].max() / fine["plasma"][cyc].min() - 1.0)
    dx.plot((sec[cyc] - 2.0 * T) * 1e3, fine["plasma"][cyc], color=RED, lw=2.0)
    dx.axhline(fine["plasma"][cyc].mean(), color=BLUE, lw=1.0, ls="--",
               label="cycle mean")
    dx.set_xlabel("time within one beat (ms)")
    dx.set_ylabel("plasma midazolam (mg/L)")
    dx.set_title("Plasma during the third beat after a bolus", fontsize=11.5,
                 loc="left", color=INK)
    dx.text(0.5, 0.06,
            f"peak-to-trough {swing:.1%}\n(post-bolus distribution)",
            transform=dx.transAxes, ha="center", va="bottom", fontsize=8.5,
            color=RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#E5B5B5", alpha=0.95))
    dx.legend(fontsize=8, frameon=False, loc="upper right")
    dx.grid(alpha=0.25)

    fig.savefig(IMG / "circulation_explained_drug.png", dpi=150,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. How close the blind predictions land
# ---------------------------------------------------------------------------
def scorecard_chart():
    rows = []
    for e in DATA["baseline"]["entries"]:
        z = e.get("error")
        if z is None:
            continue
        rows.append((e["check"], float(z)))
    rows.sort(key=lambda item: item[1])

    def color(z):
        if abs(z) <= 1.0:
            return GREEN
        if abs(z) <= 3.0:
            return AMBER
        return RED

    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    fig.subplots_adjust(left=0.31, right=0.985, top=0.88, bottom=0.20)
    ys = np.arange(len(rows))
    ax.axvspan(-1, 1, color=GREEN, alpha=0.09, lw=0)
    ax.axvspan(-3, -1, color=AMBER, alpha=0.07, lw=0)
    ax.axvspan(1, 3, color=AMBER, alpha=0.07, lw=0)
    for z in (-3, -1, 1, 3):
        ax.axvline(z, color=INK, lw=0.9, ls=":" if abs(z) == 1 else "--",
                   alpha=0.6)
    for y, (name, z) in zip(ys, rows):
        c = color(z)
        ax.plot([0, z], [y, y], color=c, lw=2.2, alpha=0.55, solid_capstyle="round")
        ax.scatter([z], [y], color=c, s=80, zorder=3, edgecolor="white",
                   linewidth=1.1)
        ax.text(z + (0.18 if z >= 0 else -0.18), y, f"{z:+.2f}",
                va="center", ha="left" if z >= 0 else "right", fontsize=8.2,
                color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([name for name, _ in rows], fontsize=9)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("distance from the published normal, in population SDs")
    ax.set_title("Baseline comparisons with published SDs (Broomé 2013, Table 1)",
                 fontsize=12, loc="left", color=INK)
    ax.grid(axis="x", alpha=0.2)
    ax.text(0.99, 0.02,
            "green: within 1 SD   •   amber: 1-3 SD\n"
            "arterial pressures have no reported SD",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
            color=GRAY)
    fig.savefig(IMG / "circulation_explained_scorecard.png", dpi=150)
    plt.close(fig)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        c = build_case()
        loop_chart(c)
        beat_chart(c)
        starling_chart(c)
        drug_chart(c)
        scorecard_chart()
    print("wrote docs/img/circulation_explained_"
          "{loop,beat,starling,drug,scorecard}.png")


if __name__ == "__main__":
    main()
