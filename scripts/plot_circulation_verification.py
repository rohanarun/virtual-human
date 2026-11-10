"""Plot the measured effects of the circulation corrections.

Pass a checkout of the original circulation implementation as --before.
Both checkouts run the same dose schedule. The AUC step-size comparison uses
this checkout's generated validation_circulation.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "img"
OLD, NEW, INK = "#bf5545", "#007f79", "#172a3a"


def snapshot(checkout):
    """Run a checkout in isolation and capture actual transport matrix inputs."""
    sys.path[:0] = [str(checkout), str(checkout / "scripts")]
    from validate_circulation import build_context
    from vh.body import Dose
    from vh.graph.uptake import CirculationUptakeModel

    class CaptureFlows(CirculationUptakeModel):
        def matrix(self, flows_l_h):
            segment_flows.append(dict(flows_l_h))
            return super().matrix(flows_l_h)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        case = build_context()
        rec, heart, doses = case["rec"], case["heart"], case["doses"]
        segment_flows = []
        model = CaptureFlows(case["pbpk"], heart)
        pulse = model.simulate_pulsatile(rec, doses, 24, sample_every_h=0.05)
        steady = case["u_h"].simulate_mean(doses, 24)
        if "plasma_auc_mg_h_l" in pulse:
            auc_pulse = pulse["plasma_auc_mg_h_l"][-1]
            auc_steady = steady["plasma_auc_mg_h_l"][-1]
        else:
            auc_pulse, auc_steady = (np.trapezoid(run["plasma"], run["time_h"])
                                     for run in (pulse, steady))
        # Off-grid start and end on a constant-flow template. Summing the drug
        # states (including elimination sinks) gives the total mass delivered.
        start_s, duration_s, amount = 60, 60, 1.0
        dose = Dose(start_s / 3600, amount, "iv", duration_h=duration_s / 3600)
        infusion = case["u"].simulate_pulsatile(case["flat"], [dose], 174 / 3600,
                                                sample_every_h=18 / 3600)
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=checkout, text=True).strip(),
        "period_ms": heart.period * 1000,
        "flow_l_h": {name: [q[name] for q in segment_flows] for name in ("lung", "liver")},
        "relative_auc": float(auc_pulse / auc_steady - 1),
        "infusion": {"start_s": start_s, "duration_s": duration_s, "amount_mg": amount,
                     "time_s": (infusion["time_h"] * 3600).tolist(),
                     "mass_mg": infusion["state"].sum(axis=1).tolist()},
    }


def draw(before, after):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    plt.rcParams.update({"figure.facecolor": "#f8fafc", "axes.facecolor": "#f8fafc",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": INK,
                         "text.color": INK, "font.size": 11})
    baseline = f"Original sampler ({before['commit']})"

    def save(fig, filename, title):
        fig.suptitle(title, fontsize=17, fontweight="bold")
        fig.savefig(IMAGES / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    for ax, name, label in zip(axes, ("lung", "liver"), ("Pulmonary bed", "Hepatic artery")):
        old, new = (np.asarray(run["flow_l_h"][name]) for run in (before, after))
        for run, q, color, legend in ((before, old, OLD, baseline),
                                      (after, new, NEW, "Full-cycle sampler")):
            edges = np.linspace(0, run["period_ms"], len(q) + 1)
            ax.stairs(q, edges, baseline=None, color=color, linewidth=2, label=legend)
            ax.axhline(q.mean(), color=color, linewidth=1, linestyle="--", alpha=0.65)
        ax.set(title=f"{label}  |  original mean {old.mean() / new.mean() - 1:+.1%}",
               xlabel="Time in the applied cycle (ms)", ylabel="Transport input (L/h)")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper left", fontsize=9)
    save(fig, "circulation_verified_flow.png", "Sampling the whole beat restores the flow averages")

    report = json.loads((ROOT / "docs" / "validation_circulation.json").read_text())
    steps = report["waveform_effect"]["relative_effect_by_step"]
    fig, (ax, detail) = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    effects = [100 * before["relative_auc"], 100 * after["relative_auc"]]
    ax.barh([1, 0], effects, color=[OLD, NEW], height=0.48)
    ax.set_yticks([1, 0], ["Original comparison\n(unmatched flows, sampled AUC)",
                           "Corrected comparison\n(matched flows, integrated AUC)"])
    ax.set(xlabel="24-hour AUC change (%)", title="Original and corrected comparisons",
           xlim=(min(effects) * 1.12, max(0, *effects) + 1.2))
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.text(effects[0] / 2, 1, f"{effects[0]:+.2f}%", ha="center", va="center",
            color="white", fontweight="bold")
    ax.annotate(f"{effects[1]:+.4f}%", (effects[1], 0), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center", color=NEW,
                fontweight="bold")
    x = [float(step.split()[0]) for step in steps]
    y = [100 * change for change in steps.values()]
    detail.plot(x, y, color=NEW, marker="o", linewidth=2)
    detail.set(xticks=sorted(x), xlabel="Transport step (ms)",
               ylabel="24-hour AUC change (%)",
               title="Corrected result by step size (zoomed)")
    center = effects[1]
    padding = max(abs(center) * 0.025, 2 * max(abs(v - center) for v in y))
    detail.set(xlim=(min(x) - 0.4, max(x) + 0.4),
               ylim=(center - padding, center + padding))
    detail.yaxis.set_major_locator(MaxNLocator(nbins=5))
    detail.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    for step, change in zip(x, y):
        detail.annotate(f"{change:+.6f}%", (step, change), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9, color=NEW)
    detail.grid(alpha=0.2)
    save(fig, "circulation_verified_auc.png", "The large exposure difference disappears with matched flows")

    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    schedule = after["infusion"]
    start, duration, amount = (schedule[k] for k in ("start_s", "duration_s", "amount_mg"))
    end = start + duration
    ax.axvspan(start, end, color=NEW, alpha=0.08)
    ax.plot([0, start, end, schedule["time_s"][-1]], [0, 0, amount, amount],
            color=INK, linestyle="--", linewidth=2.2, label="Scheduled delivery")
    for run, color, legend in ((before, OLD, "Original solver"), (after, NEW, "Corrected solver")):
        inf = run["infusion"]
        ax.scatter(inf["time_s"], inf["mass_mg"], s=34, color=color, zorder=3,
                label=f"{legend}  |  {inf['mass_mg'][-1]:.2f} mg delivered")
    ax.set(xlabel="Simulation time (s)", ylabel="Cumulative dose (mg)")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "circulation_verified_infusion.png", "Splitting at the dose boundaries restores the full infusion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--before", type=Path, help="checkout of the original circulation implementation")
    mode.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.snapshot is not None:
        print(json.dumps(snapshot(args.snapshot.resolve())))
        return
    before = json.loads(subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "--snapshot", str(args.before.resolve())],
        text=True))
    draw(before, snapshot(ROOT))


if __name__ == "__main__":
    main()
