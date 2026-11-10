"""Compare pulsatile circulation and steady-flow controls with published PK curves.

Uses the same cohort inclusion and demographics as validate_body.py, with
linked OSP study schedules when available. An externally calibrated scenario
uses Thummel et al. clearance measurements without fitting this panel.
"""
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validate_body import IV_INFUSION_H, dose_for, individual_for, summarize, to_mg_l
from vh.body import Dose, MIDAZOLAM, PBPKModel, clearance_overrides, reference_individual
from vh.graph.circulation import CirculationParams, HeartCirculation
from vh.graph.uptake import CirculationUptakeModel

DOCS = ROOT / "docs"
FLOW_MODES = ("icrp_steady", "heart_steady", "heart_pulsatile")
SAMPLE_H = 0.0025


def evaluate_panel():
    data = json.loads((ROOT / "data/clinical/midazolam_osp_observed.json").read_text())
    schedules = json.loads((ROOT / "data/clinical/midazolam_osp_schedules.json").read_text())
    usable = [ds for ds in data["datasets"]
              if ds["data_type"] in ("aggregated", "typical representative")]
    scale = json.loads((DOCS / "validation_body.json").read_text())["clint_scale_middle_out"]
    reference = json.loads((ROOT / "data/clinical/midazolam_thummel_clearance.json").read_text())
    pmid = reference["source"].rstrip("/").rsplit("/", 1)[-1]
    if any(ds["reference"].rstrip("/").rsplit("/", 1)[-1] == pmid for ds in usable):
        raise ValueError("clearance calibration study is present in the evaluation panel")
    independent = clearance_overrides(
        PBPKModel(MIDAZOLAM, reference_individual("male")),
        reference["mean_systemic_plasma_clearance_ml_min"] * 0.06,
        reference["mean_apparent_oral_plasma_clearance_ml_min"] * 0.06,
    )
    scenarios = {"bottom_up": {}, "iv_calibrated": {"clint_scale": scale},
                 "independent_clearance": independent}
    groups = defaultdict(list)
    for ds in usable:
        ind = individual_for(ds)
        groups[ind.sex, ind.age, ind.body_mass, ind.height].append((ds, ind))

    results = []
    for group in groups.values():
        ind = group[0][1]
        heart = HeartCirculation(CirculationParams.from_individual(ind))
        beat = heart.simulate(n_beats=45, dt=1e-3, record_last=8)
        for scenario, parameters in scenarios.items():
            pbpk = PBPKModel(MIDAZOLAM, ind, **parameters)
            model = CirculationUptakeModel(pbpk, heart)
            matched = CirculationUptakeModel(pbpk, heart,
                                             mean_flows_l_h=model.mean_transport_flows(beat))
            protocols = defaultdict(list)
            for ds, _ in group:
                timing = schedules["datasets"].get(ds["name"], {})
                if "doses" in timing:
                    doses = [Dose(**dose) for dose in timing["doses"]]
                else:
                    dose = dose_for(ds, ind)
                    doses = [replace(dose, duration_h=timing.get("iv_duration_h", dose.duration_h))]
                total = sum(dose.amount_mg for dose in doses)
                shape = tuple((d.time_h, d.amount_mg / total, d.route, d.duration_h) for d in doses)
                protocols[shape].append((ds, total, timing.get("protocol")))
            for shape, datasets in protocols.items():
                horizon = max(max(ds["time"]) for ds, _, _ in datasets)
                # Linear kinetics let cohorts sharing a protocol shape reuse
                # a unit-dose simulation, including multi-dose schedules.
                unit = [Dose(*d) for d in shape]
                runs = {
                    "icrp_steady": model.simulate_mean(unit, horizon, dt_h=SAMPLE_H),
                    "heart_steady": matched.simulate_mean(unit, horizon, dt_h=SAMPLE_H),
                    "heart_pulsatile": model.simulate_pulsatile(beat, unit, horizon,
                                                               sample_every_h=SAMPLE_H),
                }
                for ds, dose, protocol in datasets:
                    t = np.array(ds["time"])
                    c = to_mg_l(ds["conc"], ds["conc_unit"], MIDAZOLAM.mw)
                    keep = (t > 0) & (c > 0)
                    t, c = t[keep], c[keep]
                    assert len(t) >= 2 and np.all(np.diff(t) > 0), ds["name"]
                    entry = {"dataset": ds["name"], "study": ds["study"], "route": ds["route"],
                             "osp_protocol": protocol,
                             "scenario": scenario, "n_samples": len(t),
                             "auc_obs": float(np.trapezoid(c, t)), "cmax_obs": float(c.max())}
                    for mode, run in runs.items():
                        pred = np.interp(t, run["time_h"], run["plasma"]) * dose
                        assert np.all(np.isfinite(pred) & (pred > 0)), ds["name"]
                        results.append(entry | {
                            "flow": mode, "auc_pred": float(np.trapezoid(pred, t)),
                            "cmax_pred": float(pred.max()),
                            "aafe": float(np.exp(np.mean(np.abs(np.log(pred / c))))),
                        })
    return {"provenance": data["provenance"], "studies": len({ds["study"] for ds in usable}),
            "iv_clearance_multiplier": scale, "fallback_iv_infusion_h": IV_INFUSION_H,
            "schedule_source": schedules["source"],
            "independent_clearance": reference | {"overrides": independent},
            "sample_every_h": SAMPLE_H, "results": results}


def paired_study_comparison(results, route):
    """Resample studies, keeping all paired profiles within each study together."""
    by_scenario = {
        scenario: {r["dataset"]: r for r in results if (r["scenario"], r["route"], r["flow"])
                   == (scenario, route, "heart_pulsatile")}
        for scenario in ("iv_calibrated", "independent_clearance")
    }
    old, new = by_scenario.values()
    assert old.keys() == new.keys()
    differences = defaultdict(list)
    for name, row in new.items():
        new_error = abs(np.log(row["auc_pred"] / row["auc_obs"]))
        old_error = abs(np.log(old[name]["auc_pred"] / old[name]["auc_obs"]))
        differences[row["study"]].append(new_error - old_error)
    sums = np.array([sum(values) for values in differences.values()])
    counts = np.array([len(values) for values in differences.values()])
    draws = np.random.default_rng(0).integers(0, len(sums), size=(10_000, len(sums)))
    ratios = np.exp(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
    return {"studies": len(sums), "auc_gmfe_ratio": float(np.exp(sums.sum() / counts.sum())),
            "study_bootstrap_95pct": np.quantile(ratios, [0.025, 0.975]).tolist()}


def aggregate(report):
    summary = []
    for scenario in ("bottom_up", "iv_calibrated", "independent_clearance"):
        for route in ("IV", "PO"):
            groups = {}
            for mode in FLOW_MODES:
                rows = [r for r in report["results"] if
                        (r["scenario"], r["route"], r["flow"]) == (scenario, route, mode)]
                groups[mode] = rows
                auc = summarize(rows, "auc")
                summary.append({"scenario": scenario, "route": route, "flow": mode,
                                "auc": auc, "sampled_cmax": summarize(rows, "cmax"),
                                "auc_within_2x_count": sum(0.5 <= r["auc_pred"] / r["auc_obs"] <= 2
                                                            for r in rows),
                                "median_profile_aafe": float(np.median([r["aafe"] for r in rows]))})
            pulse = groups["heart_pulsatile"]
            steady = groups["heart_steady"]
            assert [r["dataset"] for r in pulse] == [r["dataset"] for r in steady]
            paired = np.array([abs(a["auc_pred"] / b["auc_pred"] - 1) for a, b in zip(pulse, steady)])
            summary[-1]["absolute_auc_change_vs_heart_steady"] = {
                "median": float(np.median(paired)), "max": float(paired.max()),
            }
    report["summary"] = summary
    report["independent_vs_iv_calibrated"] = {
        route: paired_study_comparison(report["results"], route) for route in ("IV", "PO")
    }
    return report


def plot_panel(report):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True, layout="constrained")
    scenarios = (("bottom_up", "No fit to this panel"),
                 ("iv_calibrated", "Clearance fitted on panel IV data"),
                 ("independent_clearance", "Clearance from a separate study"))
    for ax, (scenario, title) in zip(axes, scenarios):
        ax.axhspan(0.5, 2.0, color="#ddeeea")
        ax.axhline(1, color="#334155", linewidth=1)
        labels = []
        for i, (route, label) in enumerate((("IV", "IV"), ("PO", "Oral"))):
            rows = [r for r in report["results"] if
                    (r["scenario"], r["route"], r["flow"]) == (scenario, route, "heart_pulsatile")]
            ratio = np.sort([r["auc_pred"] / r["auc_obs"] for r in rows])
            color = ("#3265a8", "#008479")[i]
            offset = np.resize(np.linspace(-0.16, 0.16, 7), len(ratio))
            ax.scatter(i + offset, ratio, color=color, s=23, alpha=0.8)
            ax.hlines(np.median(ratio), i - 0.24, i + 0.24, color=color, linewidth=2.5)
            inside = np.count_nonzero((ratio >= 0.5) & (ratio <= 2.0))
            note = " (held out)" if scenario == "iv_calibrated" and route == "PO" else ""
            labels.append(f"{label}{note}\n{inside}/{len(ratio)} within 2×")
        ax.set(yscale="log", xlim=(-0.55, 1.55), xticks=[0, 1], xticklabels=labels, title=title)
        ax.set_yticks([0.25, 0.5, 1.0, 2.0, 3.0], ["0.25×", "0.5×", "1×", "2×", "3×"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted / observed midazolam AUC")
    fig.suptitle(f"Pulsatile model vs published plasma curves from {report['studies']} studies",
                 fontsize=15, fontweight="bold")
    fig.savefig(DOCS / "img" / "circulation_clinical_panel.png", dpi=180)
    plt.close(fig)


def main():
    report = aggregate(evaluate_panel())
    target = DOCS / "validation_circulation_panel.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    plot_panel(report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
