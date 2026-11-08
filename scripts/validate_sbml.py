"""SBML import validation: exact structure, an independent referee, and a survey.

For each vendored BioModels file (data/sbml, CC0) this records the exact
conservation laws and their initial totals, positive-mass consistency with
exact certificates, and agreement with libRoadRunner (libSBML + CVODE) on
trajectories, the conservation space, steady states, reduced Jacobian
eigenvalues and, for oscillators, the limit-cycle period.

With ``--survey DIR`` it also reads every BIOMD*.xml in DIR and records which
are refused (and why) and which fail positive-mass consistency, with the exact
certificate. Survey files are identified by SHA-256; they are not vendored.

    PYTHONPATH=. python scripts/validate_sbml.py [--survey DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vh.graph import Graph
from vh.graph import stoichiometry as S
from vh.graph.render import to_mermaid
from vh.graph.sbml import SBMLError, UnsupportedSBML, read_sbml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sbml"
OUT = ROOT / "auto-research" / "2026-11-05-E-sbml-import"
IMG = ROOT / "docs" / "img"

CASES = {  # trajectory horizon, presimulation before the steady-state solve, oscillator?
    "BIOMD0000000009": (200.0, 2000.0, False),
    "BIOMD0000000010": (4000.0, 4000.0, True),
    "BIOMD0000000012": (1000.0, 1000.0, True),
}


def _period(t, x):
    """Mean interval between upward mean-crossings over the second half."""
    half = len(t) // 2
    tt, xx = t[half:], x[half:] - x[half:].mean()
    up = np.nonzero((xx[:-1] < 0) & (xx[1:] >= 0))[0]
    cross = tt[up] - xx[up] * (tt[up + 1] - tt[up]) / (xx[up + 1] - xx[up])
    gaps = np.diff(cross)
    return float(gaps.mean()), float(gaps.std()), int(len(gaps))


def _referee(path):
    import roadrunner

    rr = roadrunner.RoadRunner(str(path))
    rr.integrator.relative_tolerance = 1e-12
    rr.integrator.absolute_tolerance = 1e-16
    rr.integrator.maximum_num_steps = 100000
    return rr, roadrunner.__version__


def validate_model(mid: str) -> dict:
    from scipy.optimize import linear_sum_assignment

    path = DATA / f"{mid}.xml"
    m = read_sbml(path)
    horizon, settle, oscillator = CASES[mid]
    ids = m.dynamic_species
    n = m.stoichiometry_matrix()
    laws = m.conservation_laws()
    x0 = dict(zip(ids, m.initial_amounts()))
    cons = m.stoichiometric_consistency()
    g, schema = m.to_graph()
    row = {
        "id": mid, "name": m.name, "sha256": m.source["sha256"], "sbml": f"L{m.level}V{m.version}",
        "species": len(m.species), "dynamic_species": len(ids), "reactions": len(m.reactions),
        "assignment_rules": len(m.rules),
        "conservation_laws": [" + ".join(s if c == 1 else f"{c}*{s}" for s, c in law.items()) for law in laws],
        "conserved_totals_initial_amount": [float(sum(c * x0[s] for s, c in law.items())) for law in laws],
        "conserved_totals_initial_concentration": [
            float(sum(c * x0[s] for s, c in law.items()) / m.compartments[m.species[next(iter(law))].compartment].size)
            if len({m.species[s].compartment for s in law}) == 1 else None for law in laws],
        "exact_residual_l_dot_N": str(S.conservation_residual(S.conservation_laws(n), n)),
        "rank_plus_laws_equals_dynamic_species": S.rank(n) + len(laws) == len(ids),
        "positive_mass_consistent": cons["consistent"], "consistency_certified": cons["certified"],
        "graph_invariant_problems": g.validate(schema),
    }

    rr, version = _referee(path)
    t = np.linspace(0, horizon, 101)
    ref = np.array(rr.simulate(0, horizon, 101, selections=["time"] + ids))
    ours = m.simulate(ref[:, 0])
    traj = ours["amount"]
    row["referee"] = {"engine": f"libRoadRunner {version}",
                      "trajectory_max_error_rel_to_species_max": float(
                          np.max(np.abs(traj - ref[:, 1:]) / np.abs(ref[:, 1:]).max(axis=0)))}
    totals = np.array([[sum(c * traj[k, ids.index(s)] for s, c in law.items()) for law in laws]
                       for k in range(len(t))]) if laws else np.zeros((len(t), 0))
    row["max_relative_drift_of_conserved_totals"] = float(
        np.max(np.abs(totals - totals[0]) / np.abs(totals[0]))) if laws else 0.0

    rr2, _ = _referee(path)
    rr2.conservedMoietyAnalysis = True
    cm = rr2.getConservationMatrix()
    arr = np.array(cm)
    refm = arr[:, [list(cm.colnames).index(s) for s in ids]] if arr.size else np.zeros((0, len(ids)))
    refm = refm[np.abs(refm).sum(axis=1) > 0]
    oursm = np.array([[law.get(s, 0) for s in ids] for law in laws], float).reshape(-1, len(ids))
    rk = np.linalg.matrix_rank
    row["referee"]["conservation_space_equal"] = bool(
        rk(oursm) == rk(refm) == (rk(np.vstack([oursm, refm])) if len(oursm) else 0))

    tail = m.simulate(np.linspace(0, settle, 401))["amount"][200:]
    ss = m.steady_state(tail.mean(axis=0))
    rr2.simulate(0, settle, 3)
    rr2.steadyState()
    rss = np.array([rr2[s] for s in ids])
    ev = m.reduced_eigenvalues(ss["amount"])
    rev = np.array(rr2.getReducedEigenValues())
    cost = np.abs(ev[:, None] - rev[None, :])
    r_i, c_i = linear_sum_assignment(cost)
    row["steady_state"] = {
        "solver_success": ss["success"], "max_abs_rate_of_change": ss["max_abs_derivative"],
        "max_diff_vs_referee_rel_to_max": float(np.max(np.abs(ss["amount"] - rss)) / np.max(np.abs(rss))),
        "eigenvalue_max_diff_vs_referee_rel": float(np.max(cost[r_i, c_i]) / np.max(np.abs(rev))),
        "max_real_eigenvalue": float(ev.real.max()), "stable": bool(ev.real.max() < 0),
    }
    if oscillator:
        t_long = np.linspace(0, 1.5 * horizon if mid == "BIOMD0000000010" else 2 * horizon, 60001)
        k = 0
        ours_p = _period(t_long, m.simulate(t_long)["amount"][:, k])
        rr3, _ = _referee(path)
        ref_p = _period(t_long, np.array(rr3.simulate(0, t_long[-1], len(t_long), selections=["time", ids[k]]))[:, 1])
        row["oscillation"] = {"species": ids[k], "period": ours_p[0], "referee_period": ref_p[0],
                              "relative_difference": abs(ours_p[0] - ref_p[0]) / ref_p[0],
                              "cycles_averaged": ours_p[2], "model_time_units": "as written in the model"}
    return row


SURVEY_HORIZON = 100.0       # model time units; the same window for every surveyed model


SURVEY_TOL = (1e-10, 1e-14)  # (rtol, atol) in both engines
TIGHT_TOL = (1e-12, 1e-16)   # re-run for any model above 1e-6, to separate integration error from semantics


def _survey_referee(path, m, tol=SURVEY_TOL):
    """Worst |ours - libRoadRunner| / species range over [0, SURVEY_HORIZON], 51 points."""
    import roadrunner

    ids = m.dynamic_species
    if not ids:
        return None
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)
    rr = roadrunner.RoadRunner(str(path))
    rr.integrator.relative_tolerance, rr.integrator.absolute_tolerance = tol
    rr.integrator.maximum_num_steps = 500000
    ref = np.array(rr.simulate(0, SURVEY_HORIZON, 51, selections=["time"] + ids))
    ours = m.simulate(ref[:, 0], rtol=tol[0], atol=tol[1])["amount"]
    scale = np.maximum(np.abs(ref[:, 1:]).max(axis=0), 1e-300)
    return float(np.max(np.abs(ours - ref[:, 1:]) / scale))


def survey(directory: Path) -> dict:
    rows, refused = [], {}
    for path in sorted(directory.glob("BIOMD*.xml")):
        data = path.read_bytes()
        entry = {"id": path.stem, "sha256": hashlib.sha256(data).hexdigest()}
        try:
            m = read_sbml(data)
        except UnsupportedSBML as exc:
            entry["status"] = "refused"
            entry["reason"] = str(exc)
            refused[str(exc).split(" '")[0].split(" http")[0]] = refused.get(str(exc).split(" '")[0].split(" http")[0], 0) + 1
            rows.append(entry)
            continue
        except SBMLError as exc:
            entry["status"] = "error"
            entry["reason"] = str(exc)
            rows.append(entry)
            continue
        c = m.stoichiometric_consistency()
        g, schema = m.to_graph()
        graph_flag = any("[stoichiometric-consistency]" in p for p in g.validate(schema))
        entry.update({"status": "read", "name": m.name, "species": len(m.species), "reactions": len(m.reactions),
                      "conservation_laws": len(m.conservation_laws()), "positive_mass_consistent": c["consistent"],
                      "certified": c["certified"], "graph_invariant_agrees": graph_flag == (not c["consistent"])})
        if not c["consistent"]:
            entry["unconserved"] = c["unconserved"]
            entry["leak_certificate"] = c["leak_combination"]
        entry["referee_trajectory_max_error"] = _survey_referee(path, m)
        if entry["referee_trajectory_max_error"] is not None and entry["referee_trajectory_max_error"] > 1e-6:
            entry["referee_trajectory_max_error_tight_tolerance"] = _survey_referee(path, m, TIGHT_TOL)
        rows.append(entry)
    read = [r for r in rows if r["status"] == "read"]
    errs = [r["referee_trajectory_max_error"] for r in read if r["referee_trajectory_max_error"] is not None]
    return {"files": len(rows), "read": len(read), "refused_by_construct": refused,
            "referee": {"horizon": SURVEY_HORIZON, "rtol": SURVEY_TOL[0], "atol": SURVEY_TOL[1],
                        "compared": len(errs), "max_error_rel_to_species_range": max(errs),
                        "median_error": float(np.median(errs)), "within_1e-6": sum(e <= 1e-6 for e in errs),
                        "rerun_at_tight_tolerance": {r["id"]: [r["referee_trajectory_max_error"],
                                                               r["referee_trajectory_max_error_tight_tolerance"]]
                                                     for r in read if "referee_trajectory_max_error_tight_tolerance" in r},
                        "tight_tolerance": {"rtol": TIGHT_TOL[0], "atol": TIGHT_TOL[1]}},
            "errors": [r for r in rows if r["status"] == "error"],
            "inconsistent": [r["id"] for r in read if not r["positive_mass_consistent"]],
            "all_classifications_certified": all(r["certified"] for r in read),
            "graph_invariant_agrees_everywhere": all(r["graph_invariant_agrees"] for r in read),
            "models": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--survey", type=Path, help="directory of BIOMD*.xml files to survey")
    args = ap.parse_args(argv)
    results = {"authority": "exact rational algebra; libRoadRunner as independent SBML referee",
               "models": [validate_model(mid) for mid in CASES]}
    if args.survey:
        results["survey"] = survey(args.survey)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")

    g, _ = read_sbml(DATA / "BIOMD0000000010.xml").to_graph()
    view = Graph(g.name)                       # built in document order, so the output is reproducible
    for node in g.nodes.values():
        if node.type in ("species", "reaction", "moiety"):
            view.add_node(node.id, node.type, node.scale, node.label, **node.attrs)
    for e in g.edges:
        if e.type in ("reactant", "product", "modifier", "member"):
            view.add_edge(e.src, e.dst, e.type, e.directed, e.model, **e.attrs)
    (IMG / "kholodenko_mapk_sbml.mmd").write_text(
        to_mermaid(view, "LR", title="Kholodenko 2000 MAPK cascade imported from SBML (BIOMD0000000010)") + "\n")

    for row in results["models"]:
        print(f"{row['id']} {row['name']}: {len(row['conservation_laws'])} exact laws, "
              f"consistent={row['positive_mass_consistent']} (certified {row['consistency_certified']}); "
              f"trajectory err {row['referee']['trajectory_max_error_rel_to_species_max']:.1e}; "
              f"steady state diff {row['steady_state']['max_diff_vs_referee_rel_to_max']:.1e}, "
              f"eig diff {row['steady_state']['eigenvalue_max_diff_vs_referee_rel']:.1e}, "
              f"stable={row['steady_state']['stable']}")
    if args.survey:
        sv = results["survey"]
        print(f"survey: {sv['read']}/{sv['files']} read; refused {sv['refused_by_construct']}; "
              f"inconsistent {sv['inconsistent']}; certified {sv['all_classifications_certified']}; "
              f"referee {sv['referee']['within_1e-6']}/{sv['referee']['compared']} within 1e-6, "
              f"worst {sv['referee']['max_error_rel_to_species_range']:.1e}")
    print(f"wrote {OUT.relative_to(ROOT)}/results.json and docs/img/kholodenko_mapk_sbml.mmd")


if __name__ == "__main__":
    main()
