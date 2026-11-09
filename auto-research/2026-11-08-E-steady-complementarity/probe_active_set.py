"""Probe: does FlowNetwork.steady_state() report converged=True on wrong answers?

Random linear resistor networks with ideal one-way valves.  Every junction is
connected to a reservoir through a spanning tree of plain (two-way) resistors,
so for every open/closed pattern the linear system is nonsingular and the true
solution is unique (strictly convex content, see notes).

For each network we
  * run the solver under test;
  * check the KKT / complementarity conditions independently;
  * find the true solution by brute force over all 2^k valve patterns.
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np

from vh.graph.flow import FlowNetwork, FlowNodeSpec, VesselSpec

TOL_Q = 1e-9      # relative flow tolerance
TOL_P = 1e-6      # Pa


def random_network(rng, n_res, n_junc, n_tree_extra, n_valves):
    """Return (nodes, vessels, meta) for a random network."""
    ids = [f"r{i}" for i in range(n_res)] + [f"j{i}" for i in range(n_junc)]
    nodes = []
    for i in range(n_res):
        nodes.append(FlowNodeSpec(ids[i], "reservoir", pressure_pa=float(rng.uniform(0, 2000))))
    for i in range(n_junc):
        src = float(rng.choice([0.0, 0.0, rng.uniform(-1, 1) * 1e-6]))
        nodes.append(FlowNodeSpec(ids[n_res + i], "junction", source_m3_s=src))
    vessels = []
    k = 0

    def add(a, b, valve):
        nonlocal k
        vessels.append(VesselSpec(f"e{k}", a, b, length_m=0.1, radius_m=0.003,
                                  resistance_pa_s_m3=float(10 ** rng.uniform(7, 9)),
                                  gravity_head_pa=0.0, valve=valve))
        k += 1

    # spanning tree of two-way resistors: each new node attaches to an earlier one
    order = list(range(len(ids)))
    for pos in range(1, len(order)):
        a = ids[order[pos]]
        b = ids[order[int(rng.integers(0, pos))]]
        if rng.random() < 0.5:
            a, b = b, a
        add(a, b, False)
    for _ in range(n_tree_extra):
        a, b = rng.choice(len(ids), 2, replace=False)
        add(ids[a], ids[b], False)
    for _ in range(n_valves):
        a, b = rng.choice(len(ids), 2, replace=False)
        add(ids[a], ids[b], True)
    return nodes, vessels


def linear_solve(net, open_mask):
    """Solve with valves fixed open/closed (closed: q=0). Returns p, q."""
    n, m = net.n, net.m
    A = np.zeros((n + m, n + m))
    b = np.zeros(n + m)
    for i in range(n):
        if net.is_reservoir[i]:
            A[i, i] = 1.0
            b[i] = net.p_fixed[i]
        else:
            A[i, n:] = net.B[i]
            b[i] = -net.source[i]
    for e in range(m):
        row = n + e
        if net.valve[e] and not open_mask[e]:
            A[row, n + e] = 1.0
        else:
            A[row, :n] = net.B[:, e]
            A[row, n + e] = net.R[e]
            b[row] = -net.h[e]
    x = np.linalg.solve(A, b)
    return x[:n], x[n:]


def kkt_violation(net, p, q):
    """Violations of the steady KKT conditions, each normalised to be O(1) when wrong.

    Pressure-like residuals are divided by pscale (largest |p|), flow-like ones by
    qscale (largest |q|), so one relative tolerance serves every check.
    For a valve the complementarity is  q >= 0,  R q - drive >= 0,  q (R q - drive) = 0,
    measured in pressure units as  min(R q, R q - drive)  (Fischer-Burmeister-like).
    """
    qscale = max(1e-30, float(np.max(np.abs(q))))
    pscale = max(1e-30, float(np.max(np.abs(p))))
    drive = p[net.src] - p[net.dst] - net.h          # src minus dst
    v = {}
    bal = net.B @ q + net.source
    junc = ~net.is_reservoir
    v["balance"] = float(np.max(np.abs(bal[junc]))) / qscale if np.any(junc) else 0.0
    tw = ~net.valve
    v["law_twoway"] = float(np.max(np.abs(drive[tw] - net.R[tw] * q[tw]))) / pscale if np.any(tw) else 0.0
    val = net.valve
    if np.any(val):
        a = net.R[val] * q[val]             # >= 0
        b = net.R[val] * q[val] - drive[val]  # >= 0 (blocking pressure of a closed valve)
        v["valve_compl"] = float(np.max(np.abs(np.minimum(a, b)))) / pscale
    else:
        v["valve_compl"] = 0.0
    return v


def is_kkt(v, tol=1e-9):
    return all(x < tol for x in v.values())


def brute_force(net):
    idx = np.nonzero(net.valve)[0]
    sols = []
    for bits in itertools.product([False, True], repeat=len(idx)):
        mask = np.ones(net.m, dtype=bool)
        mask[idx] = bits
        try:
            p, q = linear_solve(net, mask)
        except np.linalg.LinAlgError:
            # a junction cut off from every reservoir: its pressure is not determined
            # by this pattern (only bounded by the closed valves); skip it here
            continue
        if is_kkt(kkt_violation(net, p, q)):
            # a degenerate valve (q = 0 and drive = 0) is a KKT point both open and
            # closed; count distinct solutions, not patterns
            if not any(np.allclose(q, q2, rtol=1e-9, atol=1e-9 * np.max(np.abs(q)))
                       and np.allclose(p, p2, rtol=1e-9, atol=1e-9 * np.max(np.abs(p)))
                       for p2, q2 in sols):
                sols.append((p, q))
    return sols


def main(seed=0, trials=2000):
    rng = np.random.default_rng(seed)
    stats = {"trials": 0, "solver_converged": 0, "solver_wrong_but_converged": 0,
             "solver_wrong_not_converged": 0, "unique_true_solution": 0, "examples": []}
    for t in range(trials):
        n_res = int(rng.integers(1, 4))
        n_junc = int(rng.integers(2, 8))
        n_valves = int(rng.integers(1, 9))
        nodes, vessels = random_network(rng, n_res, n_junc, int(rng.integers(0, 3)), n_valves)
        net = FlowNetwork.from_specs(nodes, vessels)
        sol = net.steady_state()
        p = np.array([sol.pressure[i] for i in net.node_ids])
        q = np.array([sol.flow[i] for i in net.edge_ids])
        v = kkt_violation(net, p, q)
        ok = is_kkt(v)
        truth = brute_force(net)
        stats["trials"] += 1
        stats["unique_true_solution"] += int(len(truth) == 1)
        stats.setdefault("n_true_hist", {})
        stats["n_true_hist"][str(len(truth))] = stats["n_true_hist"].get(str(len(truth)), 0) + 1
        if sol.converged and len(truth) == 1:
            pt, qt = truth[0]
            err = float(np.max(np.abs(q - qt)) / max(1e-30, np.max(np.abs(qt))))
            perr = float(np.max(np.abs(p - pt)) / max(1e-30, np.max(np.abs(pt))))
            stats.setdefault("converged_err_hist", {"<1e-9": 0, "1e-9..1e-6": 0, "1e-6..1e-3": 0, "1e-3..1e-1": 0, ">=1e-1": 0})
            bucket = ("<1e-9" if err < 1e-9 else "1e-9..1e-6" if err < 1e-6 else "1e-6..1e-3" if err < 1e-3
                      else "1e-3..1e-1" if err < 1e-1 else ">=1e-1")
            stats["converged_err_hist"][bucket] += 1
            if err > stats.get("worst_converged", {}).get("q_rel_err", -1):
                stats["worst_converged"] = {
                    "trial": t, "q_rel_err": err, "p_rel_err": perr, "residual_norm": sol.residual_norm,
                    "violations": v, "q_solver": q.tolist(), "q_true": qt.tolist(),
                    "nodes": [n.__dict__ for n in nodes], "vessels": [e.__dict__ for e in vessels]}
        if ok and len(truth) == 1:
            qt = truth[0][1]
            stats.setdefault("max_rel_err_vs_truth_when_ok", 0.0)
            stats["max_rel_err_vs_truth_when_ok"] = max(
                stats["max_rel_err_vs_truth_when_ok"],
                float(np.max(np.abs(q - qt)) / max(1e-30, np.max(np.abs(qt)))))
        stats["solver_converged"] += int(sol.converged)
        if not ok:
            key = "solver_wrong_but_converged" if sol.converged else "solver_wrong_not_converged"
            stats[key] += 1
            if len(stats["examples"]) < 5 and sol.converged:
                stats["examples"].append({
                    "trial": t, "violations": v, "n_true_solutions": len(truth),
                    "nodes": [n.__dict__ for n in nodes],
                    "vessels": [e.__dict__ for e in vessels],
                })
    return stats


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    out = sys.argv[3] if len(sys.argv) > 3 else None
    s = main(seed, trials)
    summary = {k: v for k, v in s.items() if k not in ("examples", "worst_converged")}
    w = s.get("worst_converged", {})
    summary["worst_converged"] = {k: w.get(k) for k in ("trial", "q_rel_err", "p_rel_err", "residual_norm", "violations")}
    print(json.dumps(summary, indent=1))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1, default=float)
        print("examples ->", out)
