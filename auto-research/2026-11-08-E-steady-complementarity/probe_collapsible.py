"""Collapsible (Starling) edges: is the steady solution unique, and does the solver find it?

Edge law in vh.graph.flow (one-way, q >= 0):
    open:   R q = p_src - max(p_dst, p_ext) - h
    closed: q = 0  and  p_src - max(p_dst, p_ext) - h <= 0
Each collapsible edge has three linear regimes:
    C  closed           q = 0,                         drive <= 0
    N  open, normal     R q = p_src - p_dst - h,       p_dst >= p_ext, q >= 0
    W  open, waterfall  R q = p_src - p_ext - h,       p_dst <= p_ext, q >= 0
Enumerating all 3^k regimes and keeping the self-consistent ones gives every
steady solution exactly (the system is piecewise linear).
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np

from vh.graph.flow import FlowNetwork, FlowNodeSpec, VesselSpec

TOL = 1e-9


def random_net(rng):
    n_res, n_junc = int(rng.integers(1, 4)), int(rng.integers(1, 6))
    ids = [f"r{i}" for i in range(n_res)] + [f"j{i}" for i in range(n_junc)]
    nodes = [FlowNodeSpec(ids[i], "reservoir", pressure_pa=float(rng.uniform(0, 2000)))
             for i in range(n_res)]
    nodes += [FlowNodeSpec(ids[n_res + i], "junction",
                           source_m3_s=float(rng.choice([0.0, rng.uniform(-1, 1) * 1e-6])))
              for i in range(n_junc)]
    vessels = []

    def add(a, b, kind):
        kw = {}
        if kind == "valve":
            kw["valve"] = True
        elif kind == "coll":
            kw["collapsible"] = True
            kw["external_pressure_pa"] = float(rng.uniform(0, 2000))
        vessels.append(VesselSpec(f"e{len(vessels)}", a, b, length_m=0.1, radius_m=0.003,
                                  resistance_pa_s_m3=float(10 ** rng.uniform(7, 9)),
                                  gravity_head_pa=0.0, **kw))

    for k in range(1, len(ids)):
        a, b = ids[k], ids[int(rng.integers(0, k))]
        add(*((a, b) if rng.random() < 0.5 else (b, a)), "two")
    for _ in range(int(rng.integers(0, 2))):
        a, b = rng.choice(len(ids), 2, replace=False)
        add(ids[a], ids[b], "two")
    for _ in range(int(rng.integers(0, 3))):
        a, b = rng.choice(len(ids), 2, replace=False)
        add(ids[a], ids[b], "valve")
    for _ in range(int(rng.integers(1, 5))):
        a, b = rng.choice(len(ids), 2, replace=False)
        add(ids[a], ids[b], "coll")
    return nodes, vessels


def solve_regime(net, regime):
    """regime[e] in {'T' two-way, 'O' open valve, 'C' closed, 'N' normal, 'W' waterfall}."""
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
        r = regime[e]
        if r == "C":
            A[row, n + e] = 1.0
        elif r == "W":                 # p_src - R q = p_ext + h
            A[row, net.src[e]] = 1.0
            A[row, n + e] = -net.R[e]
            b[row] = net.external_pressure[e] + net.h[e]
        else:                          # T, O, N: p_src - p_dst - R q = h
            A[row, net.src[e]] = 1.0
            A[row, net.dst[e]] = -1.0
            A[row, n + e] = -net.R[e]
            b[row] = net.h[e]
    x = np.linalg.solve(A, b)
    return x[:n], x[n:]


def consistent(net, p, q):
    """Full steady conditions, checked directly (regime-free)."""
    ps = max(1e-30, float(np.max(np.abs(p))), float(np.max(np.abs(net.external_pressure), initial=0)))
    # flow scale from the physics, not from the answer: a network can carry no flow
    qs = max(float(np.max(np.abs(q))), ps / float(np.median(net.R)),
             float(np.max(np.abs(net.source), initial=0.0)))
    junc = ~net.is_reservoir
    if np.max(np.abs((net.B @ q + net.source)[junc]), initial=0.0) > TOL * qs:
        return False
    p_eff = p[net.dst].copy()
    p_eff[net.collapsible] = np.maximum(p_eff[net.collapsible], net.external_pressure[net.collapsible])
    drive = p[net.src] - p_eff - net.h
    one = net.valve | net.collapsible
    if np.max(np.abs(drive[~one] - net.R[~one] * q[~one]), initial=0.0) > TOL * ps:
        return False
    a = net.R[one] * q[one]
    w = a - drive[one]
    return np.max(np.abs(np.minimum(a, w)), initial=0.0) <= TOL * ps


def enumerate_all(net):
    choices = []
    for e in range(net.m):
        if net.collapsible[e]:
            choices.append("CNW")
        elif net.valve[e]:
            choices.append("CO")
        else:
            choices.append("T")
    sols = []
    for regime in itertools.product(*choices):
        try:
            p, q = solve_regime(net, regime)
        except np.linalg.LinAlgError:
            continue
        if consistent(net, p, q):
            if not any(np.allclose(q, q2, rtol=1e-8, atol=1e-8 * np.max(np.abs(q))) for _, q2 in sols):
                sols.append((p, q))
    return sols


def main(seed, trials, out):
    rng = np.random.default_rng(seed)
    stats = {"trials": trials, "hist_n_solutions": {}, "solver_converged": 0,
             "solver_matches_unique": 0, "solver_wrong_but_converged": 0, "solver_not_converged": 0}
    multi = []
    for t in range(trials):
        nodes, vessels = random_net(rng)
        net = FlowNetwork.from_specs(nodes, vessels)
        sols = enumerate_all(net)
        k = str(len(sols))
        stats["hist_n_solutions"][k] = stats["hist_n_solutions"].get(k, 0) + 1
        sol = net.steady_state()
        p = np.array([sol.pressure[i] for i in net.node_ids])
        q = np.array([sol.flow[i] for i in net.edge_ids])
        ok = consistent(net, p, q)
        stats["solver_converged"] += int(sol.converged)
        if not sol.converged:
            stats["solver_not_converged"] += 1
        elif not ok:
            stats["solver_wrong_but_converged"] += 1
        if len(sols) == 1 and ok and sol.converged:
            qscale = max(float(np.max(np.abs(sols[0][1]))),
                         float(np.max(np.abs(sols[0][0]))) / float(np.median(net.R)))
            stats["solver_matches_unique"] += int(np.max(np.abs(q - sols[0][1])) <= 1e-8 * qscale)
        if len(sols) != 1 and len(multi) < 5:
            multi.append({"trial": t, "n_solutions": len(sols),
                          "q_solutions": [s[1].tolist() for s in sols],
                          "p_solutions": [s[0].tolist() for s in sols],
                          "solver_q": q.tolist(), "solver_converged": sol.converged,
                          "nodes": [n.__dict__ for n in nodes],
                          "vessels": [e.__dict__ for e in vessels]})
    print(json.dumps(stats, indent=1))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "non_unique_examples": multi}, f, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
