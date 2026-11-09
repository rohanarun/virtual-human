"""Steady flow with one-way valves: the solution must satisfy the complementarity
(KKT) conditions, and it must equal the exact solution found by enumeration.

Referee: for a linear network the steady problem with ideal valves is the KKT
system of a strictly convex program (minimum content), so it has exactly one
flow solution.  It can be found exactly, without any iteration, by trying every
open/closed pattern of the k valves (2^k linear solves) and keeping the pattern
whose solution satisfies

    q_v >= 0,   w_v = R_v q_v - (p_src - p_dst - h_v) >= 0,   q_v * w_v = 0

for every valve v, plus node balance and the two-way edge laws.  This uses
nothing from the solver under test except the network arrays.

All checks are relative: pressure residuals are divided by the largest |p|,
flow residuals by the largest of |q|, |source| and |p|/median(R).  An absolute tolerance would be meaningless
here because pressures (~1e3 Pa) and flows (~1e-6 m^3/s) differ by ~9 orders.
"""

import itertools

import numpy as np
import pytest

from vh.graph.flow import FlowNetwork, FlowNodeSpec, VesselSpec

REL_TOL = 1e-9


def _vessel(i, a, b, r, valve=False):
    return VesselSpec(i, a, b, length_m=0.1, radius_m=0.003, resistance_pa_s_m3=r,
                      gravity_head_pa=0.0, valve=valve)


def _arrays(net, sol):
    p = np.array([sol.pressure[i] for i in net.node_ids])
    q = np.array([sol.flow[i] for i in net.edge_ids])
    return p, q


def kkt_residuals(net, p, q):
    """Relative violations of the steady KKT conditions (all ~0 at the solution)."""
    pscale = max(1e-30, float(np.max(np.abs(p))))
    # flow scale from the physics as well as the answer: a network may carry no flow
    qscale = max(float(np.max(np.abs(q))), pscale / float(np.median(net.R)),
                 float(np.max(np.abs(net.source), initial=0.0)))
    drive = p[net.src] - p[net.dst] - net.h
    junction = ~net.is_reservoir
    out = {"balance": float(np.max(np.abs((net.B @ q + net.source)[junction]), initial=0.0)) / qscale}
    two_way = ~net.valve
    out["two_way_law"] = float(np.max(np.abs(drive[two_way] - net.R[two_way] * q[two_way]),
                                      initial=0.0)) / pscale
    a = net.R[net.valve] * q[net.valve]                 # >= 0  (flow only forward)
    b = a - drive[net.valve]                            # >= 0  (blocking pressure)
    out["valve_complementarity"] = float(np.max(np.abs(np.minimum(a, b)), initial=0.0)) / pscale
    return out


def _pattern_solve(net, open_mask):
    n, m = net.n, net.m
    A = np.zeros((n + m, n + m))
    rhs = np.zeros(n + m)
    for i in range(n):
        if net.is_reservoir[i]:
            A[i, i] = 1.0
            rhs[i] = net.p_fixed[i]
        else:
            A[i, n:] = net.B[i]
            rhs[i] = -net.source[i]
    for e in range(m):
        row = n + e
        if net.valve[e] and not open_mask[e]:
            A[row, n + e] = 1.0
        else:
            A[row, :n] = net.B[:, e]
            A[row, n + e] = net.R[e]
            rhs[row] = -net.h[e]
    x = np.linalg.solve(A, rhs)
    return x[:n], x[n:]


def exact_by_enumeration(net):
    """All distinct KKT solutions over the 2^k valve patterns (expected: exactly one)."""
    valves = np.nonzero(net.valve)[0]
    found = []
    for bits in itertools.product([False, True], repeat=len(valves)):
        mask = np.ones(net.m, dtype=bool)
        mask[valves] = bits
        try:
            p, q = _pattern_solve(net, mask)
        except np.linalg.LinAlgError:
            continue
        if max(kkt_residuals(net, p, q).values()) < REL_TOL:
            if not any(np.allclose(q, q2, rtol=1e-9, atol=1e-9 * np.max(np.abs(q))) for _, q2 in found):
                found.append((p, q))
    return found


def random_valve_network(rng):
    """Random network whose junctions all reach a reservoir through two-way edges,
    so every valve pattern gives a nonsingular linear system."""
    n_res, n_junc = int(rng.integers(1, 4)), int(rng.integers(1, 7))
    ids = [f"r{i}" for i in range(n_res)] + [f"j{i}" for i in range(n_junc)]
    nodes = [FlowNodeSpec(ids[i], "reservoir", pressure_pa=float(rng.uniform(0, 2000)))
             for i in range(n_res)]
    nodes += [FlowNodeSpec(ids[n_res + i], "junction",
                           source_m3_s=float(rng.choice([0.0, rng.uniform(-1, 1) * 1e-6])))
              for i in range(n_junc)]
    vessels = []

    def add(a, b, valve):
        vessels.append(_vessel(f"e{len(vessels)}", a, b, float(10 ** rng.uniform(7, 9)), valve))

    for k in range(1, len(ids)):                      # spanning tree of two-way edges
        a, b = ids[k], ids[int(rng.integers(0, k))]
        add(*((a, b) if rng.random() < 0.5 else (b, a)), False)
    for _ in range(int(rng.integers(0, 3))):
        a, b = rng.choice(len(ids), 2, replace=False)
        add(ids[a], ids[b], False)
    for _ in range(int(rng.integers(1, 7))):
        a, b = rng.choice(len(ids), 2, replace=False)
        add(ids[a], ids[b], True)
    return nodes, vessels


def regression_network():
    """Found by random search: the original active-set loop reported converged=True
    here with flows wrong by 87% and node balance violated by 22% of the largest flow,
    because its residual mixed m^3/s and Pa under one absolute tolerance."""
    nodes = [
        FlowNodeSpec("r0", "reservoir", pressure_pa=500.0),
        FlowNodeSpec("j0", "junction", source_m3_s=8.7e-7),
        FlowNodeSpec("j1", "junction"),
        FlowNodeSpec("j2", "junction", source_m3_s=4.2e-7),
    ]
    vessels = [
        _vessel("e0", "r0", "j0", 6.7e8),
        _vessel("e1", "j1", "j0", 1.3e8),
        _vessel("e2", "j0", "j2", 5.5e8),
        _vessel("e3", "r0", "j2", 8.4e8),
        _vessel("e4", "j0", "j2", 1.3e8, valve=True),
        _vessel("e5", "j0", "j2", 1.7e7, valve=True),
        _vessel("e6", "r0", "j1", 1.2e7, valve=True),
    ]
    return nodes, vessels


def _check_against_enumeration(net):
    sol = net.steady_state()
    p, q = _arrays(net, sol)
    truth = exact_by_enumeration(net)
    assert len(truth) == 1, "strict convexity: the flow solution is unique"
    pt, qt = truth[0]
    res = kkt_residuals(net, p, q)
    if sol.converged:
        assert max(res.values()) < 1e-8, f"converged=True but KKT violated: {res}"
    assert sol.converged, f"solver did not converge: {res}"
    assert np.max(np.abs(q - qt)) <= 1e-8 * np.max(np.abs(qt))
    assert np.max(np.abs(p - pt)) <= 1e-8 * np.max(np.abs(pt))


def test_regression_network_matches_enumeration():
    net = FlowNetwork.from_specs(*regression_network())
    _check_against_enumeration(net)


@pytest.mark.parametrize("seed", range(4))
def test_random_valve_networks_match_enumeration(seed):
    rng = np.random.default_rng(1000 + seed)
    for _ in range(50):
        net = FlowNetwork.from_specs(*random_valve_network(rng))
        _check_against_enumeration(net)


def test_nonlinear_transient_relaxes_to_the_steady_state():
    """Implicit Euler with minor losses must settle on the steady solution.

    After ~100 time constants the transient is gone to far below 1e-12, so the
    end state must equal the steady-state solve to the solver tolerance.  With an
    absolute 1e-10 stopping test on a residual whose node rows are ~1e-6 m^3/s,
    each Newton step could stop with a flow error of ~1e-4 relative.
    """
    nodes = [FlowNodeSpec("up", "reservoir", pressure_pa=1500.0),
             FlowNodeSpec("c", "compartment", compliance_m3_pa=1e-9),
             FlowNodeSpec("down", "reservoir", pressure_pa=0.0)]
    vessels = [VesselSpec("a", "up", "c", length_m=0.1, radius_m=0.003, resistance_pa_s_m3=2e8,
                          gravity_head_pa=0.0, minor_loss_coeff=50.0, minor_area_m2=1e-6),
               VesselSpec("b", "c", "down", length_m=0.1, radius_m=0.003, resistance_pa_s_m3=1e8,
                          gravity_head_pa=0.0)]
    net = FlowNetwork.from_specs(nodes, vessels)
    steady = net.steady_state()
    assert steady.converged and steady.method == "nonlinear"
    tau = 1e-9 * 1e8 * 2e8 / (1e8 + 2e8)               # linearised R_parallel * C
    traj = net.simulate(t_end_s=100 * tau, dt_s=tau / 5, initial="zero")
    q_end = traj.flow[-1]
    q_ss = np.array([steady.flow[e] for e in net.edge_ids])
    assert np.max(np.abs(q_end - q_ss)) <= 1e-9 * np.max(np.abs(q_ss))
    p_c = traj.pressure_at("c")[-1]
    assert abs(p_c - steady.pressure["c"]) <= 1e-9 * 1500.0


def test_expanding_tube_follows_the_poiseuille_branch():
    """An expansion makes the edge law R q + c q|q| non-monotone (c < 0).

    Between two reservoirs with 0 < dp < R^2 / (4|c|) the steady equation has
    three exact roots (two forward, one reverse); the solver must return the one
    that tends to Poiseuille, q = (R - sqrt(R^2 - 4|c| dp)) / (2|c|).  Above that
    dp no forward root exists at all (see docs/steady_complementarity.md).
    """
    from vh.graph.flow import RHO_BLOOD, poiseuille_resistance
    r1, r2, length, dp = 0.003, 0.006, 0.1, 30.0
    a1, a2 = np.pi * r1 ** 2, np.pi * r2 ** 2
    R = poiseuille_resistance(length, r1)
    c = 0.5 * RHO_BLOOD * (1 / a2 ** 2 - 1 / a1 ** 2)
    assert c < 0 and dp < R ** 2 / (4 * abs(c))
    q_exact = (R - np.sqrt(R ** 2 - 4 * abs(c) * dp)) / (2 * abs(c))
    nodes = [FlowNodeSpec("up", "reservoir", pressure_pa=dp),
             FlowNodeSpec("down", "reservoir", pressure_pa=0.0)]
    vessels = [VesselSpec("e", "up", "down", length_m=length, radius_m=r1,
                          area_src_m2=a1, area_dst_m2=a2, gravity_head_pa=0.0)]
    sol = FlowNetwork.from_specs(nodes, vessels).steady_state()
    assert sol.converged
    assert abs(sol.flow["e"] - q_exact) <= 1e-10 * q_exact


# ---------------------------------------------------------------------------
# collapsible (Starling) edges: three linear regimes per edge
# ---------------------------------------------------------------------------
# A collapsible edge is one-way with q = max(0, (p_src - max(p_dst, p_ext) - h) / R).
# It is piecewise linear with three regimes:
#   C  closed            q = 0
#   N  open, normal      R q = p_src - p_dst - h
#   W  open, waterfall   R q = p_src - p_ext - h
# Enumerating every regime combination and keeping the self-consistent solutions
# finds every steady state exactly.  Uniqueness is NOT implied by the convexity
# argument used for valves (the waterfall flow depends on absolute pressure via
# p_ext), so the enumeration also counts solutions.

def _steady_conditions_hold(net, p, q, tol=REL_TOL):
    """Regime-free check of the full steady conditions, including collapse."""
    pscale = max(1e-30, float(np.max(np.abs(p))),
                 float(np.max(np.abs(net.external_pressure), initial=0.0)))
    qscale = max(float(np.max(np.abs(q))), pscale / float(np.median(net.R)),
                 float(np.max(np.abs(net.source), initial=0.0)))
    junction = ~net.is_reservoir
    if np.max(np.abs((net.B @ q + net.source)[junction]), initial=0.0) > tol * qscale:
        return False
    p_eff = p[net.dst].copy()
    coll = net.collapsible
    p_eff[coll] = np.maximum(p_eff[coll], net.external_pressure[coll])
    drive = p[net.src] - p_eff - net.h
    one_way = net.valve | net.collapsible
    if np.max(np.abs(drive[~one_way] - net.R[~one_way] * q[~one_way]), initial=0.0) > tol * pscale:
        return False
    a = net.R[one_way] * q[one_way]
    return np.max(np.abs(np.minimum(a, a - drive[one_way])), initial=0.0) <= tol * pscale


def _regime_solve(net, regime):
    n, m = net.n, net.m
    A = np.zeros((n + m, n + m))
    rhs = np.zeros(n + m)
    for i in range(n):
        if net.is_reservoir[i]:
            A[i, i] = 1.0
            rhs[i] = net.p_fixed[i]
        else:
            A[i, n:] = net.B[i]
            rhs[i] = -net.source[i]
    for e, r in enumerate(regime):
        row = n + e
        if r == "C":
            A[row, n + e] = 1.0
            continue
        A[row, net.src[e]] = 1.0
        A[row, n + e] = -net.R[e]
        if r == "W":
            rhs[row] = net.external_pressure[e] + net.h[e]
        else:
            A[row, net.dst[e]] = -1.0
            rhs[row] = net.h[e]
    x = np.linalg.solve(A, rhs)
    return x[:n], x[n:]


def all_steady_states(net):
    choices = ["CNW" if net.collapsible[e] else "CO" if net.valve[e] else "T"
               for e in range(net.m)]
    found = []
    for regime in itertools.product(*choices):
        try:
            p, q = _regime_solve(net, regime)
        except np.linalg.LinAlgError:
            continue
        if _steady_conditions_hold(net, p, q):
            qs = max(float(np.max(np.abs(q))), float(np.max(np.abs(p))) / float(np.median(net.R)))
            if not any(np.max(np.abs(q - q2)) <= 1e-8 * qs for _, q2 in found):
                found.append((p, q))
    return found


def random_collapsible_network(rng):
    nodes, vessels = random_valve_network(rng)
    ids = [n.id for n in nodes]
    for _ in range(int(rng.integers(1, 4))):
        a, b = rng.choice(len(ids), 2, replace=False)
        vessels.append(VesselSpec(f"e{len(vessels)}", ids[a], ids[b], length_m=0.1, radius_m=0.003,
                                  resistance_pa_s_m3=float(10 ** rng.uniform(7, 9)),
                                  gravity_head_pa=0.0, collapsible=True,
                                  external_pressure_pa=float(rng.uniform(0, 2000))))
    return nodes, vessels


@pytest.mark.parametrize("seed", range(4))
def test_random_collapsible_networks_match_enumeration(seed):
    rng = np.random.default_rng(2000 + seed)
    for _ in range(25):
        net = FlowNetwork.from_specs(*random_collapsible_network(rng))
        truth = all_steady_states(net)
        assert len(truth) == 1, f"{len(truth)} steady states"
        sol = net.steady_state()
        p, q = _arrays(net, sol)
        assert sol.converged
        assert _steady_conditions_hold(net, p, q)
        pt, qt = truth[0]
        qs = max(float(np.max(np.abs(qt))), float(np.max(np.abs(pt))) / float(np.median(net.R)))
        assert np.max(np.abs(q - qt)) <= 1e-8 * qs
