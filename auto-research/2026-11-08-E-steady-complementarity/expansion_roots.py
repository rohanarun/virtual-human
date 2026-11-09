"""Single expanding tube between two reservoirs: all steady states of R q + c q|q| = dp (c < 0)."""

import numpy as np

from vh.graph.flow import RHO_BLOOD, FlowNetwork, FlowNodeSpec, VesselSpec, poiseuille_resistance

r1, r2, length = 0.003, 0.006, 0.1
A1, A2 = np.pi * r1 ** 2, np.pi * r2 ** 2
R = poiseuille_resistance(length, r1)
c = 0.5 * RHO_BLOOD * (1 / A2 ** 2 - 1 / A1 ** 2)
a = abs(c)
print(f"R={R:.4e}  c={c:.4e}  q_crit={R / (2 * a):.4e} m3/s ({R / (2 * a) * 6e4:.3f} L/min)  "
      f"dp_max={R ** 2 / (4 * a):.3f} Pa")


def exact_roots(dp):
    roots = []
    disc = R ** 2 - 4 * a * dp
    if disc >= 0:                                     # q > 0:  a q^2 - R q + dp = 0
        roots += [(R - np.sqrt(disc)) / (2 * a), (R + np.sqrt(disc)) / (2 * a)]
    s = (-R + np.sqrt(R ** 2 + 4 * a * dp)) / (2 * a)  # q = -s < 0:  a s^2 + R s - dp = 0 ... see below
    # for q < 0: R q + c q|q| = R q - c q^2 = R q + a q^2 = dp
    disc2 = R ** 2 + 4 * a * dp
    for q in ((-R + np.sqrt(disc2)) / (2 * a), (-R - np.sqrt(disc2)) / (2 * a)):
        if q < 0:
            roots.append(q)
    return sorted(roots)


for dp in (10.0, 30.0, 48.0, 49.0, 100.0, 1000.0):
    nodes = [FlowNodeSpec("up", "reservoir", pressure_pa=dp), FlowNodeSpec("down", "reservoir", pressure_pa=0.0)]
    vessels = [VesselSpec("e", "up", "down", length_m=length, radius_m=r1,
                          area_src_m2=A1, area_dst_m2=A2, gravity_head_pa=0.0)]
    net = FlowNetwork.from_specs(nodes, vessels)
    sol = net.steady_state()
    roots = exact_roots(dp)
    resid = [R * q + c * q * abs(q) - dp for q in roots]
    print(f"dp={dp:7.1f} Pa: exact roots {[f'{q:.4e}' for q in roots]} (residuals {[f'{x:.1e}' for x in resid]})"
          f" | solver q={sol.flow['e']:.4e} converged={sol.converged} method={sol.method}"
          f" | Poiseuille dp/R={dp / R:.4e}")
