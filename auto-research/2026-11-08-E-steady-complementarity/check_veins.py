"""Check the PR's headline venous network against the brute-force exact solution.

build_vein_network is copied verbatim from scripts/validate_flow.py (branch
auto/E-vein-flow-graph, commit 8152583) to avoid importing matplotlib.
"""

import numpy as np

from vh.graph.flow import MMHG, FlowNetwork, FlowNodeSpec, VesselSpec, build_flow_graph
from probe_active_set import brute_force, kkt_violation

VEIN_POSITIONS = {
    "ra": (0.55, 1.05), "ivc": (0.50, 0.95), "iliac": (0.44, 0.88),
    "femoral": (0.38, 0.75), "popliteal": (0.42, 0.48), "calf": (0.36, 0.30),
    "ankle": (0.32, 0.06), "foot": (0.24, 0.02), "saphenous": (0.72, 0.42),
}


def build_vein_network(tilt_deg=90.0, q_cap_l_min=0.5, p_ra_mmhg=5.0):
    s = np.sin(np.deg2rad(tilt_deg))
    z = {name: h * s for name, (_, h) in VEIN_POSITIONS.items()}
    nodes = [
        FlowNodeSpec("ra", "reservoir", pressure_pa=p_ra_mmhg * MMHG, z_m=z["ra"]),
        FlowNodeSpec("ivc", "junction", z_m=z["ivc"]),
        FlowNodeSpec("iliac", "junction", z_m=z["iliac"]),
        FlowNodeSpec("femoral", "junction", z_m=z["femoral"]),
        FlowNodeSpec("popliteal", "junction", z_m=z["popliteal"]),
        FlowNodeSpec("calf", "compartment", compliance_m3_pa=2.0e-8, z_m=z["calf"]),
        FlowNodeSpec("ankle", "compartment", compliance_m3_pa=5.0e-9, z_m=z["ankle"]),
        FlowNodeSpec("foot", "compartment", compliance_m3_pa=5.0e-9, z_m=z["foot"],
                     source_m3_s=q_cap_l_min / 60000.0),
        FlowNodeSpec("saphenous", "junction", z_m=z["saphenous"]),
    ]
    vessels = [
        VesselSpec("foot_ankle", "foot", "ankle", length_m=0.08, radius_m=0.0025, valve=True),
        VesselSpec("ankle_calf", "ankle", "calf", length_m=0.24, radius_m=0.0030, valve=True),
        VesselSpec("calf_popliteal", "calf", "popliteal", length_m=0.15, radius_m=0.0040, valve=True),
        VesselSpec("popliteal_femoral", "popliteal", "femoral", length_m=0.15, radius_m=0.0045, valve=True),
        VesselSpec("femoral_iliac", "femoral", "iliac", length_m=0.15, radius_m=0.0050, valve=True),
        VesselSpec("iliac_ivc", "iliac", "ivc", length_m=0.10, radius_m=0.0060, valve=True),
        VesselSpec("ivc_ra", "ivc", "ra", length_m=0.15, radius_m=0.0080, valve=True),
        VesselSpec("ankle_saphenous", "ankle", "saphenous", length_m=0.20, radius_m=0.0020, valve=True),
        VesselSpec("saphenous_femoral", "saphenous", "femoral", length_m=0.60, radius_m=0.0020, valve=True),
    ]
    return build_flow_graph(nodes, vessels)


for tilt in (0, 30, 60, 90):
    g, _ = build_vein_network(tilt)
    net = FlowNetwork.from_graph(g)
    sol = net.steady_state()
    p = np.array([sol.pressure[i] for i in net.node_ids])
    q = np.array([sol.flow[i] for i in net.edge_ids])
    truth = brute_force(net)
    pt, qt = truth[0]
    foot = net.node_index["foot"]
    print(f"tilt {tilt:2d}: method={sol.method} converged={sol.converged} n_true={len(truth)} "
          f"foot solver={p[foot] / MMHG:.4f} true={pt[foot] / MMHG:.4f} mmHg  "
          f"max|dq|/max|q|={np.max(np.abs(q - qt)) / np.max(np.abs(qt)):.1e}  "
          f"kkt={ {k: f'{v:.1e}' for k, v in kkt_violation(net, p, q).items()} }")
