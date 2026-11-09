"""Conserved-flow graphs: topology plus constitutive laws for real flow networks.

A :class:`~vh.graph.core.Graph` says *what is connected*.  A **flow graph** adds
the physics that makes the connection quantitative.  Every node is a storage
volume or junction and every edge is a conduit.  In SI units:

    node i:   C_i dp_i/dt = sum(q_in) - sum(q_out) + source_i
    edge e:   L_e dq_e/dt = p_src - p_dst - R_e q_e - rho g (z_dst - z_src)
                            - 1/2 rho q_e |q_e| (1/A_dst^2 - 1/A_src^2)
                            - minor losses - collapse / valve terms

The node equation is conservation of the conserved quantity (volume, charge,
mass, heat).  The edge equation is the integrated momentum/energy balance.  For
an inviscid, steady edge with no losses the edge equation is exactly Bernoulli:

    p_src + 1/2 rho v_src^2 + rho g z_src
        = p_dst + 1/2 rho v_dst^2 + rho g z_dst.

So a flow graph is a lumped, 1-D Navier-Stokes solver whose topology is the
typed graph and whose constitutive laws live on the edges.  The same equations
describe blood in veins, water in pipes, current in a circuit, heat through a
wall, and many other conserved flows; only the names of the potential and flux,
and the constitutive laws, change.

The module provides:

* :func:`poiseuille_resistance`, :func:`tube_compliance`,
  :func:`tube_inertance`, :func:`hydrostatic_head`,
  :func:`kinetic_pressure_drop` -- the vein/vessel constitutive laws.
* :func:`build_flow_graph` -- a typed graph + schema from node/vessel specs.
* :class:`FlowNetwork` -- compile a graph and solve steady state or transients.
* :func:`womersley_impedance` -- the exact rigid-tube pulsatile impedance, used
  as an independent high-fidelity reference for the lumped R-L edge law.
* :func:`bernoulli_head` and :func:`extended_bernoulli_residual` -- the
  derivation, made executable.

Units are SI throughout: pressure Pa, volumetric flow m^3/s, compliance
m^3/Pa, inertance kg/m^4, length m.  Helpers convert to mmHg and L/min for
physiology.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import expm
from scipy.special import jv

from .algorithms import connected_components
from .core import EdgeRule, Graph, Schema

# ---------------------------------------------------------------------------
# constants and units
# ---------------------------------------------------------------------------
RHO_BLOOD = 1060.0          # kg/m^3, normal human blood density
MU_BLOOD = 3.5e-3           # Pa s, apparent viscosity at normal shear rates
G = 9.80665                 # m/s^2
MMHG = 133.322387415        # Pa per mmHg

#: 1 mmHg/(L/min) in Pa s/m^3 (the physiological resistance unit).
MMHG_MIN_L_TO_PA_S_M3 = MMHG / (1.0e-3 / 60.0)   # ~7.999e6


def pa_to_mmhg(p_pa):
    """Pressure in Pa -> mmHg."""
    return np.asarray(p_pa) / MMHG


def mmhg_to_pa(p_mmhg):
    """Pressure in mmHg -> Pa."""
    return np.asarray(p_mmhg) * MMHG


def m3_s_to_l_min(q_m3_s):
    """Volumetric flow m^3/s -> L/min."""
    return np.asarray(q_m3_s) * 60000.0


def l_min_to_m3_s(q_l_min):
    """Volumetric flow L/min -> m^3/s."""
    return np.asarray(q_l_min) / 60000.0


def resistance_si_to_mmhg_min_l(r_pa_s_m3):
    """Hydraulic resistance Pa s/m^3 -> mmHg/(L/min)."""
    return np.asarray(r_pa_s_m3) / MMHG_MIN_L_TO_PA_S_M3


def resistance_mmhg_min_l_to_si(r_mmhg_min_l):
    """Hydraulic resistance mmHg/(L/min) -> Pa s/m^3."""
    return np.asarray(r_mmhg_min_l) * MMHG_MIN_L_TO_PA_S_M3


# ---------------------------------------------------------------------------
# vessel constitutive laws
# ---------------------------------------------------------------------------
def poiseuille_resistance(length_m: float, radius_m: float, mu_pa_s: float = MU_BLOOD) -> float:
    """Hagen-Poiseuille resistance of a circular tube, Pa s/m^3.

    R = 8 mu L / (pi r^4).  This is the steady, laminar, fully developed limit
    of a Newtonian fluid.  Blood is non-Newtonian at low shear; use an apparent
    viscosity appropriate to the vessel if that matters.
    """
    if length_m <= 0 or radius_m <= 0:
        raise ValueError("length and radius must be positive")
    return 8.0 * mu_pa_s * length_m / (np.pi * radius_m ** 4)


def tube_inertance(length_m: float, radius_m: float, rho_kg_m3: float = RHO_BLOOD) -> float:
    """Inertance of the fluid column in a tube, kg/m^4.

    L = rho L / A.  It is the coefficient of dq/dt in the unsteady momentum
    balance: the pressure needed to accelerate the column.
    """
    if length_m <= 0 or radius_m <= 0:
        raise ValueError("length and radius must be positive")
    return rho_kg_m3 * length_m / (np.pi * radius_m ** 2)


def tube_compliance(length_m: float, radius_m: float,
                    wall_thickness_m: float, youngs_modulus_pa: float) -> float:
    """Compliance of a thin-walled elastic tube, m^3/Pa.

    For a thin-walled cylinder under uniform pressure, hoop strain gives
    dr/dp = r^2/(E h), hence

        C = dV/dp = 2 pi r^3 L / (E h).

    This is a linearised, isotropic, thin-wall estimate.  Real veins are
    nonlinear and highly distensible at low pressure; use a measured
    pressure-volume curve when available.
    """
    if length_m <= 0 or radius_m <= 0:
        raise ValueError("length and radius must be positive")
    if wall_thickness_m <= 0 or youngs_modulus_pa <= 0:
        return 0.0
    return 2.0 * np.pi * radius_m ** 3 * length_m / (youngs_modulus_pa * wall_thickness_m)


def hydrostatic_head(z_src_m: float, z_dst_m: float, rho_kg_m3: float = RHO_BLOOD) -> float:
    """Hydrostatic pressure rise from src to dst, Pa: rho g (z_dst - z_src).

    Positive means the destination is higher, so the source pressure must be
    higher by this amount to push the same flow.  A 1 m column of blood is
    about 78 mmHg.
    """
    return rho_kg_m3 * G * (z_dst_m - z_src_m)


def kinetic_pressure_drop(q_m3_s: float, area_src_m2: float, area_dst_m2: float,
                          rho_kg_m3: float = RHO_BLOOD,
                          kinetic_factor: float = 1.0) -> float:
    """Bernoulli kinetic pressure drop for a change of area, Pa.

    p_src - p_dst = 1/2 rho (v_dst^2 - v_src^2)
                  = 1/2 rho q^2 (1/A_dst^2 - 1/A_src^2).

    ``kinetic_factor`` is the kinetic-energy correction factor alpha: 1 for a
    flat profile, 2 for a fully developed parabolic (Poiseuille) profile.
    """
    if area_src_m2 <= 0 or area_dst_m2 <= 0:
        raise ValueError("areas must be positive")
    return kinetic_factor * 0.5 * rho_kg_m3 * q_m3_s * abs(q_m3_s) * (
        1.0 / area_dst_m2 ** 2 - 1.0 / area_src_m2 ** 2
    )


def starling_flow(p_up_pa: float, p_down_pa: float, p_external_pa: float,
                  resistance_pa_s_m3: float) -> float:
    """Flow through a collapsible tube (Starling resistor), m^3/s.

    The simple waterfall law:

        q = max(0, (p_up - max(p_down, p_external)) / R).

    When the downstream pressure falls below the external pressure the tube
    collapses at its downstream end: flow is set by the upstream-to-external
    pressure difference and becomes independent of the downstream pressure.
    This is the mechanism that lets veins limit flow and keep capillary
    pressure from collapsing when you stand up.
    """
    if resistance_pa_s_m3 <= 0:
        raise ValueError("resistance must be positive")
    return max(0.0, (p_up_pa - max(p_down_pa, p_external_pa)) / resistance_pa_s_m3)


def womersley_number(radius_m: float, angular_frequency_rad_s: float,
                     rho_kg_m3: float = RHO_BLOOD, mu_pa_s: float = MU_BLOOD) -> float:
    """Womersley number alpha = r sqrt(omega rho / mu), dimensionless.

    alpha << 1: viscous forces dominate, the profile is quasi-steady and
    Poiseuille-like.  alpha >> 1: inertia dominates, the profile is blunt and
    the flow is boundary-layer limited.
    """
    if angular_frequency_rad_s <= 0:
        return 0.0
    return radius_m * np.sqrt(angular_frequency_rad_s * rho_kg_m3 / mu_pa_s)


def womersley_impedance(radius_m: float, length_m: float, angular_frequency_rad_s: float,
                        mu_pa_s: float = MU_BLOOD, rho_kg_m3: float = RHO_BLOOD) -> complex:
    """Exact Womersley impedance of a rigid circular tube, Pa s/m^3.

    For a sinusoidal pressure gradient at angular frequency omega, the
    cross-sectionally averaged flow is Q = G A F / (i omega rho), with

        F = 1 - 2 J1(z) / (z J0(z)),   z = alpha sqrt(-i),

    and the hydraulic impedance over length L is

        Z = i omega rho L / (A F).

    Limits: Z -> R_Poiseuille as alpha -> 0 and Z -> i omega L_inertance as
    alpha -> infinity.  The exact result is used as an independent reference
    for the lumped R-L edge law (which is the low-alpha approximation).
    """
    if radius_m <= 0 or length_m <= 0:
        raise ValueError("radius and length must be positive")
    if angular_frequency_rad_s <= 0:
        return complex(poiseuille_resistance(length_m, radius_m, mu_pa_s))
    alpha = womersley_number(radius_m, angular_frequency_rad_s, rho_kg_m3, mu_pa_s)
    area = np.pi * radius_m ** 2
    if alpha < 0.1:
        # F = i alpha^2 / 8 + alpha^4 / 48 + O(alpha^6); avoids cancellation
        # in 1 - 2 J1(z)/(z J0(z)) at very small z.
        F = 1j * alpha ** 2 / 8.0 + alpha ** 4 / 48.0
    else:
        z = alpha * np.sqrt(-1j)
        F = 1.0 - 2.0 * jv(1, z) / (z * jv(0, z))
    return 1j * angular_frequency_rad_s * rho_kg_m3 * length_m / (area * F)


def lumped_impedance(radius_m: float, length_m: float, angular_frequency_rad_s: float,
                     mu_pa_s: float = MU_BLOOD, rho_kg_m3: float = RHO_BLOOD) -> complex:
    """The graph's lumped R-L impedance, R + i omega L, Pa s/m^3."""
    return complex(
        poiseuille_resistance(length_m, radius_m, mu_pa_s)
        + 1j * angular_frequency_rad_s * tube_inertance(length_m, radius_m, rho_kg_m3)
    )


# ---------------------------------------------------------------------------
# Bernoulli, made executable
# ---------------------------------------------------------------------------
def bernoulli_head(p_pa, z_m, velocity_m_s, rho_kg_m3: float = RHO_BLOOD,
                   kinetic_factor: float = 1.0):
    """Total head p + alpha 1/2 rho v^2 + rho g z, Pa."""
    return (np.asarray(p_pa) + kinetic_factor * 0.5 * rho_kg_m3 * np.asarray(velocity_m_s) ** 2
            + rho_kg_m3 * G * np.asarray(z_m))


def extended_bernoulli_residual(p1_pa, z1_m, v1_m_s, p2_pa, z2_m, v2_m_s,
                                losses_pa=0.0, unsteady_pa=0.0,
                                rho_kg_m3: float = RHO_BLOOD,
                                kinetic_factor: float = 1.0):
    """Residual of the extended Bernoulli equation, Pa.

    The residual is zero when

        H1 - H2 = losses + unsteady,

    with H = p + alpha 1/2 rho v^2 + rho g z.  ``losses_pa`` is the sum of
    viscous and minor losses; ``unsteady_pa`` is rho integral(dv/dt ds), which
    is L dq/dt for a uniform tube.  Classical Bernoulli is the special case
    losses = unsteady = 0.
    """
    return (bernoulli_head(p1_pa, z1_m, v1_m_s, rho_kg_m3, kinetic_factor)
            - bernoulli_head(p2_pa, z2_m, v2_m_s, rho_kg_m3, kinetic_factor)
            - np.asarray(losses_pa) - np.asarray(unsteady_pa))


# ---------------------------------------------------------------------------
# declarative node / vessel specs
# ---------------------------------------------------------------------------
@dataclass
class FlowNodeSpec:
    """A node in a flow graph.

    kind:
      * ``"reservoir"`` -- fixed pressure (an infinite source/sink);
      * ``"compartment"`` -- finite compliance (a storage volume);
      * ``"junction"`` -- zero compliance (a pure connection).
    """
    id: str
    kind: str = "junction"
    pressure_pa: float | None = None
    compliance_m3_pa: float = 0.0
    volume_m3: float = 0.0
    unstressed_volume_m3: float = 0.0
    external_pressure_pa: float = 0.0
    source_m3_s: float = 0.0
    z_m: float = 0.0
    label: str | None = None

    def __post_init__(self):
        if self.kind not in ("reservoir", "compartment", "junction"):
            raise ValueError(f"unknown node kind {self.kind!r}")
        if self.kind == "reservoir" and self.pressure_pa is None:
            raise ValueError(f"reservoir {self.id!r} needs pressure_pa")
        if self.compliance_m3_pa < 0:
            raise ValueError(f"node {self.id!r} has negative compliance")


@dataclass
class VesselSpec:
    """A conduit (vessel/pipe/wire) between two nodes.

    Geometry is used to derive resistance, inertance, and (optionally)
    compliance when the corresponding explicit value is not given.  A vessel's
    compliance is a *storage* property, so the graph builder assigns it to a
    compartment endpoint (``compliance_to`` = "dst", "src", or "split").
    """
    id: str
    src: str
    dst: str
    length_m: float
    radius_m: float
    wall_thickness_m: float = 0.0
    youngs_modulus_pa: float = 0.0
    resistance_pa_s_m3: float | None = None
    inertance_kg_m4: float | None = None
    compliance_m3_pa: float | None = None
    compliance_to: str = "dst"
    area_src_m2: float | None = None
    area_dst_m2: float | None = None
    kinetic_factor: float = 1.0
    minor_loss_coeff: float = 0.0
    minor_area_m2: float | None = None
    gravity_head_pa: float | None = None
    collapsible: bool = False
    external_pressure_pa: float = 0.0
    valve: bool = False
    label: str | None = None

    def __post_init__(self):
        if self.compliance_to not in ("dst", "src", "split"):
            raise ValueError("compliance_to must be 'dst', 'src', or 'split'")
        if self.length_m <= 0 or self.radius_m <= 0:
            raise ValueError(f"vessel {self.id!r}: length and radius must be positive")


# ---------------------------------------------------------------------------
# graph builder and schema
# ---------------------------------------------------------------------------
def build_flow_graph(nodes: Sequence[FlowNodeSpec], vessels: Sequence[VesselSpec],
                     name: str = "flow") -> tuple[Graph, Schema]:
    """Build a typed flow graph from node and vessel specs.

    Every node becomes a typed node (``reservoir`` / ``compartment`` /
    ``junction``) carrying its pressure, compliance, source, and elevation.
    Every vessel becomes a ``conduit`` edge carrying resistance, inertance,
    gravity head, areas, and optional collapse/valve parameters.  Vessel
    compliance is added to a compartment endpoint according to
    ``compliance_to``.
    """
    g = Graph(name)
    for spec in nodes:
        attrs = dict(
            compliance_m3_pa=float(spec.compliance_m3_pa),
            volume_m3=float(spec.volume_m3),
            unstressed_volume_m3=float(spec.unstressed_volume_m3),
            external_pressure_pa=float(spec.external_pressure_pa),
            source_m3_s=float(spec.source_m3_s),
            z_m=float(spec.z_m),
        )
        if spec.pressure_pa is not None:
            attrs["pressure_pa"] = float(spec.pressure_pa)
        g.add_node(spec.id, spec.kind, "organ", label=spec.label, **attrs)

    for v in vessels:
        R = (v.resistance_pa_s_m3 if v.resistance_pa_s_m3 is not None
             else poiseuille_resistance(v.length_m, v.radius_m))
        L = (v.inertance_kg_m4 if v.inertance_kg_m4 is not None
             else tube_inertance(v.length_m, v.radius_m))
        C = (v.compliance_m3_pa if v.compliance_m3_pa is not None
             else tube_compliance(v.length_m, v.radius_m,
                                  v.wall_thickness_m, v.youngs_modulus_pa))
        z_src = float(g.node(v.src).attrs["z_m"])
        z_dst = float(g.node(v.dst).attrs["z_m"])
        h = (v.gravity_head_pa if v.gravity_head_pa is not None
             else hydrostatic_head(z_src, z_dst))
        area_src = (v.area_src_m2 if v.area_src_m2 is not None
                    else np.pi * v.radius_m ** 2)
        area_dst = (v.area_dst_m2 if v.area_dst_m2 is not None
                    else np.pi * v.radius_m ** 2)
        minor_area = (v.minor_area_m2 if v.minor_area_m2 is not None
                      else min(area_src, area_dst))
        attrs = dict(
            id=v.id,
            resistance_pa_s_m3=float(R),
            conductance=float(1.0 / R) if R > 0 else 0.0,
            inertance_kg_m4=float(L),
            compliance_m3_pa=float(C),
            area_src_m2=float(area_src),
            area_dst_m2=float(area_dst),
            kinetic_factor=float(v.kinetic_factor),
            minor_loss_coeff=float(v.minor_loss_coeff),
            minor_area_m2=float(minor_area),
            gravity_head_pa=float(h),
            gravity_head_override=bool(v.gravity_head_pa is not None),
            collapsible=bool(v.collapsible),
            external_pressure_pa=float(v.external_pressure_pa),
            valve=bool(v.valve),
            length_m=float(v.length_m),
            radius_m=float(v.radius_m),
        )
        g.add_edge(v.src, v.dst, "conduit", model="flow.conduit", **attrs)

        # A vessel's compliance is storage, so it belongs on a compartment
        # node.  Add it to the chosen endpoint(s); ignore reservoir/junction
        # endpoints (their pressure/volume is fixed or algebraic).
        if C > 0:
            targets = []
            if v.compliance_to == "split":
                targets = [(v.src, 0.5), (v.dst, 0.5)]
            elif v.compliance_to == "src":
                targets = [(v.src, 1.0)]
            else:
                targets = [(v.dst, 1.0)]
            added = []
            for nid, share in targets:
                if g.node(nid).type == "compartment":
                    g.node(nid).attrs["compliance_m3_pa"] += C * share
                    added.append(nid)
            if added:
                g.edges[-1].attrs["compliance_added_to"] = ",".join(added)

    schema = flow_schema()
    return g, schema


def flow_schema() -> Schema:
    """Schema and structural invariants for a conserved-flow graph."""
    schema = Schema(
        node_types={
            "reservoir": {"pressure_pa"},
            "compartment": {"compliance_m3_pa"},
            "junction": set(),
        },
        edge_rules=[
            EdgeRule("conduit",
                     {"reservoir", "compartment", "junction"},
                     {"reservoir", "compartment", "junction"},
                     {"resistance_pa_s_m3"}),
        ],
    )

    @schema.invariant("positive-properties")
    def _positive(g: Graph) -> list[str]:
        out = []
        for n in g.nodes.values():
            if float(n.attrs.get("compliance_m3_pa", 0.0)) < 0:
                out.append(f"node {n.id}: negative compliance")
        for e in g.edges:
            if float(e.attrs.get("resistance_pa_s_m3", 0.0)) <= 0:
                out.append(f"edge {e.key}: resistance must be positive")
            if float(e.attrs.get("inertance_kg_m4", 0.0)) < 0:
                out.append(f"edge {e.key}: negative inertance")
        return out

    @schema.invariant("pressure-reference")
    def _reference(g: Graph) -> list[str]:
        out = []
        for comp in connected_components(g, ("conduit",)):
            if not any(g.node(n).type == "reservoir" for n in comp):
                out.append(f"component {sorted(comp)} has no pressure reservoir")
        return out

    @schema.invariant("no-orphan-nodes")
    def _orphans(g: Graph) -> list[str]:
        return [f"node {n.id} has no conduit" for n in g.nodes.values() if g.degree(n.id) == 0]

    @schema.invariant("collapsible-has-external-pressure")
    def _collapse(g: Graph) -> list[str]:
        out = []
        for e in g.edges:
            if e.attrs.get("collapsible") and "external_pressure_pa" not in e.attrs:
                out.append(f"edge {e.key}: collapsible edge needs external_pressure_pa")
        return out

    @schema.invariant("gravity-head-consistent")
    def _gravity(g: Graph) -> list[str]:
        out = []
        for e in g.edges:
            if e.attrs.get("gravity_head_override"):
                continue
            z1 = g.node(e.src).attrs.get("z_m")
            z2 = g.node(e.dst).attrs.get("z_m")
            if z1 is None or z2 is None:
                continue
            expected = hydrostatic_head(float(z1), float(z2))
            actual = float(e.attrs.get("gravity_head_pa", 0.0))
            if abs(actual - expected) > 1e-9 * max(1.0, abs(expected)):
                out.append(f"edge {e.key}: gravity head {actual:.6g} != rho g dz {expected:.6g}")
        return out

    return schema


# ---------------------------------------------------------------------------
# compiled network
# ---------------------------------------------------------------------------
@dataclass
class _Node:
    id: str
    kind: str
    p_fixed: float
    compliance: float
    p0: float
    source: float
    z: float


@dataclass
class _Edge:
    id: str
    src: int
    dst: int
    R: float
    L: float
    h: float
    area_src: float
    area_dst: float
    kinetic_factor: float
    minor_loss: float
    minor_area: float
    collapsible: bool
    external_pressure: float
    valve: bool

    @property
    def has_dynamic(self) -> bool:
        return (np.isfinite(self.area_src) and np.isfinite(self.area_dst)
                and self.area_src > 0 and self.area_dst > 0
                and abs(self.area_src - self.area_dst) > 0)

    @property
    def has_minor(self) -> bool:
        return self.minor_loss > 0 and np.isfinite(self.minor_area) and self.minor_area > 0


@dataclass
class FlowSolution:
    """Steady-state solution of a flow network."""
    pressure: dict[str, float]
    flow: dict[str, float]
    node_residual: dict[str, float]
    edge_residual: dict[str, float]
    converged: bool
    residual_norm: float
    method: str

    def pressure_mmhg(self) -> dict[str, float]:
        return {k: float(pa_to_mmhg(v)) for k, v in self.pressure.items()}

    def flow_l_min(self) -> dict[str, float]:
        return {k: float(m3_s_to_l_min(v)) for k, v in self.flow.items()}

    def summary(self) -> str:
        p = ", ".join(f"{k} {pa_to_mmhg(v):.1f} mmHg" for k, v in self.pressure.items())
        q = ", ".join(f"{k} {m3_s_to_l_min(v):.2f} L/min" for k, v in self.flow.items())
        return f"{self.method}: {p} | {q} | residual {self.residual_norm:.1e}"


@dataclass
class FlowTrajectory:
    """Time series from a transient flow simulation."""
    time_s: np.ndarray
    pressure: np.ndarray          # (n_times, n_nodes)
    flow: np.ndarray              # (n_times, n_edges)
    node_ids: list[str]
    edge_ids: list[str]

    def pressure_at(self, node_id: str) -> np.ndarray:
        return self.pressure[:, self.node_ids.index(node_id)]

    def flow_at(self, edge_id: str) -> np.ndarray:
        return self.flow[:, self.edge_ids.index(edge_id)]

    def final(self) -> FlowSolution:
        n, m = len(self.node_ids), len(self.edge_ids)
        p = {self.node_ids[i]: float(self.pressure[-1, i]) for i in range(n)}
        q = {self.edge_ids[i]: float(self.flow[-1, i]) for i in range(m)}
        return FlowSolution(p, q, {}, {}, True, 0.0, "trajectory")


class FlowNetwork:
    """An executable conserved-flow graph.

    Build it from a typed graph with :meth:`from_graph`, or directly from
    :class:`FlowNodeSpec` / :class:`VesselSpec` lists with :meth:`from_specs`.
    """

    def __init__(self, nodes: Sequence[_Node], edges: Sequence[_Edge]):
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.node_ids = [n.id for n in self.nodes]
        self.edge_ids = [e.id for e in self.edges]
        self.node_index = {nid: i for i, nid in enumerate(self.node_ids)}
        self.edge_index = {eid: i for i, eid in enumerate(self.edge_ids)}
        self.n = len(self.nodes)
        self.m = len(self.edges)

        self.B = np.zeros((self.n, self.m))
        for j, e in enumerate(self.edges):
            self.B[e.src, j] -= 1.0
            self.B[e.dst, j] += 1.0

        self.is_reservoir = np.array([n.kind == "reservoir" for n in self.nodes], dtype=bool)
        self.p_fixed = np.array([n.p_fixed if n.kind == "reservoir" else 0.0
                                 for n in self.nodes], dtype=float)
        self.C = np.array([n.compliance for n in self.nodes], dtype=float)
        self.source = np.array([n.source for n in self.nodes], dtype=float)
        self.z = np.array([n.z for n in self.nodes], dtype=float)
        self.p0 = np.array([n.p0 for n in self.nodes], dtype=float)

        self.R = np.array([e.R for e in self.edges], dtype=float)
        self.L = np.array([e.L for e in self.edges], dtype=float)
        self.h = np.array([e.h for e in self.edges], dtype=float)
        self.area_src = np.array([e.area_src for e in self.edges], dtype=float)
        self.area_dst = np.array([e.area_dst for e in self.edges], dtype=float)
        self.kinetic_factor = np.array([e.kinetic_factor for e in self.edges], dtype=float)
        self.minor_loss = np.array([e.minor_loss for e in self.edges], dtype=float)
        self.minor_area = np.array([e.minor_area for e in self.edges], dtype=float)
        self.collapsible = np.array([e.collapsible for e in self.edges], dtype=bool)
        self.external_pressure = np.array([e.external_pressure for e in self.edges], dtype=float)
        self.valve = np.array([e.valve for e in self.edges], dtype=bool)
        self.src = np.array([e.src for e in self.edges], dtype=int)
        self.dst = np.array([e.dst for e in self.edges], dtype=int)

        if np.any(~np.isfinite(self.R)) or np.any(self.R <= 0):
            raise ValueError("every edge needs a positive finite resistance")
        if not np.any(self.is_reservoir):
            raise ValueError("a flow network needs at least one pressure reservoir")

    # -- construction ------------------------------------------------------
    @classmethod
    def from_specs(cls, nodes: Sequence[FlowNodeSpec], vessels: Sequence[VesselSpec],
                   name: str = "flow") -> "FlowNetwork":
        g, _ = build_flow_graph(nodes, vessels, name)
        return cls.from_graph(g)

    @classmethod
    def from_graph(cls, g: Graph) -> "FlowNetwork":
        node_specs = []
        index = {}
        for i, n in enumerate(g.nodes.values()):
            index[n.id] = i
            kind = n.type
            p_fixed = float(n.attrs.get("pressure_pa", 0.0)) if kind == "reservoir" else 0.0
            C = float(n.attrs.get("compliance_m3_pa", 0.0))
            external = float(n.attrs.get("external_pressure_pa", 0.0))
            if "pressure_pa" in n.attrs:
                p0 = float(n.attrs["pressure_pa"])
            elif C > 0 and "volume_m3" in n.attrs:
                p0 = external + (float(n.attrs["volume_m3"])
                                 - float(n.attrs.get("unstressed_volume_m3", 0.0))) / C
            else:
                p0 = external
            node_specs.append(_Node(
                id=n.id, kind=kind, p_fixed=p_fixed, compliance=C,
                p0=p0, source=float(n.attrs.get("source_m3_s", 0.0)),
                z=float(n.attrs.get("z_m", 0.0)),
            ))
        edge_specs = []
        for e in g.edges:
            if e.type != "conduit":
                continue
            edge_specs.append(_Edge(
                id=str(e.attrs.get("id", f"e{len(edge_specs)}")),
                src=index[e.src], dst=index[e.dst],
                R=float(e.attrs["resistance_pa_s_m3"]),
                L=float(e.attrs.get("inertance_kg_m4", 0.0)),
                h=float(e.attrs.get("gravity_head_pa", 0.0)),
                area_src=float(e.attrs.get("area_src_m2", np.nan)),
                area_dst=float(e.attrs.get("area_dst_m2", np.nan)),
                kinetic_factor=float(e.attrs.get("kinetic_factor", 1.0)),
                minor_loss=float(e.attrs.get("minor_loss_coeff", 0.0)),
                minor_area=float(e.attrs.get("minor_area_m2", np.nan)),
                collapsible=bool(e.attrs.get("collapsible", False)),
                external_pressure=float(e.attrs.get("external_pressure_pa", 0.0)),
                valve=bool(e.attrs.get("valve", False)),
            ))
        return cls(node_specs, edge_specs)

    # -- edge physics ------------------------------------------------------
    def _edge_drop(self, q: np.ndarray) -> np.ndarray:
        """R q + kinetic + minor pressure drop, Pa (no gravity/collapse)."""
        q = np.asarray(q, dtype=float)
        drop = self.R * q
        if self.m:
            dynamic = (np.isfinite(self.area_src) & np.isfinite(self.area_dst)
                       & (self.area_src > 0) & (self.area_dst > 0)
                       & (self.area_src != self.area_dst))
            if np.any(dynamic):
                drop = drop.copy()
                drop[dynamic] += (self.kinetic_factor[dynamic] * 0.5 * RHO_BLOOD
                                  * q[dynamic] * np.abs(q[dynamic])
                                  * (1.0 / self.area_dst[dynamic] ** 2
                                     - 1.0 / self.area_src[dynamic] ** 2))
            minor = (self.minor_loss > 0) & np.isfinite(self.minor_area) & (self.minor_area > 0)
            if np.any(minor):
                drop = drop.copy()
                drop[minor] += (0.5 * RHO_BLOOD * self.minor_loss[minor]
                                * q[minor] * np.abs(q[minor]) / self.minor_area[minor] ** 2)
        return drop

    def _edge_drop_derivative(self, q: np.ndarray) -> np.ndarray:
        """d(edge_drop)/dq for the nonlinear Newton step."""
        q = np.asarray(q, dtype=float)
        d = self.R.copy()
        if self.m:
            dynamic = (np.isfinite(self.area_src) & np.isfinite(self.area_dst)
                       & (self.area_src > 0) & (self.area_dst > 0)
                       & (self.area_src != self.area_dst))
            if np.any(dynamic):
                d[dynamic] += (self.kinetic_factor[dynamic] * RHO_BLOOD
                               * np.abs(q[dynamic])
                               * (1.0 / self.area_dst[dynamic] ** 2
                                  - 1.0 / self.area_src[dynamic] ** 2))
            minor = (self.minor_loss > 0) & np.isfinite(self.minor_area) & (self.minor_area > 0)
            if np.any(minor):
                d[minor] += (RHO_BLOOD * self.minor_loss[minor]
                             * np.abs(q[minor]) / self.minor_area[minor] ** 2)
        return d

    def _effective_downstream(self, p: np.ndarray) -> np.ndarray:
        """Downstream pressure seen by each edge.

        A collapsible tube collapses when its downstream pressure falls below
        the external pressure; the relevant downstream pressure is then the
        external pressure (Starling resistor waterfall).
        """
        p_dst = p[self.dst].copy()
        if self.m and np.any(self.collapsible):
            p_dst[self.collapsible] = np.maximum(
                p_dst[self.collapsible], self.external_pressure[self.collapsible]
            )
        return p_dst

    # -- steady state ------------------------------------------------------
    def _is_linear(self) -> bool:
        if self.m == 0:
            return True
        dynamic = (np.isfinite(self.area_src) & np.isfinite(self.area_dst)
                   & (self.area_src > 0) & (self.area_dst > 0)
                   & (self.area_src != self.area_dst))
        minor = (self.minor_loss > 0) & np.isfinite(self.minor_area) & (self.minor_area > 0)
        return not (np.any(dynamic) or np.any(minor)
                    or np.any(self.collapsible) or np.any(self.valve))

    def _linear_steady(self) -> tuple[np.ndarray, np.ndarray]:
        n, m = self.n, self.m
        A = np.zeros((n + m, n + m))
        b = np.zeros(n + m)
        for i in range(n):
            if self.is_reservoir[i]:
                A[i, i] = 1.0
                b[i] = self.p_fixed[i]
            else:
                A[i, n:] = self.B[i]
                b[i] = -self.source[i]
        for e in range(m):
            row = n + e
            A[row, :n] = self.B[:, e]
            A[row, n + e] = self.R[e]
            b[row] = -self.h[e]
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x[:n], x[n:]

    def _residual_steady(self, x: np.ndarray, open_mask: np.ndarray | None = None) -> np.ndarray:
        n, m = self.n, self.m
        p, q = x[:n], x[n:]
        p_dst = self._effective_downstream(p)
        res = np.empty(n + m)
        res[:n] = np.where(self.is_reservoir, p - self.p_fixed,
                           self.B @ q + self.source)
        res[n:] = p[self.src] - p_dst - self.h - self._edge_drop(q)
        if open_mask is not None:
            closed = (self.valve | self.collapsible) & ~open_mask
            res[n:][closed] = q[closed]
        return res

    # -- scaling ------------------------------------------------------------
    def _scales(self, p: np.ndarray | None = None, q: np.ndarray | None = None) -> tuple[float, float]:
        """Characteristic pressure (Pa) and flow (m^3/s) of the problem.

        Node rows of the residual are flows (m^3/s) and edge rows are pressures
        (Pa); in physiological units they differ by ~9 orders of magnitude, so
        every tolerance is applied to residuals divided by these scales.  When a
        solution (p, q) is given, its own magnitude is included, so the scales
        never fall below the size of the answer.
        """
        p_s = max(float(np.max(np.abs(self.p_fixed[self.is_reservoir]), initial=0.0)),
                  float(np.max(np.abs(self.h), initial=0.0)),
                  float(np.max(np.abs(self.external_pressure), initial=0.0)))
        if p is not None:
            p_s = max(p_s, float(np.max(np.abs(p), initial=0.0)))
        r_s = float(np.median(self.R)) if self.m else 1.0
        q_s = max(float(np.max(np.abs(self.source), initial=0.0)), p_s / r_s)
        if q is not None:
            q_s = max(q_s, float(np.max(np.abs(q), initial=0.0)))
        p_s = max(p_s, q_s * r_s, 1e-300)
        return p_s, max(q_s, 1e-300)

    def _scaled_residual(self, x: np.ndarray, open_mask: np.ndarray,
                         p_s: float, q_s: float) -> np.ndarray:
        n = self.n
        res = self._residual_steady(x, open_mask)
        out = np.empty_like(res)
        out[:n] = np.where(self.is_reservoir, res[:n] / p_s, res[:n] / q_s)
        closed = (self.valve | self.collapsible) & ~open_mask
        out[n:] = np.where(closed, res[n:] / q_s, res[n:] / p_s)
        return out

    def _steady_jacobian(self, x: np.ndarray, open_mask: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of :meth:`_residual_steady` for a fixed open/closed pattern."""
        n, m = self.n, self.m
        p, q = x[:n], x[n:]
        J = np.zeros((n + m, n + m))
        for i in range(n):
            if self.is_reservoir[i]:
                J[i, i] = 1.0
            else:
                J[i, n:] = self.B[i]
        d_drop = self._edge_drop_derivative(q)
        closed = (self.valve | self.collapsible) & ~open_mask
        for e in range(m):
            row = n + e
            if closed[e]:
                J[row, n + e] = 1.0
                continue
            J[row, self.src[e]] += 1.0
            # a collapsed tube (downstream below external) does not feel p_dst
            if not (self.collapsible[e] and p[self.dst[e]] < self.external_pressure[e]):
                J[row, self.dst[e]] -= 1.0
            J[row, n + e] = -d_drop[e]
        return J

    def _solve_active(self, open_mask: np.ndarray, x0: np.ndarray,
                      tol: float = 1e-13, max_iter: int = 60) -> tuple[np.ndarray, bool]:
        """Solve the equations of one open/closed pattern by damped Newton.

        Works on the scaled residual, so the stopping test means the same thing
        for pressures and flows.  For a linear network the first full step is
        exact.  Returns (x, success).
        """
        x = np.asarray(x0, dtype=float).copy()
        p_s, q_s = self._scales(x[:self.n], x[self.n:])
        row_scale = np.r_[np.where(self.is_reservoir, p_s, q_s),
                          np.where((self.valve | self.collapsible) & ~open_mask, q_s, p_s)]
        col_scale = np.r_[np.full(self.n, p_s), np.full(self.m, q_s)]
        r = self._scaled_residual(x, open_mask, p_s, q_s)
        norm = float(np.max(np.abs(r), initial=0.0))
        for _ in range(max_iter):
            if norm < tol:
                return x, True
            J = self._steady_jacobian(x, open_mask) * col_scale[None, :] / row_scale[:, None]
            try:
                dy = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dy, *_ = np.linalg.lstsq(J, -r, rcond=None)
            dx = dy * col_scale
            step = 1.0
            for _ in range(30):
                trial = x + step * dx
                r_trial = self._scaled_residual(trial, open_mask, p_s, q_s)
                n_trial = float(np.max(np.abs(r_trial), initial=0.0))
                if n_trial < norm or n_trial < tol:
                    break
                step *= 0.5
            else:
                return x, False
            x, r, norm = trial, r_trial, n_trial
        return x, norm < tol

    def _one_way_violations(self, x: np.ndarray, open_mask: np.ndarray,
                            p_s: float, q_s: float, tol: float) -> np.ndarray:
        """Indices of one-way edges whose open/closed state contradicts the solution.

        Open edge: must carry q >= 0.  Closed edge: its driving pressure
        p_src - p_dst(effective) - h must be <= 0 (otherwise it would open).
        """
        p, q = x[:self.n], x[self.n:]
        one_way = self.valve | self.collapsible
        drive = p[self.src] - self._effective_downstream(p) - self.h
        bad_open = one_way & open_mask & (q < -tol * q_s)
        bad_closed = one_way & ~open_mask & (drive > tol * p_s)
        return np.nonzero(bad_open | bad_closed)[0]

    def steady_state(self, initial: np.ndarray | None = None,
                     tol: float = 1e-10) -> FlowSolution:
        """Solve the steady state.

        Linear R networks are solved exactly by a square linear system.
        Networks with kinetic terms or minor losses use Newton's method on the
        scaled residual.  Collapsible veins and valves add a complementarity
        condition (flow is one-way, and a closed tube carries no flow):

            q_e >= 0,   blocking pressure w_e >= 0,   q_e * w_e = 0.

        It is solved by single principal pivots with Murty's least-index rule:
        solve the equations of the current open/closed pattern, then flip only
        the lowest-index edge whose state contradicts the solution.  For linear
        edges with valves this is a P-matrix linear complementarity problem
        (the minimum-content program is strictly convex), for which the rule is
        finite and the solution is unique (Murty 1974).

        ``converged`` is True only if the scaled residual is below ``tol`` *and*
        every one-way edge satisfies complementarity; ``residual_norm`` is the
        largest scaled (dimensionless) residual.  Node/edge residuals are in
        physical units (m^3/s and Pa; m^3/s for the q=0 row of a closed edge).
        """
        one_way = self.valve | self.collapsible
        open_mask = np.ones(self.m, dtype=bool)
        pivots_ok = True
        if not np.any(one_way) and self._is_linear():
            p, q = self._linear_steady()
            method = "linear"
        else:
            if initial is None:
                p0, q0 = self._linear_steady()
                x = np.r_[p0, q0]
            else:
                x = np.asarray(initial, dtype=float)
            method = "nonlinear" if not np.any(one_way) else "active-set"
            n_one_way = int(np.count_nonzero(one_way))
            max_pivots = 50 + 10 * n_one_way
            for _ in range(max_pivots + 1):
                x, _ok = self._solve_active(open_mask, x)
                p_s, q_s = self._scales(x[:self.n], x[self.n:])
                bad = self._one_way_violations(x, open_mask, p_s, q_s, tol)
                if bad.size == 0:
                    break
                open_mask[bad[0]] = ~open_mask[bad[0]]      # Murty: least index only
            else:
                pivots_ok = False
            p, q = x[:self.n], x[self.n:]
        x = np.r_[p, q]
        p_s, q_s = self._scales(p, q)
        res = self._residual_steady(x, open_mask)
        res_scaled = self._scaled_residual(x, open_mask, p_s, q_s)
        norm = float(np.max(np.abs(res_scaled), initial=0.0))
        compl_ok = self._one_way_violations(x, open_mask, p_s, q_s, tol).size == 0
        # Closed edges carry q = 0 exactly (a linear row); open one-way edges may
        # carry a negative q only within tolerance, which is clamped for reporting.
        q_out = q.copy()
        if np.any(one_way):
            q_out[one_way] = np.maximum(q_out[one_way], 0.0)
        return FlowSolution(
            pressure={nid: float(p[i]) for i, nid in enumerate(self.node_ids)},
            flow={eid: float(q_out[i]) for i, eid in enumerate(self.edge_ids)},
            node_residual={nid: float(res[i]) for i, nid in enumerate(self.node_ids)},
            edge_residual={eid: float(res[self.n + i]) for i, eid in enumerate(self.edge_ids)},
            converged=bool(norm < tol and compl_ok and pivots_ok),
            residual_norm=norm,
            method=method,
        )

    # -- transients --------------------------------------------------------
    def _initial_state(self, p0, q0, initial: str):
        if p0 is not None:
            p = np.asarray(p0, dtype=float).copy()
        elif initial == "steady":
            sol = self.steady_state()
            p = np.array([sol.pressure[nid] for nid in self.node_ids])
            if q0 is None:
                q0 = np.array([sol.flow[eid] for eid in self.edge_ids])
        elif initial == "zero":
            p = self.p0.copy()
        else:
            p = self.p0.copy()
        if q0 is None:
            q0 = np.zeros(self.m)
        p = np.asarray(p, dtype=float).copy()
        p[self.is_reservoir] = self.p_fixed[self.is_reservoir]
        return p, np.asarray(q0, dtype=float).copy()

    def _linear_step(self, dt: float, p_old: np.ndarray, q_old: np.ndarray):
        n, m = self.n, self.m
        A = np.zeros((n + m, n + m))
        b = np.zeros(n + m)
        for i in range(n):
            if self.is_reservoir[i]:
                A[i, i] = 1.0
                b[i] = self.p_fixed[i]
            else:
                A[i, i] = self.C[i] / dt
                A[i, n:] = -self.B[i]
                b[i] = self.C[i] / dt * p_old[i] + self.source[i]
        for e in range(m):
            row = n + e
            A[row, :n] = self.B[:, e]
            A[row, n + e] = self.L[e] / dt + self.R[e]
            b[row] = self.L[e] / dt * q_old[e] - self.h[e]
        x = np.linalg.solve(A, b)
        return x[:n], x[n:]

    def _nonlinear_step(self, dt: float, p_old: np.ndarray, q_old: np.ndarray,
                        tol: float = 1e-12, max_iter: int = 30):
        if np.any(self.collapsible) or np.any(self.valve):
            raise NotImplementedError(
                "transient collapse/valve edges are not implemented; use the "
                "steady-state solver for Starling-resistor / valve problems"
            )
        n, m = self.n, self.m
        # node rows are flows (m^3/s), edge rows pressures (Pa): test them relative
        # to their own scales, not against one absolute number
        p_s, q_s = self._scales(p_old, q_old)
        row_scale = np.r_[np.where(self.is_reservoir, p_s, q_s), np.full(m, p_s)]

        def residual(x):
            p, q = x[:n], x[n:]
            res = np.empty(n + m)
            res[:n] = np.where(self.is_reservoir, p - self.p_fixed,
                               self.C * (p - p_old) / dt - self.B @ q - self.source)
            res[n:] = (self.L * (q - q_old) / dt + self.B.T @ p + self.h
                       + self._edge_drop(q))
            return res / row_scale

        x = np.r_[p_old, q_old]
        for _ in range(max_iter):
            r = residual(x)
            if np.max(np.abs(r)) < tol:
                return x[:n], x[n:]
            J = np.zeros((n + m, n + m))
            for i in range(n):
                if self.is_reservoir[i]:
                    J[i, i] = 1.0
                else:
                    J[i, i] = self.C[i] / dt
                    J[i, n:] = -self.B[i]
            for e in range(m):
                row = n + e
                J[row, :n] = self.B[:, e]
                J[row, n + e] = self.L[e] / dt + self._edge_drop_derivative(x[n:])[e]
            try:
                dx = np.linalg.solve(J / row_scale[:, None], -r)   # r is row-scaled
            except np.linalg.LinAlgError:
                break
            # mild damping keeps the Newton step physical for large dt
            step = 1.0
            for _ in range(8):
                trial = x + step * dx
                if np.max(np.abs(residual(trial))) < np.max(np.abs(r)):
                    x = trial
                    break
                step *= 0.5
            else:
                x = x + 0.1 * dx
        final = float(np.max(np.abs(residual(x))))
        if final >= tol:
            warnings.warn(f"implicit Euler Newton step did not converge "
                          f"(scaled residual {final:.1e} after {max_iter} iterations)",
                          RuntimeWarning, stacklevel=3)
        return x[:n], x[n:]

    def simulate(self, t_end_s: float, dt_s: float, p0=None, q0=None,
                 initial: str = "steady", exact: bool = False) -> FlowTrajectory:
        """Integrate the network from t=0 to t_end.

        ``initial``: "steady" starts from the steady state, "zero" from the
        stored initial pressures, "given" uses ``self.p0``.  ``exact=True``
        uses the matrix exponential for linear R-C networks (all L=0, all
        non-reservoir nodes have C>0, no nonlinear edges); otherwise implicit
        Euler is used.  Implicit Euler is unconditionally stable and first
        order in dt.
        """
        if dt_s <= 0 or t_end_s <= 0:
            raise ValueError("t_end and dt must be positive")
        p, q = self._initial_state(p0, q0, initial)
        if exact:
            return self._simulate_exact(t_end_s, dt_s, p)
        times = np.arange(0.0, t_end_s + 0.5 * dt_s, dt_s)
        times[-1] = min(times[-1], t_end_s)
        P = np.zeros((len(times), self.n))
        Q = np.zeros((len(times), self.m))
        P[0], Q[0] = p, q
        for k in range(1, len(times)):
            h = times[k] - times[k - 1]
            if self._is_linear():
                p, q = self._linear_step(h, p, q)
            else:
                p, q = self._nonlinear_step(h, p, q)
            P[k], Q[k] = p, q
        return FlowTrajectory(times, P, Q, list(self.node_ids), list(self.edge_ids))

    def _simulate_exact(self, t_end_s: float, dt_s: float, p0: np.ndarray) -> FlowTrajectory:
        """Exact matrix-exponential solution for linear R-C networks (L=0)."""
        if np.any(self.L != 0):
            raise ValueError("exact solver requires all inertances to be zero")
        if np.any(self.collapsible) or np.any(self.valve) or not self._is_linear():
            raise ValueError("exact solver requires linear edges")
        free = np.where(~self.is_reservoir)[0]
        if np.any(self.C[free] <= 0):
            raise ValueError("exact solver requires positive compliance at every non-reservoir node")
        nf = len(free)
        if nf == 0:
            times = np.arange(0.0, t_end_s + 0.5 * dt_s, dt_s)
            times[-1] = min(times[-1], t_end_s)
            P = np.tile(self.p_fixed, (len(times), 1))
            q = ((self.p_fixed[self.src] - self.p_fixed[self.dst] - self.h)
                 / self.R)
            Q = np.tile(q, (len(times), 1))
            return FlowTrajectory(times, P, Q, list(self.node_ids), list(self.edge_ids))
        Bf = self.B[free, :]
        Br = self.B[self.is_reservoir, :]
        Rinv = np.diag(1.0 / self.R)
        Cf = np.diag(self.C[free])
        # C dp/dt = -Bf R^-1 Bf^T p - Bf R^-1 (Br^T p_res + h) + source
        M = -np.linalg.solve(Cf, Bf @ Rinv @ Bf.T)
        c = np.linalg.solve(
            Cf,
            -Bf @ Rinv @ (Br.T @ self.p_fixed[self.is_reservoir] + self.h)
            + self.source[free],
        )
        Aaug = np.zeros((nf + 1, nf + 1))
        Aaug[:nf, :nf] = M
        Aaug[:nf, nf] = c
        x0 = np.r_[p0[free], 1.0]
        times = np.arange(0.0, t_end_s + 0.5 * dt_s, dt_s)
        times[-1] = min(times[-1], t_end_s)
        P = np.zeros((len(times), self.n))
        Q = np.zeros((len(times), self.m))
        for k, t in enumerate(times):
            pf = (expm(Aaug * t) @ x0)[:nf]
            p = np.empty(self.n)
            p[self.is_reservoir] = self.p_fixed[self.is_reservoir]
            p[free] = pf
            P[k] = p
            Q[k] = (p[self.src] - p[self.dst] - self.h) / self.R
        return FlowTrajectory(times, P, Q, list(self.node_ids), list(self.edge_ids))


# ---------------------------------------------------------------------------
# applying a solution back to the graph
# ---------------------------------------------------------------------------
def apply_solution(g: Graph, solution: FlowSolution) -> Graph:
    """Write pressures and flows back onto the graph.

    Node ``pressure_pa``/``pressure_mmhg`` and edge ``flow_m3_s``/``flow_l_min``
    are set.  The edge ``flow`` attribute is in L/min so the existing graph
    algorithms (``flow_balance``, ``widest_path``) can use it directly.
    """
    for nid, p in solution.pressure.items():
        if nid in g.nodes:
            g.node(nid).attrs["pressure_pa"] = float(p)
            g.node(nid).attrs["pressure_mmhg"] = float(pa_to_mmhg(p))
    for e in g.edges:
        eid = e.attrs.get("id")
        if eid in solution.flow:
            q = float(solution.flow[eid])
            e.attrs["flow_m3_s"] = q
            e.attrs["flow_l_min"] = float(m3_s_to_l_min(q))
            e.attrs["flow"] = float(m3_s_to_l_min(q))
    return g
