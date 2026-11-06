"""The molecular -> body bridge: target engagement over time.

The body model gives the unbound drug concentration next to the target
(e.g. brain interstitial water for a CNS receptor). Molecular-scale binding
parameters turn that into the fraction of receptors occupied:

    equilibrium:  occupancy = Cu / (Cu + Kd)
    kinetic:      dO/dt = kon * Cu * (1 - O) - koff * O,   kon = koff / Kd

Kd is exactly the quantity an atomistic binding free-energy calculation
produces: Kd = c0 * exp(dG_bind / RT) with c0 = 1 M. Today Kd comes from
experiment; the roadmap computes it with vh.molecular (alchemical free
energy), at which point this module becomes the hand-off point.
"""

from __future__ import annotations

import numpy as np

from ..constants import KB


def kd_from_binding_free_energy(dg_kj_mol: float, temperature_k: float = 310.15) -> float:
    """Binding free energy (kJ/mol, negative = favorable) -> Kd in nM (1 M standard state)."""
    return float(np.exp(dg_kj_mol / (KB * temperature_k)) * 1e9)


def binding_free_energy_from_kd(kd_nm: float, temperature_k: float = 310.15) -> float:
    return float(KB * temperature_k * np.log(kd_nm * 1e-9))


def mg_per_l_to_nm(c_mg_l, mw_g_mol):
    return np.asarray(c_mg_l) / mw_g_mol * 1e6


def equilibrium_occupancy(cu_nm, kd_nm):
    cu = np.asarray(cu_nm, dtype=float)
    return cu / (cu + kd_nm)


def competitive_equilibrium_occupancy(cu_a_nm, kd_a_nm: float, cu_b_nm, kd_b_nm: float) -> dict:
    """Exact Gaddum competitive equilibrium for two ligands sharing one receptor or groove.

    Given unbound concentrations Cu_A, Cu_B >= 0 and dissociation constants Kd_A, Kd_B > 0:
        O_A = (Cu_A / Kd_A) / (1 + Cu_A / Kd_A + Cu_B / Kd_B)
        O_B = (Cu_B / Kd_B) / (1 + Cu_A / Kd_A + Cu_B / Kd_B)
        O_total = O_A + O_B
    """
    if kd_a_nm <= 0 or kd_b_nm <= 0 or not (np.isfinite(kd_a_nm) and np.isfinite(kd_b_nm)):
        raise ValueError("dissociation constants kd_a_nm and kd_b_nm must be finite and positive")
    ca = np.asarray(cu_a_nm, dtype=float)
    cb = np.asarray(cu_b_nm, dtype=float)
    if np.any(~np.isfinite(ca)) or np.any(~np.isfinite(cb)) or np.any(ca < 0) or np.any(cb < 0):
        raise ValueError("unbound concentrations must be finite and nonnegative")
    alpha = ca / float(kd_a_nm)
    beta = cb / float(kd_b_nm)
    denom = 1.0 + alpha + beta
    occ_a = alpha / denom
    occ_b = beta / denom
    occ_total = (alpha + beta) / denom
    if np.ndim(ca) == 0 and np.ndim(cb) == 0:
        return {
            "occupancy_a": float(occ_a),
            "occupancy_b": float(occ_b),
            "occupancy_total": float(occ_total),
            "free_receptor_fraction": float(1.0 / denom),
        }
    return {
        "occupancy_a": occ_a,
        "occupancy_b": occ_b,
        "occupancy_total": occ_total,
        "free_receptor_fraction": 1.0 / denom,
    }


def shared_core_identifiability_locus(
    target_total_occupancy: float,
    kd_a_nm: float,
    kd_b_nm: float,
    cu_a_nm: float,
) -> float:
    """Concentration Cu_B of ligand B yielding target_total_occupancy alongside Cu_A.

    When two cross-reactive ligands (for example paralogous peptides sharing a 9-mer
    MHC-II binding core) contribute equally to a bulk occupancy or activation readout
    O_total = (Cu_A/Kd_A + Cu_B/Kd_B) / (1 + Cu_A/Kd_A + Cu_B/Kd_B), any pair (Cu_A, Cu_B)
    on the affine locus Cu_A/Kd_A + Cu_B/Kd_B = O_total / (1 - O_total) produces the
    identical total occupancy.
    """
    if not (0.0 <= target_total_occupancy < 1.0 and np.isfinite(target_total_occupancy)):
        raise ValueError("target_total_occupancy must be finite and in [0, 1)")
    if kd_a_nm <= 0 or kd_b_nm <= 0 or not (np.isfinite(kd_a_nm) and np.isfinite(kd_b_nm)):
        raise ValueError("kd_a_nm and kd_b_nm must be finite and positive")
    if cu_a_nm < 0 or not np.isfinite(cu_a_nm):
        raise ValueError("cu_a_nm must be finite and nonnegative")
    total_normalized_affinity = target_total_occupancy / (1.0 - target_total_occupancy)
    remaining_affinity = total_normalized_affinity - (cu_a_nm / kd_a_nm)
    if remaining_affinity < -1e-15:
        raise ValueError("cu_a_nm alone exceeds target_total_occupancy")
    return float(max(0.0, remaining_affinity) * kd_b_nm)


def competitive_kinetic_occupancy(
    t_h,
    cu_a_nm,
    kd_a_nm: float,
    koff_a_per_h: float,
    cu_b_nm,
    kd_b_nm: float,
    koff_b_per_h: float,
    oa0: float = 0.0,
    ob0: float = 0.0,
) -> dict:
    """Exact piecewise 2x2 matrix-exponential (Motulsky-Mahan) solution for competitive kinetics.

    Solves on each interval [t_{i-1}, t_i] with midpoint unbound concentrations:
        dO_A/dt = kon_A * Cu_A * (1 - O_A - O_B) - koff_A * O_A
        dO_B/dt = kon_B * Cu_B * (1 - O_A - O_B) - koff_B * O_B
    where kon_A = koff_A / Kd_A and kon_B = koff_B / Kd_B.
    """
    if kd_a_nm <= 0 or kd_b_nm <= 0 or koff_a_per_h <= 0 or koff_b_per_h <= 0:
        raise ValueError("Kd and koff parameters must be positive")
    if oa0 < 0 or ob0 < 0 or oa0 + ob0 > 1.0 + 1e-12:
        raise ValueError("initial occupancies must be nonnegative and sum to at most 1")
    t = np.asarray(t_h, dtype=float)
    if t.ndim != 1 or len(t) == 0 or np.any(np.diff(t) < 0):
        raise ValueError("t_h must be a nonempty 1D nondecreasing array")
    ca = np.broadcast_to(np.asarray(cu_a_nm, dtype=float), t.shape)
    cb = np.broadcast_to(np.asarray(cu_b_nm, dtype=float), t.shape)
    if np.any(ca < 0) or np.any(cb < 0) or np.any(~np.isfinite(ca)) or np.any(~np.isfinite(cb)):
        raise ValueError("unbound concentrations must be finite and nonnegative")

    kon_a = float(koff_a_per_h) / float(kd_a_nm)
    kon_b = float(koff_b_per_h) / float(kd_b_nm)
    k2 = float(koff_a_per_h)
    k4 = float(koff_b_per_h)

    oa = np.empty_like(t)
    ob = np.empty_like(t)
    oa[0] = float(oa0)
    ob[0] = float(ob0)

    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        if dt == 0.0:
            oa[i] = oa[i - 1]
            ob[i] = ob[i - 1]
            continue
        c_a = 0.5 * float(ca[i] + ca[i - 1])
        c_b = 0.5 * float(cb[i] + cb[i - 1])
        k1 = kon_a * c_a
        k3 = kon_b * c_b
        m11 = k1 + k2
        m12 = k1
        m21 = k3
        m22 = k3 + k4
        det = k1 * k4 + k2 * k3 + k2 * k4
        oa_inf = (k1 * k4) / det
        ob_inf = (k2 * k3) / det

        tau = 0.5 * (m11 + m22)
        half_diff = 0.5 * (m11 - m22)
        delta = float(np.sqrt(half_diff * half_diff + m12 * m21))
        lam_minus = tau - delta
        lam_plus = tau + delta
        e_minus = float(np.exp(-lam_minus * dt))
        e_plus = float(np.exp(-lam_plus * dt))
        c0 = 0.5 * (e_minus + e_plus)
        if delta * dt >= 1e-8:
            s_coeff = (e_minus - e_plus) / (2.0 * delta)
        else:
            s_coeff = float(np.exp(-tau * dt)) * dt * (1.0 - (delta * dt) ** 2 / 6.0)

        da0 = oa[i - 1] - oa_inf
        db0 = ob[i - 1] - ob_inf
        # exp(-M dt) = c0 * I - s_coeff * (M - tau * I)
        oa[i] = oa_inf + (c0 - s_coeff * half_diff) * da0 - (s_coeff * m12) * db0
        ob[i] = ob_inf - (s_coeff * m21) * da0 + (c0 + s_coeff * half_diff) * db0

    return {
        "time_h": t,
        "occupancy_a": oa,
        "occupancy_b": ob,
        "occupancy_total": oa + ob,
    }


def kinetic_occupancy(t_h, cu_nm, kd_nm, koff_per_h, o0=0.0):
    """Exact piecewise solution assuming Cu is constant within each interval."""
    t = np.asarray(t_h, dtype=float)
    cu = np.asarray(cu_nm, dtype=float)
    kon = koff_per_h / kd_nm
    o = np.empty_like(t)
    o[0] = o0
    for i in range(1, len(t)):
        c = 0.5 * (cu[i] + cu[i - 1])
        k = kon * c + koff_per_h
        o_inf = kon * c / k
        o[i] = o_inf + (o[i - 1] - o_inf) * np.exp(-k * (t[i] - t[i - 1]))
    return o


def target_engagement(model, sim, koff_per_h: float | None = None) -> dict:
    """Unbound concentration at the compound's target tissue and receptor occupancy."""
    tgt = model.compound.target
    if tgt is None:
        raise ValueError(f"{model.compound.name} has no target defined")
    p = model.p
    c_tissue = sim[tgt.tissue]
    cu = c_tissue * p.fu / p.kp[tgt.tissue]             # unbound, in equilibrium with plasma water
    cu_nm = mg_per_l_to_nm(cu, model.compound.mw)
    occ = (equilibrium_occupancy(cu_nm, tgt.kd_nm) if koff_per_h is None
           else kinetic_occupancy(sim["time_h"], cu_nm, tgt.kd_nm, koff_per_h))
    return {"time_h": sim["time_h"], "cu_nm": cu_nm, "occupancy": occ, "target": tgt.name, "kd_nm": tgt.kd_nm,
            "dg_bind_kj_mol": binding_free_energy_from_kd(tgt.kd_nm)}
