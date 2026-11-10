"""Analytical clearance calibration for the linear, steady-flow PBPK model."""
from vh.body.pbpk import PBPKModel


def clearance_overrides(model: PBPKModel, systemic_l_h: float,
                        apparent_oral_l_h: float) -> dict[str, float]:
    """Derive hepatic activity and gut escape from reference plasma clearances.

    Use an unscaled reference PBPK model. The returned activity multiplier
    preserves its liver-size and enzyme scaling for other individuals. Gut
    escape is specified independently instead of inheriting the liver multiplier.
    """
    p, ind = model.p, model.ind
    if not 0 < systemic_l_h <= apparent_oral_l_h or p.fa <= 0 or p.clint_hep_l_h <= 0:
        raise ValueError("clearance calibration requires positive clearance, absorption and hepatic activity")

    # Well-stirred organ extraction at the reference flows. The kidney's
    # intrinsic clearance is on a plasma basis, the liver's on an unbound basis.
    kidney_flow = ind.flows["kidney"]
    renal_l_h = p.bp * kidney_flow * p.cl_renal_l_h / (p.bp * kidney_flow + p.cl_renal_l_h)
    hepatic_flow = ind.liver_flow_total
    hepatic_extraction = (systemic_l_h - renal_l_h) / (p.bp * hepatic_flow)
    if not 0 < hepatic_extraction < 1:
        raise ValueError("systemic clearance is outside the reference organ flow limits")
    clint = p.bp * hepatic_flow * hepatic_extraction / (p.fu * (1 - hepatic_extraction))

    bioavailability = systemic_l_h / apparent_oral_l_h
    fg = bioavailability / (p.fa * (1 - hepatic_extraction))
    if not 0 < fg <= 1:
        raise ValueError("oral clearance implies an impossible gut escape fraction")
    return {"clint_scale": clint / p.clint_hep_l_h, "fg": fg}
