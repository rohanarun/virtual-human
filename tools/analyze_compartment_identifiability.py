# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# See docs/verification-tooling.md for provenance and local changes.

"""Exact arithmetic on invented cell-state proportions; no study inputs.

This demonstrates ambiguity in a snapshot summary, not a fitted biological
model. Run from the repository root; JSON goes to stdout. No network or files.
"""

from fractions import Fraction as F
from itertools import product
import json


def compartment_fraction(blood_fraction, selection_ratio, conversion):
    """Select high/low states, then convert a fraction of selected low cells."""
    high = selection_ratio * blood_fraction
    low = 1 - blood_fraction
    return (high + conversion * low) / (high + low)


def sign_flip_tail(effects, two_sided):
    """Enumerate a sign-flip reference distribution for a fixed sum statistic."""
    observed = sum(effects)
    values = [sum(sign * value for sign, value in zip(signs, effects))
              for signs in product((-1, 1), repeat=len(effects))]
    extreme = sum(abs(value) >= abs(observed) if two_sided else value >= observed
                  for value in values)
    return F(extreme, len(values))


def main():
    p, q = F(1, 5), F(4, 5)  # Invented; NOT the paper's effect or direction.
    mechanisms = []
    for conversion in (F(0), F(1, 4), F(1, 2), F(3, 4)):
        selection = (q - conversion) * (1 - p) / (p * (1 - q))
        actual = compartment_fraction(p, selection, conversion)
        if actual != q:
            raise RuntimeError("Counterexample does not reproduce its target")
        mechanisms.append({"selection_ratio": str(selection),
                           "low_to_high_conversion": str(conversion),
                           "resulting_csf_high_fraction": str(actual)})

    # Independent count construction checks the two endpoint mechanisms.
    # Starting low/high weights 80/20: selection -> 80/320 -> 20/80;
    # conversion -> (80 - 60)/(20 + 60) -> 20/80.
    if F(320, 80 + 320) != q or F(20 + 60, 80 + 20) != q:
        raise RuntimeError("Endpoint count check failed")

    tails = []
    for count in (3, 6):
        # Positive, distinct, arbitrary effects give the smallest possible tails.
        effects = tuple(range(1, count + 1))
        one = sign_flip_tail(effects, False)
        two = sign_flip_tail(effects, True)
        if one != F(1, 2**count) or two != F(2, 2**count):
            raise RuntimeError("Enumeration and theoretical tail bound disagree")
        tails.append({"independent_pairs": count, "sign_patterns": 2**count,
                      "minimum_one_sided_tail": str(one),
                      "minimum_two_sided_tail": str(two)})

    # A clone with true blood frequency f can be missed in n independent draws.
    # Illustrative sampling only; real capture need not satisfy this model.
    miss = (1 - F(1, 1000)) ** 1000
    # Hypothetical independently established 1 <= r <= 4; NOT a measured bound.
    conversion_bounds = [q - r * p * (1 - q) / (1 - p) for r in (F(4), F(1))]
    if conversion_bounds != [F(3, 5), F(3, 4)]:
        raise RuntimeError("Partial-identification bound check failed")
    print(json.dumps({
        "schema_version": 1,
        "status": "executed_synthetic_exact_arithmetic_not_study_analysis",
        "blood_high_fraction": str(p), "target_csf_high_fraction": str(q),
        "observationally_equivalent_mechanisms": mechanisms,
        "exact_sign_flip_resolution": tails,
        "hypothetical_selection_bound": {"selection_min": "1", "selection_max": "4",
                                         "compatible_conversion_min": str(conversion_bounds[0]),
                                         "compatible_conversion_max": str(conversion_bounds[1])},
        "illustrative_missing_clone": {"true_frequency": "1/1000",
                                       "independent_draws": 1000,
                                       "probability_not_detected": round(float(miss), 9)},
    }, indent=2))


if __name__ == "__main__":
    main()
