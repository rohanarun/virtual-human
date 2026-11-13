# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# Adapted for portability. See docs/verification-tooling.md for provenance.
"""Exact binomial design bounds for zero-false-positive validations.

Answers: with zero false positives observed in n independent negatives, what
specificity lower bound does that certify, and how large must n be to certify
a target lower bound? Offline, deterministic, standard library only.

This is the generic mathematics extracted from Project Asma's
``tools/analyze_validation_capacity.py``. The study-pinned variant (hashes of
pinned registry files, the committed artifact and its drift checks) remains in
Project Asma; none of those pins are meaningful outside that repository.

JSON on stdout. Exit codes: 0 on success, 2 on invalid input.
"""
import argparse
import json
from fractions import Fraction
import math


def positive_integer(n):
    if type(n) is not int or not 1 <= n <= 10000:
        raise ValueError('count must be an integer from 1 through 10000')


def probability(p):
    if type(p) not in (int, float, Fraction) or not math.isfinite(p) or not 0 < p < 1:
        raise ValueError('probability must be finite and strictly between zero and one')


def perfect_lower_bound(n, tail_probability):
    """Exact binomial all-success lower limit, numerically evaluated."""
    positive_integer(n)
    probability(tail_probability)
    return float(tail_probability) ** (1 / n)


def minimum_perfect_n(target, tail_probability):
    """Minimum zero-failure n, using rational comparisons to prove minimality."""
    probability(target)
    probability(tail_probability)
    target = Fraction(str(target))
    tail_probability = Fraction(str(tail_probability))
    for n in range(1, 10001):
        if target ** n <= tail_probability:
            return n
    raise ValueError('required count exceeds supported range')


def specificity_grid(n):
    positive_integer(n)
    return [
        {'false_positives': fp, 'true_negatives': n - fp,
         'specificity_exact': str(Fraction(n - fp, n)),
         'specificity': (n - fp) / n}
        for fp in range(n + 1)
    ]


def build_result(n, target, tail_probability):
    demanded = minimum_perfect_n(target, tail_probability)
    return {
        'zero_false_positive_lower_bound': perfect_lower_bound(n, tail_probability),
        'minimum_n_for_target': demanded,
        'specificity_grid': specificity_grid(n),
        'inputs': {
            'n': n,
            'target_lower_bound': target,
            'tail_probability': tail_probability,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, required=True,
                        help='number of independent negative cases (1..10000)')
    parser.add_argument('--target', type=float, default=0.98,
                        help='desired one-sided lower specificity bound (default 0.98)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='one-sided tail probability (default 0.05)')
    args = parser.parse_args(argv)
    try:
        result = build_result(args.n, args.target, args.alpha)
    except ValueError as error:
        print(f'invalid input: {error}', file=__import__('sys').stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
