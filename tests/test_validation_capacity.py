# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# Adapted: study-pinned drift tests were dropped because the pins only exist in
# Project Asma. See docs/verification-tooling.md for provenance.

"""Ported math checks for the validation-capacity design bounds."""
import json
import subprocess
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_validation_capacity as capacity


class ValidationCapacityTests(unittest.TestCase):
    def test_all_success_binomial_identity(self):
        # At the lower limit, probability of n successes is the tail mass.
        for n in (1, 2, 13, 22, 149, 183, 1000):
            for alpha in (.05, .025):
                self.assertAlmostEqual(capacity.perfect_lower_bound(n, alpha) ** n, alpha, places=13)

    def test_reference_values(self):
        self.assertAlmostEqual(capacity.perfect_lower_bound(13, .05), .7941833348134493)
        self.assertAlmostEqual(capacity.perfect_lower_bound(22, .05), .8726945683451612)
        self.assertAlmostEqual(capacity.perfect_lower_bound(13, .025), .752947361999529)

    def test_exact_minimality(self):
        target = Fraction(49, 50)
        for alpha, expected in ((Fraction(1, 20), 149), (Fraction(1, 40), 183)):
            n = capacity.minimum_perfect_n(target, alpha)
            self.assertEqual(n, expected)
            self.assertLessEqual(target ** n, alpha)
            self.assertGreater(target ** (n - 1), alpha)

    def test_grid_and_unattainable_98_percent(self):
        for n in (13, 22):
            grid = capacity.specificity_grid(n)
            self.assertEqual(len(grid), n + 1)
            self.assertEqual(grid[0]['specificity'], 1)
            self.assertEqual(grid[-1]['specificity'], 0)
            self.assertNotIn(Fraction(49, 50), [Fraction(r['specificity_exact']) for r in grid])
            self.assertEqual([r['false_positives'] + r['true_negatives'] for r in grid], [n] * (n + 1))

    def test_reject_invalid_inputs(self):
        for n in (True, 0, -1, 1.5, '13', 10001):
            with self.assertRaises(ValueError):
                capacity.perfect_lower_bound(n, .05)
        for p in (True, 0, 1, -.1, float('nan'), float('inf'), '.05'):
            with self.assertRaises(ValueError):
                capacity.perfect_lower_bound(13, p)
            with self.assertRaises(ValueError):
                capacity.minimum_perfect_n(p, .05)

    def test_cli_emits_deterministic_json(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'tools/analyze_validation_capacity.py'),
             '--n', '13', '--target', '0.98', '--alpha', '0.05'],
            capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload['minimum_n_for_target'], 149)
        self.assertEqual(payload['inputs']['n'], 13)


if __name__ == '__main__':
    unittest.main()
