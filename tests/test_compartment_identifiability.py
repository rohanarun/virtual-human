# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# See docs/verification-tooling.md for provenance and local changes.

"""Smoke checks for the ported synthetic identifiability arithmetic."""
import json
from fractions import Fraction
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_compartment_identifiability as ci


class CompartmentIdentifiabilityTests(unittest.TestCase):
    def test_sign_flip_enumeration_matches_worst_case_tail(self):
        for count in (3, 6):
            effects = tuple(range(1, count + 1))
            self.assertEqual(ci.sign_flip_tail(effects, False), Fraction(1, 2 ** count))
            self.assertEqual(ci.sign_flip_tail(effects, True), Fraction(2, 2 ** count))

    def test_main_emits_synthetic_only_marker(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/analyze_compartment_identifiability.py")],
            capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["status"],
                         "executed_synthetic_exact_arithmetic_not_study_analysis")
        self.assertEqual(len(payload["observationally_equivalent_mechanisms"]), 4)


if __name__ == "__main__":
    unittest.main()
