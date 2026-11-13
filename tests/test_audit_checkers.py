# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# Adapted: the fixture now uses a small stub ledger because Project Asma's
# committed ledger is not part of this repository. See docs/verification-tooling.md.

"""Integrity audit behaviour with fast, isolated probes."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import audit_checkers as audit


def stub_ledger():
    return {
        "schema_version": 1,
        "ledger_sha256": "0" * 64,
        "counts": {"PROVEN": 0, "VERIFIED": 2, "KNOWN": 0, "CONJECTURE": 0},
        "claims": [
            {"id": "V01", "label": "VERIFIED", "statement": "offline check passes",
             "recheck_commands": ["true"], "citations": []},
            {"id": "V02", "label": "VERIFIED", "statement": "network audit only",
             "recheck_commands": ["example_audit --live"], "citations": []},
        ],
    }


class AuditCheckersTests(unittest.TestCase):
    def test_skip_run_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "claims-ledger.json"
            ledger_path.write_text(json.dumps(stub_ledger()), encoding="utf-8")
            original_ledger = audit.LEDGER
            audit.LEDGER = ledger_path
            try:
                report = audit.build_audit(run_checks=False)
            finally:
                audit.LEDGER = original_ledger
        summary = report["summary"]
        self.assertEqual(summary["verified_claims"], 2)
        self.assertEqual(summary["offline_recheck_defined"], 1)
        self.assertEqual(summary["no_offline_checker"]["network_only"], ["V02"])

    def test_stale_audit_fails_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "claims-ledger.json"
            ledger_path.write_text(json.dumps(stub_ledger()), encoding="utf-8")
            audit_path = Path(tmp) / "audit.json"
            audit_path.write_text(json.dumps({"schema_version": 0}) + "\n", encoding="utf-8")
            original_ledger = audit.LEDGER
            original = audit.AUDIT_JSON
            audit.LEDGER = ledger_path
            audit.AUDIT_JSON = audit_path
            try:
                self.assertEqual(audit.main(["--check", "--skip-run"]), 1)
            finally:
                audit.LEDGER = original_ledger
                audit.AUDIT_JSON = original


if __name__ == "__main__":
    unittest.main()
