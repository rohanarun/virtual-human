# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# See docs/verification-tooling.md for provenance and local changes.

"""Export and stale-check behaviour for the claims ledger."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import export_claims_ledger as ledger


class ExportClaimsLedgerTests(unittest.TestCase):
    def test_same_command_resolves_to_prior_verified_checker(self):
        text = "| V01 | a | `python3 tools/validate.py` |\n| V02 | b | Same command. |\n"
        rows = ledger.parse_claim_rows(text)
        ledger.resolve_same_command(rows)
        self.assertEqual(rows[1]["recheck_commands"], ["python3 tools/validate.py"])

    def test_stale_export_fails_check(self):
        header = "**Counts (2026-11-13):** PROVEN **0** · VERIFIED **1** · KNOWN **0** · CONJECTURE **0**\n"
        row = "| V99 | VERIFIED | stub | `python3 tools/validate.py` |\n"
        with tempfile.TemporaryDirectory() as tmp:
            claims_path = Path(tmp) / "CLAIMS.md"
            out_path = Path(tmp) / "claims-ledger.json"
            claims_path.write_text(header + row, encoding="utf-8")
            built = ledger.build_ledger(claims_path.read_text(encoding="utf-8"))
            out_path.write_text(json.dumps(built, indent=2) + "\n", encoding="utf-8")
            claims_path.write_text(header + row.replace("stub", "mutated"), encoding="utf-8")
            old_claims = ledger.CLAIMS
            ledger.CLAIMS = claims_path
            try:
                code = ledger.main(["--check", "--output", str(out_path)])
            finally:
                ledger.CLAIMS = old_claims
            self.assertEqual(code, 1)

    @unittest.skipUnless((ROOT / "CLAIMS.md").is_file()
                         and (ROOT / "registries/claims-ledger.json").is_file(),
                         "no committed claims ledger in this repository yet")
    def test_commit_matches_claims(self):
        code = subprocess.run(
            [sys.executable, str(ROOT / "tools/export_claims_ledger.py"), "--check"],
            cwd=ROOT,
            check=False,
        ).returncode
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
