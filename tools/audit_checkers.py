# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# See docs/verification-tooling.md for provenance and local changes.

#!/usr/bin/env python3
"""Audit VERIFIED claim re-check commands and digest-bound certificates offline."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "registries/claims-ledger.json"
RESULT_DIR = ROOT / "results/integrity-audit-2026-11-13"
AUDIT_JSON = RESULT_DIR / "audit.json"

SKIP_SELF_RECHECK = {
    "python3 tools/audit_checkers.py --check",
    "python3 tools/audit_checkers.py --write",
}
SKIP_HEAVY_RECHECK = {
    "python3 -m unittest discover -s tests -v",
}

NETWORK_MARKERS = (
    "audit_pdc_metadata",
    "audit_published_cohorts",
    "audit_matrix_headers",
    "audit_author_archive",
    "audit_author_features",
    "audit_panel_names",
    "audit_validation_panel_coverage",
    "verify_published_auc",
    "analyze_auc_discrepancy",
    "audit_trial_landscape",
    " --live",
    "network",
)
DIGEST_TOOLS = {
    "tools/audit_cdr2l_sequences.py": ("--output", ROOT / "registries/cdr2l-sequence-audit-2026-11-06.json"),
    "tools/audit_cdr2l_atlas.py": ("--output", ROOT / "registries/cdr2l-atlas-source-2026-11-06.json"),
    "tools/audit_cdr2l_source_psm.py": ("--check", "--certificate"),
    "tools/audit_cdr2l_spectrum_locator.py": ("--check", "--certificate"),
    "tools/check_stated_specificity_bounds.py": ("--check",),
    "tools/plan_exploratory_subset_design.py": ("--check",),
}
CERTIFICATE_DIGEST_KEYS = ("certificate_sha256", "implementation_sha256")


def load_ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def offline_command(command: str) -> bool:
    lowered = command.lower()
    if "same command" in lowered or lowered.startswith("same "):
        return False
    return not any(marker in command for marker in NETWORK_MARKERS)


def offline_absence_reason(row: dict) -> str | None:
    cmds = row.get("recheck_commands") or []
    if not cmds:
        return "no_command"
    if not any(offline_command(cmd) for cmd in cmds):
        return "network_only"
    return None


def primary_checker(command: str) -> str | None:
    match = re.search(r"(?:^|[\s`])(tools/[\w_./-]+\.py)", command)
    if match:
        return match.group(1)
    match = re.search(r"(results/[\w\-/]+/verify\.py)", command)
    return match.group(1) if match else None


def run_command(command: str) -> tuple[bool, int]:
    completed = subprocess.run(
        command,
        shell=True,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0, completed.returncode


def tamper_mode(checker: str) -> str:
    if checker == "tools/audit_cdr2l_atlas.py":
        return "default_output"
    if checker in {"tools/audit_cdr2l_source_psm.py", "tools/audit_cdr2l_spectrum_locator.py"}:
        return "certificate_flag"
    return "default_output"


IMPORT_CHECKS = {
    "tools/audit_cdr2l_sequences.py": ("audit_cdr2l_sequences", "check_report"),
    "tools/audit_cdr2l_atlas.py": ("audit_cdr2l_atlas", "check_report"),
}


def run_import_check(module: str, func: str, broken: Path) -> tuple[bool, int]:
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
        f"from {module} import {func}\n"
        f"{func}(json.loads(open({str(broken)!r}).read()))\n"
    )
    return run_command(f"python3 -c {json.dumps(code)}")


def run_tamper(checker: str, certificate: Path) -> bool:
    if not certificate.is_file():
        return False
    data = json.loads(certificate.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(data)
    if "certificate_sha256" in mutated:
        mutated["certificate_sha256"] = "0" * 64
    elif "implementation_sha256" in mutated:
        mutated["implementation_sha256"] = "0" * 64
    else:
        mutated["schema_version"] = 0
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / certificate.name
        broken.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if checker in IMPORT_CHECKS:
            module, func = IMPORT_CHECKS[checker]
            ok, _ = run_import_check(module, func, broken)
            return not ok
        if tamper_mode(checker) == "certificate_flag":
            ok, _ = run_command(f"python3 {checker} --check --certificate {broken}")
            return not ok
        return False


def certificate_path(row: dict) -> Path | None:
    for rel in row.get("certificate_paths") or []:
        path = ROOT / rel
        if path.is_file():
            return path
    for rel in row.get("artifact_paths") or []:
        if rel.startswith("registries/") and rel.endswith(".json"):
            path = ROOT / rel
            if path.is_file():
                return path
    return None


def has_recorded_digest(path: Path | None, checker: str | None) -> bool:
    if not path or not path.is_file() or not checker:
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if "certificate_sha256" in data:
        return True
    impl = data.get("implementation_sha256")
    if isinstance(impl, str) and checker in DIGEST_TOOLS:
        return True
    return False


def build_audit(run_checks: bool = True) -> dict:
    ledger = load_ledger()
    verified = [row for row in ledger["claims"] if row["label"] == "VERIFIED"]
    command_cache: dict[str, tuple[bool, int]] = {}
    claims_out = []
    tamper_misses: list[str] = []
    no_offline_checker: dict[str, list[str]] = {"no_command": [], "network_only": []}
    skipped_heavy_recheck: list[str] = []
    offline_defined = offline_ran_ok = tamper_applicable = tamper_failed = 0

    for row in sorted(verified, key=lambda item: item["id"]):
        cmds = [cmd for cmd in row.get("recheck_commands", []) if offline_command(cmd)]
        checker = None
        for cmd in cmds:
            checker = primary_checker(cmd) or checker
        offline_defined_row = bool(cmds)
        if offline_defined_row:
            offline_defined += 1
        else:
            reason = offline_absence_reason(row)
            if reason:
                no_offline_checker[reason].append(row["id"])
        ran = False
        exit_zero = False
        skipped_heavy = False
        if run_checks and cmds:
            cmd = cmds[0]
            if cmd in SKIP_HEAVY_RECHECK:
                skipped_heavy = True
                skipped_heavy_recheck.append(row["id"])
            elif cmd in SKIP_SELF_RECHECK:
                ran = True
                exit_zero = True
                offline_ran_ok += 1
            else:
                ran = True
                if cmd not in command_cache:
                    command_cache[cmd] = run_command(cmd)
                exit_zero, _ = command_cache[cmd]
                if exit_zero:
                    offline_ran_ok += 1

        cert = certificate_path(row)
        digest = has_recorded_digest(cert, checker)
        tamper = "not-applicable"
        if digest and checker:
            tamper_applicable += 1
            if run_checks:
                failed = run_tamper(checker, cert)
                tamper = "yes" if failed else "no"
                if failed:
                    tamper_failed += 1
                else:
                    tamper_misses.append(f"{row['id']}: tamper did not fail for {checker}")
        claim = {
            "id": row["id"],
            "checker": checker,
            "recheck_commands": cmds,
            "runs_offline": offline_defined_row,
            "run_executed": ran,
            "run_exit_zero": exit_zero if ran else None,
            "certificate_path": str(cert.relative_to(ROOT)) if cert else None,
            "fails_on_tamper": tamper,
        }
        if skipped_heavy:
            claim["skipped_heavy_recheck"] = True
        if not offline_defined_row:
            claim["offline_absence_reason"] = offline_absence_reason(row)
        claims_out.append(claim)

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ledger_sha256": ledger["ledger_sha256"],
        "summary": {
            "verified_claims": len(verified),
            "offline_recheck_defined": offline_defined,
            "offline_ran_exit_zero": offline_ran_ok,
            "tamper_applicable": tamper_applicable,
            "tamper_fails_on_tamper": tamper_failed,
            "tamper_misses": tamper_misses,
            "no_offline_checker": no_offline_checker,
            "skipped_heavy_recheck": skipped_heavy_recheck,
        },
        "claims": claims_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when audit.json is stale")
    parser.add_argument("--write", action="store_true", help="write results/integrity-audit-2026-11-13/audit.json")
    parser.add_argument("--skip-run", action="store_true", help="structure-only audit for tests")
    args = parser.parse_args(argv)
    try:
        report = build_audit(run_checks=not args.skip_run)
    except FileNotFoundError as error:
        print(f"FAIL: missing required file: {error}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not AUDIT_JSON.is_file():
            print("FAIL: missing committed audit.json", file=sys.stderr)
            return 1
        committed = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        report["generated_utc"] = committed.get("generated_utc")
        for key in ("schema_version", "ledger_sha256", "summary", "claims"):
            if committed.get(key) != report.get(key):
                print(f"FAIL: stale field {key} in audit.json", file=sys.stderr)
                return 1
        if committed.get("summary", {}).get("tamper_misses"):
            print("FAIL: committed tamper_misses is non-empty", file=sys.stderr)
            for item in committed["summary"]["tamper_misses"]:
                print(item, file=sys.stderr)
            return 1
        print("PASS: integrity audit report matches live audit")
        return 0
    if args.write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        AUDIT_JSON.write_text(rendered, encoding="utf-8")
        print(f"Wrote {AUDIT_JSON.relative_to(ROOT)}")
    else:
        print(rendered, end="")
    if report["summary"]["tamper_misses"]:
        for item in report["summary"]["tamper_misses"]:
            print(item, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
