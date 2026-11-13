# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# See docs/verification-tooling.md for provenance and local changes.

"""Verify registry implementation_sha256 entries against files on disk.

Statuses per entry:
- **OK** — recorded hash matches the current file bytes.
- **STALE** — recorded hash matches an older revision of the file in ``git`` history
  (not current). Fails unless the registry documents the drift under
  ``implementation_hash_drift``.
- **UNKNOWN** — recorded hash matches neither current nor any historical revision.

Also handles a string ``implementation_sha256`` on the validation-capacity registry
(hash of ``tools/analyze_validation_capacity.py``).

Command-line exit codes:

- **0** — inside a git work tree and every entry is OK or an allowlisted stale hash.
- **1** — the repository root is not the git work-tree top level (stale detection
  needs this project's history), or one or more entries failed. When ``.git`` is
  absent or the tree sits inside another repository (e.g. a GitHub/Zenodo source
  archive unpacked under a foreign ``.git``), prints
  ``ERROR: git checkout required for stale-hash detection.`` to stderr and exits
  **1** without scanning registries.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRIES = ROOT / "registries"
CAPACITY_TOOL = ROOT / "tools" / "analyze_validation_capacity.py"

STATUS_OK = "OK"
STATUS_STALE = "STALE"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_MISSING = "MISSING"
STATUS_INVALID = "INVALID"


@dataclass(frozen=True)
class EntryResult:
    registry: str
    key: str
    expected: str
    status: str
    stale_commit: str | None = None
    detail: str = ""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_inside_git_work_tree(root: Path | None = None) -> bool:
    """Return True when ``root`` is the top level of the git work tree containing it.

    Returns False when ``git`` is missing, when ``root`` is not in a work tree, or
    when ``root`` lies inside another repository (e.g. a source archive directory
    nested under an unrelated checkout).
    """
    root = (root or ROOT).resolve()
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return False
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    if top.returncode != 0 or not top.stdout.strip():
        return False
    return Path(top.stdout.strip()).resolve() == root


def resolve_implementation_path(key: str, root: Path) -> Path:
    direct = root / key
    if direct.is_file():
        return direct
    under_tools = root / "tools" / key
    if under_tools.is_file():
        return under_tools
    raise FileNotFoundError(key)


def git_commit_for_file_hash(root: Path, rel_path: Path, expected: str) -> str | None:
    """Return the newest commit on ``rel_path`` whose blob hash equals ``expected``."""
    cache_key = (str(rel_path), expected)
    if not hasattr(git_commit_for_file_hash, "_cache"):
        git_commit_for_file_hash._cache = {}
    if cache_key in git_commit_for_file_hash._cache:
        return git_commit_for_file_hash._cache[cache_key]

    log = subprocess.run(
        ["git", "log", "--format=%H", "--", str(rel_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    match = None
    if log.returncode == 0:
        for commit in log.stdout.splitlines():
            commit = commit.strip()
            if not commit:
                continue
            blob = subprocess.run(
                ["git", "show", f"{commit}:{rel_path.as_posix()}"],
                cwd=root,
                capture_output=True,
                check=False,
            )
            if blob.returncode != 0:
                continue
            if hashlib.sha256(blob.stdout).hexdigest() == expected:
                match = commit
                break
    git_commit_for_file_hash._cache[cache_key] = match
    return match


def drift_allows_stale(registry_data: dict, key: str, expected: str) -> bool:
    drift = registry_data.get("implementation_hash_drift")
    if not isinstance(drift, dict):
        return False
    entry = drift.get(key)
    if not isinstance(entry, dict):
        return False
    allowed = entry.get("allowed_recorded_hash")
    return isinstance(allowed, str) and allowed.lower() == expected.lower()


def check_entry(
    root: Path,
    registry_name: str,
    registry_data: dict,
    key: str,
    expected: str,
) -> EntryResult:
    if not isinstance(expected, str) or len(expected) != 64:
        return EntryResult(registry_name, key, str(expected), STATUS_INVALID, detail="invalid hash")

    try:
        path = resolve_implementation_path(key, root)
    except FileNotFoundError:
        return EntryResult(registry_name, key, expected, STATUS_MISSING, detail="file not found")

    rel = path.relative_to(root)
    current = file_sha256(path)
    expected_lower = expected.lower()

    if current == expected_lower:
        return EntryResult(registry_name, key, expected, STATUS_OK)

    stale_commit = git_commit_for_file_hash(root, rel, expected_lower)
    if stale_commit:
        if drift_allows_stale(registry_data, key, expected_lower):
            return EntryResult(
                registry_name,
                key,
                expected,
                STATUS_STALE,
                stale_commit=stale_commit,
                detail="allowed stale hash (implementation_hash_drift)",
            )
        return EntryResult(
            registry_name,
            key,
            expected,
            STATUS_STALE,
            stale_commit=stale_commit,
            detail="stale hash; document under implementation_hash_drift or re-run audit",
        )

    return EntryResult(
        registry_name,
        key,
        expected,
        STATUS_UNKNOWN,
        detail=f"current={current}",
    )


def iter_registry_payloads(root: Path):
    reg_dir = root / "registries"
    for registry_path in sorted(reg_dir.glob("pilot-001-*.json")):
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        yield registry_path.name, data


def check_implementation_hashes(root: Path | None = None) -> list[EntryResult]:
    root = root or ROOT
    results: list[EntryResult] = []
    capacity_key = str(CAPACITY_TOOL.relative_to(root))

    for registry_name, data in iter_registry_payloads(root):
        impl = data.get("implementation_sha256")
        if impl is None:
            continue
        if isinstance(impl, str):
            results.append(
                check_entry(root, registry_name, data, capacity_key, impl)
            )
        elif isinstance(impl, dict):
            for key, expected in impl.items():
                results.append(check_entry(root, registry_name, data, key, expected))
        else:
            results.append(
                EntryResult(
                    registry_name,
                    "",
                    "",
                    STATUS_INVALID,
                    detail="implementation_sha256 must be object or string",
                )
            )
    return results


def failures(results: list[EntryResult]) -> list[EntryResult]:
    out = []
    for item in results:
        if item.status == STATUS_OK:
            continue
        if item.status == STATUS_STALE and "allowed stale" in item.detail:
            continue
        out.append(item)
    return out


def main(argv=None) -> int:
    _ = argv
    if not is_inside_git_work_tree(ROOT):
        print("ERROR: git checkout required for stale-hash detection.", file=sys.stderr)
        return 1

    results = check_implementation_hashes()
    stale_allowed = [r for r in results if r.status == STATUS_STALE and "allowed" in r.detail]
    failed = failures(results)

    for item in stale_allowed:
        print(
            f"STALE (allowed): {item.registry} {item.key} at commit {item.stale_commit}",
            file=sys.stderr,
        )

    if failed:
        print("implementation_sha256 check failed:", file=sys.stderr)
        for item in failed:
            line = f"{item.status}: {item.registry} {item.key} recorded {item.expected}"
            if item.stale_commit:
                line += f" (git {item.stale_commit})"
            if item.detail:
                line += f" — {item.detail}"
            print(line, file=sys.stderr)
        return 1

    print("OK: every implementation_sha256 entry is current or an allowlisted stale hash.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
