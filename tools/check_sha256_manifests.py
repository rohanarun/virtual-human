# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# See docs/verification-tooling.md for provenance and local changes.

#!/usr/bin/env python3
"""Recompute pinned SHA-256 manifests under results/ and digest fields in registries/."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (ROOT / "results", ROOT / "registries")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_sums(path: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if "  " not in line:
            raise ValueError(f"{path}:{lineno}: malformed SHA256SUMS line")
        digest, name = line.split("  ", 1)
        if not HEX64.fullmatch(digest):
            raise ValueError(f"{path}:{lineno}: invalid digest")
        if name in table:
            raise ValueError(f"{path}:{lineno}: duplicate path {name}")
        target = (ROOT / name) if name.startswith(("protocols/", "tools/", "results/", "registries/")) else path.parent / name
        if not target.is_file():
            raise ValueError(f"{path}:{lineno}: missing file {name}")
        got = sha256_bytes(target.read_bytes())
        if got != digest:
            raise ValueError(f"{path}:{lineno}: digest mismatch for {name}")
        table[name] = digest
    return table


def check_result_manifest(result_dir: Path) -> None:
    manifest = result_dir / "manifest.json"
    sums = result_dir / "SHA256SUMS"
    if not manifest.is_file() or not sums.is_file():
        return
    parse_sums(sums)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    paths = ["manifest.json"]
    for item in data.get("files", []):
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError(f"{manifest}: invalid file entry")
        paths.append(item["path"])
        target = result_dir / item["path"]
        if not target.is_file():
            raise ValueError(f"{manifest}: missing snapshot {item['path']}")
        if "source_zip_sha256" in item:
            digest = item["source_zip_sha256"]
            if not HEX64.fullmatch(digest):
                raise ValueError(f"{manifest}: invalid source_zip_sha256 for {item['path']}")


def check_registry_certificate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    digest = data.get("certificate_sha256")
    if not digest:
        return
    if not HEX64.fullmatch(digest):
        raise ValueError(f"{path}: invalid certificate_sha256")
    payload = {k: v for k, v in data.items() if k != "certificate_sha256"}
    got = sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    )
    if got != digest:
        raise ValueError(f"{path}: certificate_sha256 mismatch")


def check_tree() -> None:
    checks = 0
    for base in SCAN_ROOTS:
        if not base.is_dir():
            continue
        for sums in sorted(base.rglob("SHA256SUMS")):
            parse_sums(sums)
            checks += 1
        if base.name == "results":
            for manifest in sorted(base.rglob("manifest.json")):
                check_result_manifest(manifest.parent)
                checks += 1
        if base.name == "registries":
            for path in sorted(base.glob("*.json")):
                check_registry_certificate(path)
                checks += 1
    if checks == 0:
        raise ValueError("no manifest checks discovered")
    print(f"PASS: {checks} offline SHA-256 manifest checks")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        check_tree()
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

