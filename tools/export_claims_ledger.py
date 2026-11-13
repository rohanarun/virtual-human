# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# See docs/verification-tooling.md for provenance and local changes.

#!/usr/bin/env python3
"""Export CLAIMS.md to a deterministic machine-readable ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "CLAIMS.md"
DEFAULT_OUT = ROOT / "registries/claims-ledger.json"

ROW = re.compile(r"^\|\s*([VKCP]\d+)\s*\|")
HEADER = re.compile(
    r"PROVEN \*\*(\d+)\*\* · VERIFIED \*\*(\d+)\*\* · KNOWN \*\*(\d+)\*\* · CONJECTURE \*\*(\d+)\*\*"
)
PREFIX_LABEL = {
    "V": "VERIFIED",
    "K": "KNOWN",
    "C": "CONJECTURE",
    "P": "PROVEN",
}
SOURCE_ID = re.compile(r"\bS\d{2,}\b")
REPO_PATH = re.compile(
    r"(?:registries|results|tools|protocols)/[\w\-\./]+\.(?:json|py|md|cjs|R)"
)


def parse_header_counts(text: str) -> dict[str, int]:
    match = HEADER.search(text)
    if not match:
        raise ValueError("CLAIMS.md header counts line not found")
    keys = ("PROVEN", "VERIFIED", "KNOWN", "CONJECTURE")
    return dict(zip(keys, map(int, match.groups())))


def parse_table_row(line: str) -> dict | None:
    match = ROW.match(line)
    if not match:
        return None
    claim_id = match.group(1)
    parts = [part.strip() for part in line.split("|") if part.strip()]
    if len(parts) < 3:
        return None
    if parts[1] in PREFIX_LABEL.values():
        label = parts[1]
        statement = parts[2] if len(parts) > 2 else ""
        evidence = parts[3] if len(parts) > 3 else ""
    else:
        label = PREFIX_LABEL[claim_id[0]]
        statement = parts[1]
        evidence = parts[2] if len(parts) > 2 else ""
    sources = sorted(set(SOURCE_ID.findall(line)))
    paths = sorted({m.group(0) for m in REPO_PATH.finditer(line)})
    commands = []
    for hit in re.finditer(r"`([^`]+)`", evidence):
        cmd = hit.group(1).strip()
        if cmd.lower() == "same command":
            commands.append("Same command")
        elif cmd.startswith(("python3 ", "node ", "Rscript ")) or cmd.endswith("/verify.py"):
            commands.append(cmd)
    if re.search(r"\bSame command\b", evidence):
        commands.append("Same command")
    return {
        "id": claim_id,
        "label": label,
        "statement": statement,
        "source_ids": sources,
        "certificate_paths": [p for p in paths if p.startswith("registries/") or "certificate" in p],
        "artifact_paths": paths,
        "recheck_commands": commands,
        "row_text": line,
    }


def parse_claim_rows(text: str) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for line in text.splitlines():
        parsed = parse_table_row(line)
        if not parsed:
            continue
        if parsed["id"] in seen:
            raise ValueError(f"duplicate claim id {parsed['id']}")
        seen.add(parsed["id"])
        rows.append(parsed)
    return rows


def resolve_same_command(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        resolved: list[str] = []
        prior_commands: list[str] = []
        for other in rows[:index]:
            if other["label"] == "VERIFIED" and other["recheck_commands"]:
                prior_commands = list(other["recheck_commands"])
        for cmd in row["recheck_commands"]:
            if cmd.lower() != "same command":
                resolved.append(cmd)
            elif prior_commands:
                resolved.extend(prior_commands)
            else:
                resolved.append(cmd)
        row["recheck_commands"] = resolved


def canonical_claim(record: dict) -> str:
    payload = {
        "artifact_paths": record["artifact_paths"],
        "certificate_paths": record["certificate_paths"],
        "id": record["id"],
        "label": record["label"],
        "recheck_commands": record["recheck_commands"],
        "source_ids": record["source_ids"],
        "statement": record["statement"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def claim_sha256(record: dict) -> str:
    return hashlib.sha256(canonical_claim(record).encode("utf-8")).hexdigest()


def build_ledger(text: str) -> dict:
    rows = parse_claim_rows(text)
    resolve_same_command(rows)
    header = parse_header_counts(text)
    by_label: dict[str, int] = {label: 0 for label in PREFIX_LABEL.values()}
    for row in rows:
        by_label[row["label"]] += 1
    if header != by_label:
        raise ValueError(f"header counts {header} do not match rows {by_label}")
    for row in rows:
        row["canonical_sha256"] = claim_sha256(row)
    body = {
        "schema_version": 1,
        "source": "CLAIMS.md",
        "counts": by_label,
        "claims": rows,
    }
    ledger_sha = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"schema_version": 1, "ledger_sha256": ledger_sha, **body}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the export is stale")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    text = CLAIMS.read_text(encoding="utf-8")
    ledger = build_ledger(text)
    rendered = json.dumps(ledger, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.check:
        if not args.output.is_file():
            print(f"FAIL: missing {args.output}", file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != rendered:
            print("FAIL: claims ledger export is stale vs CLAIMS.md", file=sys.stderr)
            return 1
        print("PASS: claims ledger matches CLAIMS.md")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output.relative_to(ROOT)} ({len(ledger['claims'])} claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def parse_claim_rows_by_label(text: str) -> tuple[dict[str, int], int]:
    """Return per-label counts and total; shared with tests/test_claim_counts.py."""
    rows = parse_claim_rows(text)
    by_label = {label: 0 for label in PREFIX_LABEL.values()}
    for row in rows:
        by_label[row["label"]] += 1
    return by_label, sum(by_label.values())

