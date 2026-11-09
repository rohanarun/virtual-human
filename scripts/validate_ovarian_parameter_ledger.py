"""Validate the ovarian HGSOC parameter ledger (schema + optional Crossref DOI checks).

Offline schema validation (default):
  PYTHONPATH=. python scripts/validate_ovarian_parameter_ledger.py

Network DOI verification (separate step; not run in default CI):
  PYTHONPATH=. python scripts/validate_ovarian_parameter_ledger.py --check-dois
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "clinical" / "ovarian_hgsoc_parameter_ledger.json"
SCHEMA_PATH = ROOT / "data" / "clinical" / "ovarian_hgsoc_parameter_ledger.schema.json"
USER_AGENT = "VirtualHumanLedger/1.0 (mailto:agent@cursor.com)"
LABELS = frozenset({"KNOWN", "CONJECTURE", "MISSING"})


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.casefold())


def _crossref_first_author_surname(author: dict) -> str:
    if author.get("family"):
        return str(author["family"])
    name = str(author.get("name", "")).strip()
    if not name:
        return ""
    return name.split()[-1]


def _titles_match(ledger_title: str, crossref_title: str) -> bool:
    a = _normalize_title(ledger_title)
    b = _normalize_title(crossref_title)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _surnames_match(ledger_surname: str, crossref_surname: str) -> bool:
    return ledger_surname.casefold() == crossref_surname.casefold()


def load_ledger() -> dict:
    with LEDGER_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_schema(ledger: dict) -> list[str]:
    errors: list[str] = []
    required_top = ("schema_version", "ledger_id", "checked_on", "scope", "extends", "entries")
    for key in required_top:
        if key not in ledger:
            errors.append(f"missing top-level field {key!r}")

    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("entries must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        prefix = f"entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in (
            "id",
            "quantity",
            "value",
            "range",
            "units",
            "population",
            "citation",
            "locator",
            "label",
            "limits",
        ):
            if field not in entry:
                errors.append(f"{prefix} missing field {field!r}")

        entry_id = entry.get("id")
        if isinstance(entry_id, str):
            if entry_id in seen_ids:
                errors.append(f"duplicate id {entry_id!r}")
            seen_ids.add(entry_id)

        label = entry.get("label")
        if label not in LABELS:
            errors.append(f"{prefix} label {label!r} not in {sorted(LABELS)}")

        if label == "MISSING" and entry.get("value") is not None:
            errors.append(f"{prefix} MISSING row must have null value")

        citation = entry.get("citation")
        if isinstance(citation, dict):
            for field in ("doi", "first_author_surname", "year", "title"):
                if field not in citation or not citation[field]:
                    errors.append(f"{prefix}.citation missing {field!r}")
        else:
            errors.append(f"{prefix}.citation must be an object")

        rng = entry.get("range")
        if rng is not None:
            if not isinstance(rng, dict) or "kind" not in rng:
                errors.append(f"{prefix}.range must be null or an object with kind")

        if label == "CONJECTURE" and not entry.get("derivation"):
            errors.append(f"{prefix} CONJECTURE requires non-empty derivation")

    return errors


def fetch_crossref_work(doi: str) -> dict:
    encoded = urllib.parse.quote(doi, safe="")
    url = f"https://api.crossref.org/works/{encoded}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return payload["message"]


def check_dois(ledger: dict) -> dict:
    entries = ledger["entries"]
    unique_dois: dict[str, list[str]] = {}
    for entry in entries:
        doi = entry["citation"]["doi"]
        unique_dois.setdefault(doi, []).append(entry["id"])

    results = []
    failures = 0
    for doi, entry_ids in sorted(unique_dois.items()):
        row = {"doi": doi, "entry_ids": entry_ids, "status": "ok"}
        try:
            work = fetch_crossref_work(doi)
        except urllib.error.HTTPError as exc:
            failures += 1
            row["status"] = "http_error"
            row["detail"] = str(exc)
            results.append(row)
            continue
        except urllib.error.URLError as exc:
            failures += 1
            row["status"] = "network_error"
            row["detail"] = str(exc)
            results.append(row)
            continue

        authors = work.get("author") or [{}]
        crossref_surname = _crossref_first_author_surname(authors[0])
        crossref_title = (work.get("title") or [""])[0]
        ledger_entry = next(e for e in entries if e["citation"]["doi"] == doi)
        ledger_surname = ledger_entry["citation"]["first_author_surname"]
        ledger_title = ledger_entry["citation"]["title"]

        surname_ok = _surnames_match(ledger_surname, crossref_surname)
        title_ok = _titles_match(ledger_title, crossref_title)
        row["crossref_first_author_surname"] = crossref_surname
        row["crossref_title"] = crossref_title
        row["surname_match"] = surname_ok
        row["title_match"] = title_ok
        if not (surname_ok and title_ok):
            failures += 1
            row["status"] = "mismatch"
        results.append(row)

    return {
        "ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
        "unique_dois_checked": len(unique_dois),
        "failures": failures,
        "results": results,
    }


def label_counts(ledger: dict) -> dict[str, int]:
    counts = {label: 0 for label in sorted(LABELS)}
    for entry in ledger["entries"]:
        counts[entry["label"]] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-dois",
        action="store_true",
        help="Contact Crossref and verify DOI title and first-author surname (network step).",
    )
    args = parser.parse_args(argv)

    if not LEDGER_PATH.is_file():
        print(f"ledger not found: {LEDGER_PATH}", file=sys.stderr)
        return 2

    ledger = load_ledger()
    schema_errors = validate_schema(ledger)
    if schema_errors:
        print("Schema validation failed:")
        for err in schema_errors:
            print(f"  - {err}")
        return 1

    summary = {
        "status": "pass",
        "schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
        "label_counts": label_counts(ledger),
        "entry_count": len(ledger["entries"]),
    }
    print(json.dumps(summary, indent=2))

    if args.check_dois:
        print("\n--- Crossref DOI verification (network) ---")
        doi_report = check_dois(ledger)
        print(json.dumps(doi_report, indent=2))
        if doi_report["failures"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
