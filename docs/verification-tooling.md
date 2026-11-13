# Verification tooling, ported from Project Asma

This directory tree (`tools/` and the tests in `tests/` listed below) is a
faithful port of the domain-agnostic verification machinery developed in
[Project Asma](https://github.com/Foundation-0/project-asma), an evidence-led
research workspace. Project Asma licenses its code under the MIT license and
its text under CC BY 4.0; copyright is retained in each ported file.

Project Asma is a private, owner-stewarded research record. Nothing derived
from its study inputs, patient-adjacent material, licensed source publications,
or its committed registries is part of this port. What moved here is reusable
mathematics and verification process, nothing domain-specific.

## What was ported

| Path | Origin | Local changes |
| --- | --- | --- |
| `tools/analyze_compartment_identifiability.py` | Project Asma `tools/` | None |
| `tools/evaluate_heldout.py` | Project Asma `tools/` | None |
| `tools/export_claims_ledger.py` | Project Asma `tools/` | None |
| `tools/check_sha256_manifests.py` | Project Asma `tools/` | None |
| `tools/check_implementation_hashes.py` | Project Asma `tools/` | None |
| `tools/audit_checkers.py` | Project Asma `tools/` | Only a clean exit message when the adopting repository has not committed `registries/claims-ledger.json` yet (upstream raised a traceback); test adapted; see below |
| `tools/analyze_validation_capacity.py` | Project Asma `tools/` | Adapted: the study-pinned integrity checks (registry file hashes, the split-phrase pin, the committed artifact) were dropped. The exact binomial mathematics (`perfect_lower_bound`, `minimum_perfect_n`, `specificity_grid`) is unchanged, and the CLI is now generic (`--n`, `--target`, `--alpha`). |

| Test path | Local changes |
| --- | --- |
| `tests/test_compartment_identifiability.py` | New, minimal; the upstream tool shipped without a test file |
| `tests/test_evaluate_heldout.py` | Attribution header only |
| `tests/test_validation_capacity.py` | Adapted: math-only tests ported; drift and committed-artifact tests dropped (they pin Project Asma study files) |
| `tests/test_export_claims_ledger.py` | Attribution header; the committed-state check now skips when the adopting repository has no `CLAIMS.md` and `registries/claims-ledger.json` |
| `tests/test_check_sha256_manifests.py` | Attribution header; the repo-wide manifest check skips when no `results/` or `registries/` tree exists |
| `tests/test_implementation_hashes.py` | Attribution header; two git-history stale-hash tests pinned to Project Asma's recorded hashes were dropped, the current-hash test now uses the ported `tools/evaluate_heldout.py` |
| `tests/test_audit_checkers.py` | Adapted: uses a stub ledger; the upstream test asserted Project Asma's committed claim count of 40. `tools/audit_checkers.py` keeps Project Asma's marker lists; they only match tools that exist in a repository adopting the pattern |

## What was deliberately not ported

- Study-pinned integrity plumbing of `analyze_validation_capacity.py` (file
  hashes for Project Asma registry artifacts, the exact-phrase pin, the
  committed output check). Those pins have no meaning in this repository.
- All cancer-track tooling: source adapters, workbook audits, cohort and
  panel-name audits, trial-landscape and registry audits, pilot runners.
  They are inseparable from their study inputs and their data-use terms.
- Anything derived from CC BY-NC-ND licensed source workbooks.
- Project Asma's claims ledger, results, and registries. This port ships the
  machinery, not that repository's research state.

## How to adopt the ledger pattern

Project Asma labels every material public statement with one of four labels.
Writing the labels explicitly is the innovation; the JSON export and the
integrity audit are the enforcement.

- PROVEN: a written proof exists in the repository.
- VERIFIED: a finite check replayable with a recorded command.
- KNOWN: cited from a primary source, exact citation given.
- CONJECTURE: a hypothesis, design target, or open question.

A repository adopting this port would:

1. Keep a `CLAIMS.md` with a labeled table (`| ID | Claim | Re-check |` rows
   with IDs like `V01`, `K03`, `C07`, `P01`).
2. Export it with `tools/export_claims_ledger.py` and commit the result.
3. Pin any committed digest-bearing artifacts with a `SHA256SUMS` manifest and
   replay them with `tools/check_sha256_manifests.py`.
4. Track `implementation_sha256` fields in any registry JSON with
   `tools/check_implementation_hashes.py`.
5. Periodically run `tools/audit_checkers.py` to replay every offline
   re-check command and probe digest-bound certificates for tamper sensitivity.

The test suite runs under this repository's normal `pytest -q` invocation;
every ported test is standard-library only.

## Synthetic-only status

`tools/analyze_compartment_identifiability.py` demonstrates exact rational
ambiguity on invented cell-state proportions. It is synthetic exact
arithmetic, not a fitted biological model and not study analysis. Its output
marker says so explicitly; keep that marker if you reuse it.

`tools/analyze_validation_capacity.py` answers a design question for a
hypothetical zero-false-positive validation. The bounds are exact binomial
calculations, not measurements of any cohort.
