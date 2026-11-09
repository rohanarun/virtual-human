Research model inputs drawn from published population studies; not medical advice, dosing guidance or treatment recommendations.

# Ovarian HGSOC parameter ledger

Machine-readable population parameters for **high-grade serous ovarian cancer (HGSOC)** and **first-line / maintenance treatment-sequence** research models. The ledger is intended as a shared input layer for:

- [Foundation-0/project-asma PR #47](https://github.com/Foundation-0/project-asma/pull/47) (surgical-removal assumption audits), and
- [Foundation-0/virtual-human PR #3](https://github.com/Foundation-0/virtual-human/pull/3) reaction / intervention graph experiments referenced in [`docs/project-asma-integration.md`](../docs/project-asma-integration.md).

## Files

| File | Role |
| --- | --- |
| [`ovarian_hgsoc_parameter_ledger.json`](ovarian_hgsoc_parameter_ledger.json) | Parameter rows |
| [`ovarian_hgsoc_parameter_ledger.schema.json`](ovarian_hgsoc_parameter_ledger.schema.json) | JSON Schema for row shape |
| [`../scripts/validate_ovarian_parameter_ledger.py`](../scripts/validate_ovarian_parameter_ledger.py) | Pinned validator |

## Row labels

Each row carries **`label`**:

- **KNOWN** — numeric or categorical value read from **open full text** of the cited primary source (locator includes `verified from full text, ...`).
- **CONJECTURE** — value taken from abstract or secondary summary when full text is not openly accessible; **`derivation`** must include `value from abstract only` or an explicit derivation chain.
- **CONJECTURE** — value inferred or relayed from a secondary summary; see **`derivation`**.
- **MISSING** — quantity required for modeling but no adequate primary-source estimate; **`value`** is null (no guess).

Every row includes **`population`**, **`units`**, **`range`** (point, CI95, IQR, etc.), **`citation`** (DOI, first author surname, year, title), **`locator`**, and **`limits`**.

## What this extends

The citation/locator/label pattern follows the public registry style introduced in **project-asma PR #47** (`registries/rosen-treatment-sequence-sources.json`). Virtual-human adds quantitative fields and stores the ledger under **`data/clinical/`** next to other provenance-annotated inputs (for example [`midazolam_osp_observed.json`](midazolam_osp_observed.json)).

Schema remains compatible with that registry vocabulary; downstream tools may ignore unknown quantitative keys.

## Validation

**Offline (CI):**

```bash
PYTHONPATH=. python scripts/validate_ovarian_parameter_ledger.py
PYTHONPATH=. pytest -q tests/test_ovarian_parameter_ledger.py -m "not network"
```

**Crossref DOI check (network; run before merging ledger changes):**

```bash
PYTHONPATH=. python scripts/validate_ovarian_parameter_ledger.py --check-dois
```

The DOI step resolves each unique DOI via `https://api.crossref.org/works/<doi>` and requires **title** and **first-author surname** agreement with the ledger, not merely HTTP 200.

## Scope limits

- Public, published sources only; no private patient data or clinical guidance.
- Associations from observational studies and meta-analyses are not causal treatment effects.
- PK rows remain **MISSING** until a registered virtual-human model contract consumes carboplatin/paclitaxel parameters in the sequence layer.
