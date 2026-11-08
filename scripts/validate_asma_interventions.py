"""Optional cross-repository check of wholly synthetic intervention operators.

Run with a trusted Project Asma checkout; this imports its local Python module.
The same invented numbers are used as mol/L in Virtual Human and arbitrary
burden units in Asma. This is a numerical comparison, not a physiological unit
conversion, clinical model, reproduction of Gu et al., or dataset analysis.
Asma is not a package/runtime dependency and no files are downloaded.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
from hashlib import sha256
import importlib.util
import json
from math import log
from pathlib import Path
import subprocess

from vh.graph.reactions import Phase, Reaction, ReactionNetwork, Species, run_protocol

SOURCE = ("Original artificial mathematical fixture; no empirical calibration",)


def revision(root):
    """Report revision and dirty state so a hash does not imply a clean checkout."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def validate(asma_root):
    exact_path = asma_root / "tools/analyze_treatment_sequence.py"
    spec = importlib.util.spec_from_file_location("asma_exact_sequence", exact_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the trusted Asma exact-arithmetic module")
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)
    vh_root = Path(__file__).resolve().parents[1]
    rows = []
    # Main contrast, no selective enrichment, and a target above available burden.
    for label, a, b, q in (("selective", F(1, 10), F(1), F(1, 10)),
                            ("nonselective", F(1, 2), F(1, 2), F(1, 10)),
                            ("capped", F(1, 10), F(1, 2), F(1, 2))):
        initial_exact = (F(9, 10), F(1, 10))
        initial = dict(zip(("S", "R"), map(float, initial_exact)))
        model = ReactionNetwork(
            [Species(name, SOURCE) for name in initial],
            [Reaction("loss_S", {"S": 1}, {}, -log(float(a)), "1/s", SOURCE),
             Reaction("loss_R", {"R": 1}, {}, -log(float(b)), "1/s", SOURCE)],
            context={"system": "synthetic", "compartment": "constant-volume"},
            provenance=SOURCE,
        )
        expected = exact.compare(initial_exact, a, b, q)
        evolve = Phase("evolve", 1, SOURCE)

        def run(state, phases):
            return run_protocol(model, state, phases, max_step_s=0.001)[-1].trajectory.final

        retain = {name: float(q) for name in initial}
        first = run(initial, [Phase("retain-then-evolve", 1, SOURCE, retain=retain)])
        prop_last = run(initial, [evolve, Phase("retain", 0, SOURCE, retain=retain)])
        pre_last = run(initial, [evolve])
        # State-dependent removal is explicit caller logic, not a hidden VH rule.
        target = float(q * sum(initial_exact))
        fraction = min(1.0, target / sum(pre_last.values()))
        fixed_last = run(pre_last, [Phase("fixed-residual", 0, SOURCE,
                                         retain={name: fraction for name in initial})])
        comparisons = (("proportional_surgery_first", first),
                       ("proportional_chemo_first", prop_last),
                       ("fixed_surgery_first", first),
                       ("fixed_chemo_first", fixed_last))
        errors = {name: max(abs(observed[key] - float(F(expected[name][field])))
                           for key, field in (("S", "sensitive"), ("R", "resistant")))
                  for name, observed in comparisons}
        rows.append({"fixture": label, "a": str(a), "b": str(b), "q": str(q),
                     "maximum_absolute_error_mol_per_L": max(errors.values()),
                     "comparison_errors_mol_per_L": errors,
                     "exact_fixed_after_to_before_resistant_ratio":
                         expected["fixed_after_to_before_resistant_ratio"]})
    maximum = max(row["maximum_absolute_error_mol_per_L"] for row in rows)
    tolerance = 1e-10
    result = {
        "status": "pass" if maximum <= tolerance else "fail",
        "scope": "synthetic_operator_consistency_only_not_biological_validation",
        "as_of": "2026-11-05",
        "authority": "Project Asma exact rational operators and explicit linear-flow solution",
        "asma_checkout": revision(asma_root), "virtual_human_checkout": revision(vh_root),
        "source_sha256": {"asma_exact_module": sha256(exact_path.read_bytes()).hexdigest(),
                          "vh_reactions_module": sha256((vh_root / "vh/graph/reactions.py").read_bytes()).hexdigest(),
                          "comparison_script": sha256(Path(__file__).read_bytes()).hexdigest()},
        "tolerance_absolute_mol_per_L": tolerance,
        "maximum_absolute_error_mol_per_L": maximum,
        "fixtures": rows,
        "limitations": ["invented quantities and rates", "no physiological conversion",
                        "fixed-residual map explicitly applied outside the VH protocol runner",
                        "no full Gu model, patient model, survival inference or clinical recommendation"],
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asma-root", required=True, type=Path,
                        help="trusted local Project Asma checkout containing the exact operator module")
    args = parser.parse_args()
    result = validate(args.asma_root.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)
