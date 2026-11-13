# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# See docs/verification-tooling.md for provenance and local changes.

"""Abstract-only evaluation checks; no study values or network access."""

import copy
from pathlib import Path
import platform
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_heldout as ev


def fixture():
    return dict(patient_ids=["a", "b", "c", "d"],
                labels=dict(a=0, b=0, c=1, d=1),
                predictions={"panel": dict(a=.1, b=.5, c=.5, d=.9)},
                thresholds={"panel": .5},
                development_patient_ids=["train-a", "train-b"],
                baseline_training_patient_ids=["train-a", "train-b"],
                baseline_training_labels={"train-a": 0, "train-b": 1},
                calibration_patient_ids=["cal-a", "cal-b"],
                contract={"design": "independent_heldout", "positive_class": "refractory",
                          "development_inventory_complete": True,
                          "upstream_independence_verified": True,
                          "models_and_thresholds_frozen_before_evaluation": True},
                bootstrap_draws=100, seed=7)


class EvaluationTests(unittest.TestCase):
    def test_hand_arithmetic_and_ties(self):
        result = ev.evaluate_heldout(**fixture())
        m = result["metrics"]["panel"]
        self.assertEqual(m["auroc"], .875)  # 3 wins, one tie / four pairs.
        self.assertAlmostEqual(m["average_precision"], 5 / 6)
        self.assertAlmostEqual(m["brier"], .13)
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (2, 1, 1, 0))
        self.assertEqual((m["sensitivity"], m["specificity"]), (1, .5))
        baseline = result["metrics"]["prevalence"]
        self.assertEqual((baseline["auroc"], baseline["average_precision"], baseline["brier"]),
                         (.5, .5, .25))

    def test_paired_draws_identical_models_have_identical_intervals(self):
        args = fixture()
        args["bootstrap_draws"] = 2000
        args["predictions"]["copy"] = dict(args["predictions"]["panel"])
        args["thresholds"]["copy"] = .5
        result = ev.evaluate_heldout(**args)
        intervals = result["bootstrap"]["paired_difference_95_intervals"]
        self.assertEqual(intervals["copy"], intervals["panel"])
        args["predictions"]["constant"] = dict.fromkeys(args["patient_ids"], .5)
        args["thresholds"]["constant"] = .5
        difference = ev.evaluate_heldout(**args)["bootstrap"]["paired_difference_95_intervals"]["constant"]
        self.assertTrue(all(interval == [0, 0] for interval in difference.values()))

    def test_bootstrap_records_python_version(self):
        result = ev.evaluate_heldout(**fixture())["bootstrap"]
        self.assertEqual(result["python_version"], platform.python_version())
        self.assertEqual(result["python_implementation"], "CPython")

    def test_order_invariance_and_rng_isolation(self):
        args = fixture()
        before = copy.deepcopy(args)
        random.seed(119)
        state = random.getstate()
        result = ev.evaluate_heldout(**args)
        self.assertEqual(state, random.getstate())
        self.assertEqual(args, before)
        args["patient_ids"].reverse()
        self.assertEqual(result, ev.evaluate_heldout(**args))

    def test_prespecified_logistic_reference_and_paired_errors(self):
        args = fixture()
        args.update(bootstrap_draws=2000, reference_model="logistic")
        args["predictions"]["logistic"] = dict(args["predictions"]["panel"])
        args["thresholds"]["logistic"] = .5
        result = ev.evaluate_heldout(**args)
        self.assertEqual(result["reference_model"], "logistic")
        self.assertNotIn("logistic", result["differences_from_reference"])
        self.assertTrue(all(value == 0 for value in result["differences_from_reference"]["panel"].values()))
        status = result["bootstrap"]["paired_difference_interval_status"]["panel"]
        self.assertTrue(all(value["status"] == "degenerate" for value in status.values()))
        self.assertTrue(all(interval == [0, 0] for interval in
                            result["bootstrap"]["paired_difference_95_intervals"]["panel"].values()))
        args["predictions"]["logistic"] = dict(a=.8, b=.1, c=.2, d=.9)
        different = ev.evaluate_heldout(**args)
        self.assertAlmostEqual(different["differences_from_reference"]["panel"]["auroc"], .125)
        self.assertEqual(different["bootstrap"]["class_empty"], result["bootstrap"]["class_empty"])
        self.assertEqual(different, ev.evaluate_heldout(**args))
        self.assertEqual(different["bootstrap"]["paired_difference_interval_status"]["panel"]["auroc"]["status"], "computed")
        args["reference_model"] = "unknown"
        with self.assertRaisesRegex(ev.EvaluationError, "invalid_reference_model"):
            ev.evaluate_heldout(**args)

    def test_perfect_small_cohort_marks_boundary_degeneracy(self):
        args = fixture()
        ids = [f"abstract-eval-{i}" for i in range(20)]
        labels = {key: int(i >= 13) for i, key in enumerate(ids)}
        args.update(patient_ids=ids, labels=labels, bootstrap_draws=2000,
                    predictions={"panel": {key: .9 if label else .1 for key, label in labels.items()}})
        result = ev.evaluate_heldout(**args)
        for metric in ("auroc", "specificity"):
            self.assertEqual(result["bootstrap"]["percentile_95_intervals"]["panel"][metric], [1, 1])
            self.assertEqual(result["bootstrap"]["metric_interval_status"]["panel"][metric]["status"], "degenerate")
        # Exact boundary identity: 13/13 is not evidence of 98% specificity.
        self.assertAlmostEqual(.05 ** (1 / 13), .7941833348, places=9)

    def test_threshold_ties_and_test_label_independence(self):
        ids = ["ca", "cb", "cc", "cd"]
        labels = dict(ca=0, cb=0, cc=0, cd=1)
        scores = dict(ca=.4, cb=.4, cc=.8, cd=.9)
        threshold = ev.calibrate_threshold(ids, labels, scores, .5)
        self.assertGreater(threshold, .4)
        self.assertLess(threshold, .8)
        args = fixture()
        args["thresholds"]["panel"] = threshold
        original = ev.evaluate_heldout(**args)
        args["labels"] = {key: 1 - value for key, value in args["labels"].items()}
        reversed_result = ev.evaluate_heldout(**args)
        self.assertEqual(args["thresholds"]["panel"], threshold)
        self.assertEqual(original["metrics"]["panel"]["auroc"] +
                         reversed_result["metrics"]["panel"]["auroc"], 1)
        # Positive calibration labels/scores cannot improve a negative-only cutoff.
        scores["cd"] = 0
        self.assertEqual(threshold, ev.calibrate_threshold(ids, labels, scores, .5))

    def test_threshold_boundary_and_fractional_specificity(self):
        ids = [f"c{i}" for i in range(51)]
        labels = {key: int(i == 50) for i, key in enumerate(ids)}
        scores = {key: i / 50 for i, key in enumerate(ids)}
        threshold = ev.calibrate_threshold(ids, labels, scores, .98)
        self.assertEqual(sum(scores[k] >= threshold for k in ids[:-1]), 1)
        scores[ids[0]] = 1
        self.assertGreater(ev.calibrate_threshold(ids, labels, scores, 1), 1)
        self.assertEqual(ev.calibrate_threshold(ids, labels, scores, 0), 0)

    def test_class_empty_draws_are_counted_not_replaced(self):
        args = fixture()
        args["patient_ids"] = ["a", "c"]
        args["labels"] = dict(a=0, c=1)
        args["predictions"] = {"panel": dict(a=.1, c=.9)}
        result = ev.evaluate_heldout(**args)["bootstrap"]
        rng = random.Random(7)
        expected_empty = sum(len({rng.randrange(2), rng.randrange(2)}) == 1 for _ in range(100))
        self.assertEqual(result["class_empty"], expected_empty)
        self.assertEqual(result["valid"] + result["class_empty"], 100)
        self.assertEqual(result["interval_status"], "insufficient_valid_draws")
        self.assertIsNotNone(result["interval_reason"])
        self.assertIsNone(result["percentile_95_intervals"]["panel"]["brier"])

    def test_no_valid_draws_yields_no_interval(self):
        args = fixture()
        args.update(patient_ids=["a", "c"], labels=dict(a=0, c=1),
                    predictions={"panel": dict(a=.1, c=.9)}, bootstrap_draws=2, seed=0)
        result = ev.evaluate_heldout(**args)["bootstrap"]
        self.assertLess(result["valid"], 2)
        self.assertIsNone(result["percentile_95_intervals"]["panel"]["auroc"])

    def test_overlap_and_false_attestations_rejected(self):
        for role in ("development_patient_ids", "calibration_patient_ids"):
            args = fixture()
            args[role][0] = "a"
            with self.assertRaisesRegex(ev.EvaluationError, "evaluation_patient_overlap"):
                ev.evaluate_heldout(**args)
        for field in fixture()["contract"]:
            args = fixture()
            args["contract"][field] = False
            with self.assertRaises(ev.EvaluationError):
                ev.evaluate_heldout(**args)
        args = fixture()
        args["contract"]["design"] = "repeated_cross_validation"
        with self.assertRaises(ev.EvaluationError):
            ev.evaluate_heldout(**args)

    def test_adversarial_inputs(self):
        mutations = [lambda a: a["patient_ids"].append("a"),
                     lambda a: a["labels"].pop("a"),
                     lambda a: a["labels"].update(a=True),
                     lambda a: a["predictions"]["panel"].update(a=float("nan")),
                     lambda a: a["predictions"]["panel"].update(a=1.1),
                     lambda a: a["thresholds"].update(panel=float("inf")),
                     lambda a: a.update(bootstrap_draws=True),
                     lambda a: a.update(bootstrap_draws=10001),
                     lambda a: a.update(seed=-1)]
        for mutate in mutations:
            args = fixture()
            mutate(args)
            with self.assertRaises(ev.EvaluationError):
                ev.evaluate_heldout(**args)

    def test_development_prevalence_not_evaluation_prevalence(self):
        args = fixture()
        args["development_patient_ids"].append("train-c")
        args["baseline_training_patient_ids"].append("train-c")
        args["baseline_training_labels"]["train-c"] = 0
        result = ev.evaluate_heldout(**args)
        self.assertEqual(result["training_prevalence"], 1 / 3)
        self.assertEqual(result["evaluation_prevalence"], .5)
        self.assertAlmostEqual(result["metrics"]["prevalence"]["brier"], 5 / 18)

    def test_baseline_training_set_distinct_from_development_inventory(self):
        args = fixture()
        args["development_patient_ids"].append("alignment-only")
        self.assertEqual(ev.evaluate_heldout(**args)["training_prevalence"], .5)
        args["baseline_training_patient_ids"].append("not-in-inventory")
        args["baseline_training_labels"]["not-in-inventory"] = 0
        with self.assertRaisesRegex(ev.EvaluationError, "baseline_training_inventory"):
            ev.evaluate_heldout(**args)


if __name__ == "__main__":
    unittest.main()
