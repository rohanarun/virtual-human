# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# See docs/verification-tooling.md for provenance and local changes.

"""Original, in-memory evaluation of frozen models on independent held-out patients.

No loader, CLI, fitting, persistence or independence inference is provided.
All outputs are aggregate; callers still need a disclosure review before release.
"""

import math
import platform
import random
from fractions import Fraction


class EvaluationError(ValueError):
    """Fixed messages intentionally exclude supplied keys, values and metadata."""


def require(condition, code):
    if not condition:
        raise EvaluationError(code)


def keys(values):
    require(isinstance(values, (list, tuple)) and 0 < len(values) <= 100000,
            "invalid_patient_keys")
    require(all(isinstance(x, str) and x and x.strip() == x for x in values),
            "invalid_patient_keys")
    require(len(set(values)) == len(values), "duplicate_patient_keys")
    return tuple(sorted(values))


def keyed(values, patients, binary=False):
    require(isinstance(values, dict) and set(values) == set(patients), "key_mismatch")
    result = [values[p] for p in patients]
    require(all(type(v) in (int, float) and math.isfinite(v) and
                (v in (0, 1) if binary else 0 <= v <= 1) for v in result),
            "invalid_labels" if binary else "invalid_probabilities")
    return result


def both_classes(y):
    require(0 in y and 1 in y, "both_classes_required")


def calibrate_threshold(patient_ids, labels, probabilities, specificity=0.98):
    """Select a threshold using calibration negatives only; positive iff score >= t.

    Move just above the negative boundary, conservatively excluding all ties.
    The return can exceed 1 when a negative probability is 1 (predict none).
    This controls empirical calibration specificity, not population specificity.
    Positive labels are required only so a calibration set missing a class
    is rejected; this does not detect swapped or partly wrong labels.
    They do not set the threshold.
    """
    patients = keys(patient_ids)
    y = keyed(labels, patients, True)
    p = keyed(probabilities, patients)
    both_classes(y)
    require(type(specificity) in (int, float) and math.isfinite(specificity)
            and 0 <= specificity <= 1, "invalid_specificity")
    negative = sorted(s for s, label in zip(p, y) if label == 0)
    allowed = int((1 - Fraction(str(specificity))) * len(negative))
    return 0.0 if allowed == len(negative) else math.nextafter(
        negative[len(negative) - allowed - 1], math.inf)


def _metrics(y, p, threshold):
    n = len(y)
    positives = sum(y)
    negatives = n - positives
    ordered = sorted(zip(p, y), reverse=True)
    tp = fp = 0
    ap = credit = 0.0
    index = 0
    while index < n:
        end = index + 1
        while end < n and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positive = sum(label for _, label in ordered[index:end])
        group_negative = end - index - group_positive
        # Negatives below this score, and half credit for negatives tied here.
        credit += group_positive * (negatives - fp - group_negative / 2)
        tp += group_positive
        fp += group_negative
        if positives:
            ap += group_positive / positives * tp / (tp + fp)
        index = end
    tp = sum(label == 1 and score >= threshold for label, score in zip(y, p))
    fp = sum(label == 0 and score >= threshold for label, score in zip(y, p))
    return {"auroc": credit / (positives * negatives) if positives and negatives else None,
            "average_precision": ap if positives and negatives else None,
            "brier": sum((score - label) ** 2 for score, label in zip(p, y)) / n,
            "sensitivity": tp / positives if positives else None,
            "specificity": (negatives - fp) / negatives if negatives else None,
            "tp": tp, "fp": fp, "tn": negatives - fp, "fn": positives - tp}


def _quantile(values, probability):
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low = math.floor(location)
    high = math.ceil(location)
    return ordered[low] + (location - low) * (ordered[high] - ordered[low])


def evaluate_heldout(*, patient_ids, labels, predictions, thresholds,
                     development_patient_ids, baseline_training_patient_ids,
                     baseline_training_labels,
                     calibration_patient_ids, contract, bootstrap_draws=2000, seed=1729,
                     reference_model="prevalence"):
    """Evaluate supplied frozen probabilities and a development-prevalence baseline.

    Contract assertions are caller-supplied provenance, not evidence established
    by this code. Development includes every patient used for fitting, selection,
    preprocessing or alignment. Calibration may overlap development, never test.
    Thresholds must have been selected without evaluation labels or scores.
    Bootstrap resamples patients jointly across models; models/thresholds stay fixed.
    """
    required = {"design": "independent_heldout", "positive_class": "refractory",
                "development_inventory_complete": True,
                "upstream_independence_verified": True,
                "models_and_thresholds_frozen_before_evaluation": True}
    require(isinstance(contract, dict) and all(
        type(contract.get(k)) is type(v) and contract.get(k) == v
        for k, v in required.items()), "independent_heldout_contract_required")
    patients = keys(patient_ids)
    development = keys(development_patient_ids)
    calibration = keys(calibration_patient_ids)
    require(not set(patients).intersection(development) and
            not set(patients).intersection(calibration), "evaluation_patient_overlap")
    y = keyed(labels, patients, True)
    baseline_training = keys(baseline_training_patient_ids)
    require(set(baseline_training).issubset(development), "baseline_training_inventory")
    training_y = keyed(baseline_training_labels, baseline_training, True)
    both_classes(y)
    both_classes(training_y)
    require(isinstance(predictions, dict) and 1 <= len(predictions) <= 20 and
            all(isinstance(name, str) and name and name != "prevalence" for name in predictions),
            "invalid_models")
    require(isinstance(thresholds, dict) and set(thresholds) == set(predictions) and
            all(type(t) in (int, float) and math.isfinite(t) and
                0 <= t <= math.nextafter(1.0, math.inf) for t in thresholds.values()),
            "invalid_thresholds")
    require(type(bootstrap_draws) is int and 2 <= bootstrap_draws <= 10000,
            "invalid_bootstrap_draws")
    require(type(seed) is int and 0 <= seed <= 2**32 - 1, "invalid_seed")
    scores = {name: keyed(predictions[name], patients) for name in sorted(predictions)}
    prevalence = sum(training_y) / len(training_y)
    scores["prevalence"] = [prevalence] * len(patients)
    require(isinstance(reference_model, str) and reference_model in scores, "invalid_reference_model")
    cutoffs = dict(thresholds, prevalence=0.5)
    metrics = {name: _metrics(y, p, cutoffs[name]) for name, p in scores.items()}
    fields = ("auroc", "average_precision", "brier", "sensitivity", "specificity")
    samples = {name: {field: [] for field in fields} for name in scores}
    differences = {name: {field: [] for field in fields} for name in scores if name != reference_model}
    rng = random.Random(seed)
    invalid = 0
    for _ in range(bootstrap_draws):
        indices = [rng.randrange(len(patients)) for _ in patients]
        sampled_y = [y[i] for i in indices]
        if not (0 in sampled_y and 1 in sampled_y):
            invalid += 1
            continue
        draw = {name: _metrics(sampled_y, [p[i] for i in indices], cutoffs[name])
                for name, p in scores.items()}
        for name in scores:
            for field in fields:
                samples[name][field].append(draw[name][field])
                if name != reference_model:
                    differences[name][field].append(draw[name][field] - draw[reference_model][field])
    def intervals(collection):
        return {name: {field: ([_quantile(v, .025), _quantile(v, .975)]
                              if len(v) >= 1000 else None)
                       for field, v in fields.items()} for name, fields in collection.items()}
    def interval_statuses(collection):
        return {name: {field: {"status": ("insufficient_valid_draws" if len(v) < 1000 else
                                          "degenerate" if min(v) == max(v) else "computed"),
                              "bootstrap_range": [min(v), max(v)] if v else None,
                              "reason": ("At least 1000 valid draws required" if len(v) < 1000 else
                                         "No observed bootstrap variation; may be structural or a boundary artifact, not proof of precision"
                                         if min(v) == max(v) else None)}
                       for field, v in fields.items()} for name, fields in collection.items()}
    return {"patients": len(y), "positive_count": sum(y), "negative_count": len(y) - sum(y),
            "evaluation_prevalence": sum(y) / len(y), "training_prevalence": prevalence,
            "metrics": metrics,
            "reference_model": reference_model,
            "differences_from_reference": {
                name: {field: metrics[name][field] - metrics[reference_model][field] for field in fields}
                for name in differences},
            "bootstrap": {"requested": bootstrap_draws, "valid": bootstrap_draws - invalid,
                          "class_empty": invalid, "seed": seed,
                          "python_version": platform.python_version(),
                          "python_implementation": platform.python_implementation(),
                          "interval_status": ("computed" if bootstrap_draws - invalid >= 1000
                                              else "insufficient_valid_draws"),
                          "minimum_valid_draws": 1000,
                          "interval_reason": (None if bootstrap_draws - invalid >= 1000 else
                                              "At least 1000 valid draws required; increase draws or reassess sparse classes"),
                          "percentile_95_intervals": intervals(samples),
                          "paired_difference_95_intervals": intervals(differences),
                          "metric_interval_status": interval_statuses(samples),
                          "paired_difference_interval_status": interval_statuses(differences),
                          "scope": "Conditional on frozen models and thresholds; class-empty draws excluded"},
            "limitations": ["Caller assertions do not establish independence or eligibility.",
                            "Intervals exclude development, fitting and threshold-selection uncertainty.",
                            "Percentile intervals may collapse at boundaries; no high-specificity guarantee.",
                            "Selected-cohort prevalence, AP and Brier are not population-risk calibration."]}
