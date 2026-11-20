"""Scoring, run after every task has finished. The only module that reads ``Task.eval_only``.

T1-T4: the official VSI-Bench rule (ported from VideoCoS eval/vsi.py, itself from thinking-in-space):
first space-delimited token, trailing periods removed; multiple choice by case-insensitive exact
match; numeric answers by mean relative accuracy over the 10 thresholds 0.50..0.95. Unanswered
questions score 0 and stay in the denominator. T5-T7 produce videos; their metrics are separate.
"""
from __future__ import annotations

import math
from collections import defaultdict

NUMERIC_TYPES = {"object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"}


def normalize_prediction(prediction):
    if not isinstance(prediction, str):
        return ""
    return prediction.split(" ")[0].rstrip(".").strip()


def _thresholds(start=.5, end=.95, interval=.05):
    count = int((end - start) / interval + 2)            # the official np.linspace sizing
    step = (end - start) / (count - 1)
    return tuple(start + i * step for i in range(count - 1)) + (end,)


def mean_relative_accuracy(prediction, target):
    if not math.isfinite(target) or target <= 0:
        raise ValueError("VSI numerical ground truth must be finite and positive")
    error = abs(prediction - target) / target
    thresholds = _thresholds()
    return sum(error <= 1 - t for t in thresholds) / len(thresholds)


def score_vsi(question_type, prediction, ground_truth):
    answer = normalize_prediction(prediction)
    if question_type not in NUMERIC_TYPES:
        return float(answer.lower() == str(ground_truth).lower())
    try:
        value = float(answer)
    except (TypeError, ValueError):
        return 0.0
    return mean_relative_accuracy(value, float(ground_truth))


def score_query_tasks(rows):
    """rows: dicts with family, question_type, answer (canonical, '' if none), ground_truth."""
    scored = []
    for row in rows:
        scored.append({**row, "score": score_vsi(row["question_type"], row.get("answer", ""), row["ground_truth"])})
    by_family, by_type = defaultdict(list), defaultdict(list)
    for row in scored:
        by_family[row["family"]].append(row["score"])
        by_type[row["question_type"]].append(row["score"])
    mean = lambda values: round(100 * sum(values) / len(values), 2) if values else None
    return {"rows": scored, "n": len(scored), "mean_percent": mean([r["score"] for r in scored]),
            "by_family_percent": {k: mean(v) for k, v in sorted(by_family.items())},
            "by_question_type_percent": {k: mean(v) for k, v in sorted(by_type.items())},
            "note": "Few questions per type here; the bench's official number is the 8-subtask macro over the full set."}
