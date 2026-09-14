from __future__ import annotations

import math
import random
import statistics
from typing import Any, Iterable


def finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize_measurements(
    values: Iterable[float],
    *,
    failed_requests: int,
    successful_texts: int,
) -> dict[str, Any]:
    numbers = finite(values)
    if not numbers:
        raise ValueError("At least one finite measurement is required")
    mean = statistics.fmean(numbers)
    standard_deviation = statistics.stdev(numbers) if len(numbers) >= 2 else None
    cv_percent = (
        standard_deviation / mean * 100
        if standard_deviation is not None and mean
        else None
    )
    return {
        "count": len(numbers),
        "values": numbers,
        "mean": mean,
        "median": statistics.median(numbers),
        "sample_standard_deviation": standard_deviation,
        "cv_percent": cv_percent,
        "min": min(numbers),
        "max": max(numbers),
        "successful_texts": successful_texts,
        "failed_requests": failed_requests,
    }


def stability_status(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_cv_percent: float = 5.0,
) -> str:
    if baseline["failed_requests"] or candidate["failed_requests"]:
        return "INCONCLUSIVE / UNSTABLE"
    if baseline["cv_percent"] is None or candidate["cv_percent"] is None:
        return "INCONCLUSIVE / UNSTABLE"
    if baseline["cv_percent"] > max_cv_percent:
        return "INCONCLUSIVE / UNSTABLE"
    if candidate["cv_percent"] > max_cv_percent:
        return "INCONCLUSIVE / UNSTABLE"
    return "STABLE"


def regression_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_cv_percent: float = 5.0,
    regression_threshold_percent: float = -10.0,
) -> dict[str, Any]:
    controls = stability_status(
        baseline,
        candidate,
        max_cv_percent=max_cv_percent,
    )
    delta_percent = (candidate["mean"] - baseline["mean"]) / baseline["mean"] * 100
    if controls != "STABLE":
        status = "INCONCLUSIVE"
    elif delta_percent <= regression_threshold_percent:
        status = "FAIL / REGRESSION"
    else:
        status = "PASS"
    return {
        "status": status,
        "controls": controls,
        "baseline_mean": baseline["mean"],
        "candidate_mean": candidate["mean"],
        "delta_percent": delta_percent,
        "threshold_percent": regression_threshold_percent,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def bootstrap_mean_difference(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    samples: int = 10000,
    seed: int = 52630001,
) -> dict[str, Any]:
    if not baseline_values or not candidate_values:
        raise ValueError("Both versions require measurements")
    rng = random.Random(seed)
    absolute_differences: list[float] = []
    percentage_differences: list[float] = []
    for _ in range(samples):
        baseline = [rng.choice(baseline_values) for _ in baseline_values]
        candidate = [rng.choice(candidate_values) for _ in candidate_values]
        baseline_mean = statistics.fmean(baseline)
        candidate_mean = statistics.fmean(candidate)
        absolute_differences.append(candidate_mean - baseline_mean)
        percentage_differences.append(
            (candidate_mean - baseline_mean) / baseline_mean * 100
        )
    return {
        "method": "independent empirical bootstrap with replacement across measured blocks",
        "samples": samples,
        "seed": seed,
        "absolute_difference_texts_per_second": {
            "mean": statistics.fmean(absolute_differences),
            "lower_2_5_percent": _percentile(absolute_differences, 0.025),
            "upper_97_5_percent": _percentile(absolute_differences, 0.975),
        },
        "percentage_difference": {
            "mean": statistics.fmean(percentage_differences),
            "lower_2_5_percent": _percentile(percentage_differences, 0.025),
            "upper_97_5_percent": _percentile(percentage_differences, 0.975),
        },
    }
