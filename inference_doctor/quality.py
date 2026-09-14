from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_doctor.benchmark_results import (
    BenchmarkResult,
    BenchmarkResultError,
    IncomparableBenchmarkError,
    ensure_comparable,
    load_normalized_benchmark,
)


QUALITY_SCHEMA_VERSION = "inference-ci-quality-1"


class QualityInputError(BenchmarkResultError):
    """Raised when repeated-run input cannot be evaluated."""


@dataclass(frozen=True)
class QualityThresholds:
    stability_cv_percent: float = 5.0
    regression_percent: float = 10.0

    def validate(self) -> None:
        if self.stability_cv_percent < 0 or self.regression_percent < 0:
            raise ValueError("Quality thresholds must be non-negative")


@dataclass(frozen=True)
class RepeatedRun:
    path: Path
    result: BenchmarkResult
    value: float | None


METRICS: dict[str, tuple[str, bool]] = {
    "output_token_throughput": ("Output token throughput", False),
    "request_throughput": ("Request throughput", False),
    "texts_per_second": ("Texts per second", False),
    "ttft_mean_ms": ("TTFT mean", True),
    "ttft_median_ms": ("TTFT median", True),
    "ttft_p95_ms": ("TTFT p95", True),
    "tpot_mean_ms": ("TPOT mean", True),
    "tpot_median_ms": ("TPOT median", True),
    "tpot_p95_ms": ("TPOT p95", True),
    "requests_successful": ("Successful requests", False),
    "requests_failed": ("Failed requests", True),
}


def metric_choices() -> tuple[str, ...]:
    return tuple(METRICS)


def _metric_value(result: BenchmarkResult, metric: str) -> float | None:
    if metric == "output_token_throughput":
        return result.metrics.output_token_throughput
    if metric == "request_throughput":
        return result.metrics.request_throughput
    if metric == "texts_per_second":
        return result.metrics.texts_per_second
    if metric == "requests_successful":
        return result.metrics.requests.successful
    if metric == "requests_failed":
        return result.metrics.requests.failed

    prefix, statistic = metric.split("_", 1)
    stats = result.metrics.ttft_ms if prefix == "ttft" else result.metrics.tpot_ms
    return getattr(stats, statistic.removesuffix("_ms"), None)


def discover_run_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise QualityInputError(f"Repeated-run input is not a directory: '{directory}'")

    paths = {
        path
        for path in directory.rglob("*.json")
        if path.name == "normalized.json"
        or path.name.endswith(".normalized.json")
        or path.name.startswith("repetition-")
    }
    paths = {
        path
        for path in paths
        if "aggregate" not in path.name.lower()
        and path.name not in {"summary.json", "run.json"}
    }
    if not paths:
        raise QualityInputError(
            f"No normalized repeated-run JSON files found under '{directory}'"
        )
    return sorted(paths)


def load_repeated_runs(directory: Path, metric: str) -> list[RepeatedRun]:
    if metric not in METRICS:
        raise QualityInputError(
            f"Unknown metric '{metric}'. Choose one of: {', '.join(METRICS)}"
        )
    runs: list[RepeatedRun] = []
    for path in discover_run_paths(directory):
        result = load_normalized_benchmark(path)
        runs.append(RepeatedRun(path, result, _metric_value(result, metric)))
    return runs


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _outlier_paths(runs: list[RepeatedRun]) -> list[str]:
    values = [run.value for run in runs]
    if len(values) < 3 or any(value is None for value in values):
        return []
    numbers = [float(value) for value in values if value is not None]
    median = statistics.median(numbers)
    deviations = [abs(value - median) for value in numbers]
    mad = statistics.median(deviations)
    q1 = _percentile(numbers, 0.25)
    q3 = _percentile(numbers, 0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    flagged: set[int] = {
        index for index, value in enumerate(numbers) if value < low or value > high
    }
    if mad > 0:
        flagged.update(
            index
            for index, value in enumerate(numbers)
            if abs(value - median) > 3 * mad
        )
    return [str(runs[index].path) for index in sorted(flagged)]


def summarize_runs(runs: list[RepeatedRun]) -> dict[str, Any]:
    values = [run.value for run in runs if run.value is not None]
    stats: dict[str, Any] = {
        "count": len(runs),
        "values": values,
        "missing_values": len(runs) - len(values),
        "mean": None,
        "median": None,
        "sample_standard_deviation": None,
        "cv_percent": None,
        "min": None,
        "max": None,
        "outlier_paths": _outlier_paths(runs),
    }
    if not values:
        return stats
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) >= 2 else None
    stats.update(
        {
            "mean": mean,
            "median": statistics.median(values),
            "sample_standard_deviation": standard_deviation,
            "cv_percent": (
                standard_deviation / mean * 100
                if standard_deviation is not None and mean
                else None
            ),
            "min": min(values),
            "max": max(values),
        }
    )
    return stats


def _validate_environment(
    baseline: list[RepeatedRun],
    candidate: list[RepeatedRun],
) -> list[str]:
    reasons: list[str] = []
    references = [
        ("baseline", baseline),
        ("candidate", candidate),
    ]
    for label, runs in references:
        for run in runs[1:]:
            try:
                ensure_comparable(runs[0].result, run.result)
            except IncomparableBenchmarkError as exc:
                reasons.append(f"{label} environment/configuration mismatch: {exc}")
    try:
        ensure_comparable(baseline[0].result, candidate[0].result)
    except IncomparableBenchmarkError as exc:
        reasons.append(f"baseline/candidate environment/configuration mismatch: {exc}")
    return reasons


def compare_repeated_runs(
    baseline: list[RepeatedRun],
    candidate: list[RepeatedRun],
    *,
    metric: str,
    thresholds: QualityThresholds = QualityThresholds(),
) -> dict[str, Any]:
    thresholds.validate()
    if not baseline or not candidate:
        raise QualityInputError("Both baseline and candidate require repeated runs")
    if metric not in METRICS:
        raise QualityInputError(f"Unknown metric '{metric}'")

    baseline_stats = summarize_runs(baseline)
    candidate_stats = summarize_runs(candidate)
    reasons = _validate_environment(baseline, candidate)
    failed_requests = [
        (label, run.path)
        for label, runs in (("baseline", baseline), ("candidate", candidate))
        for run in runs
        if run.result.metrics.requests.failed is None
        or run.result.metrics.requests.failed > 0
    ]
    if failed_requests:
        reasons.append("one or more repetitions recorded failed or unavailable requests")
    if baseline_stats["missing_values"] or candidate_stats["missing_values"]:
        reasons.append(f"{METRICS[metric][0]} was unavailable in one or more repetitions")
    if baseline_stats["count"] < 2 or candidate_stats["count"] < 2:
        reasons.append("at least two repetitions per side are required for stability")

    baseline_cv = baseline_stats["cv_percent"]
    candidate_cv = candidate_stats["cv_percent"]
    if baseline_cv is None or baseline_cv > thresholds.stability_cv_percent:
        reasons.append("baseline measurements exceeded the allowed stability threshold")
    if candidate_cv is None or candidate_cv > thresholds.stability_cv_percent:
        reasons.append("candidate measurements exceeded the allowed stability threshold")

    baseline_mean = baseline_stats["mean"]
    candidate_mean = candidate_stats["mean"]
    delta_percent = None
    regression_percent = None
    if baseline_mean is not None and candidate_mean is not None and baseline_mean != 0:
        delta_percent = (candidate_mean - baseline_mean) / baseline_mean * 100
        lower_is_better = METRICS[metric][1]
        regression_percent = delta_percent if lower_is_better else -delta_percent

    if reasons:
        status = "INCONCLUSIVE"
    elif regression_percent is not None and regression_percent >= thresholds.regression_percent:
        status = "FAIL"
        reasons.append(f"candidate regression reached the {thresholds.regression_percent:g}% threshold")
    else:
        status = "PASS"

    explanation = (
        "; ".join(reasons)
        if reasons
        else "measurements were stable and within the regression threshold"
    )
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "result": status,
        "metric": metric,
        "metric_label": METRICS[metric][0],
        "environment": {
            "baseline": baseline[0].result.metadata_dict(),
            "candidate": candidate[0].result.metadata_dict(),
        },
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "delta_percent": delta_percent,
        "regression_percent": regression_percent,
        "thresholds": {
            "stability_cv_percent": thresholds.stability_cv_percent,
            "regression_percent": thresholds.regression_percent,
        },
        "measurement_quality": {
            "baseline_stable": baseline_cv is not None
            and baseline_cv <= thresholds.stability_cv_percent,
            "candidate_stable": candidate_cv is not None
            and candidate_cv <= thresholds.stability_cv_percent,
            "failed_requests": bool(failed_requests),
            "environment_valid": not any("mismatch" in reason for reason in reasons),
            "outliers_included": True,
        },
        "explanation": explanation,
        "reasons": reasons,
        "baseline_files": [str(run.path) for run in baseline],
        "candidate_files": [str(run.path) for run in candidate],
    }


def load_and_compare_directories(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    metric: str = "output_token_throughput",
    thresholds: QualityThresholds = QualityThresholds(),
) -> dict[str, Any]:
    return compare_repeated_runs(
        load_repeated_runs(baseline_dir, metric),
        load_repeated_runs(candidate_dir, metric),
        metric=metric,
        thresholds=thresholds,
    )


def load_json_result(path: Path) -> dict[str, Any]:
    """Small helper used by integrations that need the machine-readable report."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
