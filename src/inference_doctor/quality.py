from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_doctor.adapters import UNSPECIFIED, LoadedDirectory, SourceOptions, load_directory
from inference_doctor.benchmark_results import (
    BenchmarkResult,
    BenchmarkResultError,
    IncomparableBenchmarkError,
    ensure_comparable,
)
from inference_doctor.stats import welch_log_ratio_interval

QUALITY_SCHEMA_VERSION = "inference-ci-quality-2"


class QualityInputError(BenchmarkResultError):
    """Raised when repeated-run input cannot be evaluated."""


@dataclass(frozen=True)
class QualityThresholds:
    """How compare-runs decides.

    The candidate FAILs when the whole confidence interval for the regression is at or
    past `regression_percent`, and PASSes when the whole interval is below it.
    `stability_cv_percent` is an optional extra ceiling on either side's CV; by default
    the interval alone accounts for noise.
    """

    regression_percent: float = 10.0
    confidence: float = 0.90
    min_repetitions: int = 3
    stability_cv_percent: float | None = None

    def validate(self) -> None:
        if self.regression_percent <= 0:
            raise ValueError("The regression threshold must be positive")
        if not 0.5 <= self.confidence < 1:
            raise ValueError("The confidence level must be at least 0.5 and below 1")
        if self.min_repetitions < 2:
            raise ValueError("At least two repetitions per side are required")
        if self.stability_cv_percent is not None and self.stability_cv_percent < 0:
            raise ValueError("The stability CV ceiling must be non-negative")


@dataclass(frozen=True)
class RegressionDecision:
    """The interval-based verdict on two sets of valid repeated measurements.

    Regression percentages are signed so that positive is worse for the metric: a
    throughput drop and a latency rise are both positive.
    """

    status: str  # PASS, FAIL or INCONCLUSIVE
    regression_percent: float
    regression_percent_low: float
    regression_percent_high: float
    ratio: float
    ratio_low: float
    ratio_high: float
    degrees_of_freedom: float | None
    reason: str


def _signed(value: float) -> str:
    if math.isfinite(value):
        return f"{value:+.2f}%"
    return "+inf%" if value > 0 else "-inf%"


def decide_regression(
    baseline: list[float],
    candidate: list[float],
    *,
    lower_is_better: bool,
    thresholds: QualityThresholds,
) -> RegressionDecision:
    """Compare candidate with baseline using a confidence interval on the ratio of means."""
    baseline_mean = statistics.fmean(baseline)
    candidate_mean = statistics.fmean(candidate)
    degrees_of_freedom: float | None = None
    if baseline_mean == 0 and candidate_mean == 0:
        ratio = low = high = 1.0
    elif baseline_mean == 0 or candidate_mean == 0:
        # No ratio on the log scale; the change is total in one direction.
        ratio = low = high = 0.0 if candidate_mean == 0 else math.inf
    else:
        interval = welch_log_ratio_interval(baseline, candidate, thresholds.confidence)
        ratio, low, high = interval.ratio, interval.low, interval.high
        degrees_of_freedom = interval.degrees_of_freedom

    def regression(value: float) -> float:
        if math.isinf(value):
            return math.inf if lower_is_better else -math.inf
        return (value - 1) * 100 if lower_is_better else (1 - value) * 100

    bound_low, bound_high = sorted((regression(low), regression(high)))
    level = f"{thresholds.confidence:.0%}"
    threshold = f"{thresholds.regression_percent:g}%"
    interval_text = (
        f"the {level} confidence interval for the regression, "
        f"{_signed(bound_low)} to {_signed(bound_high)},"
    )
    if bound_low >= thresholds.regression_percent:
        status = "FAIL"
        reason = f"{interval_text} is entirely at or past the {threshold} threshold"
    elif bound_high < thresholds.regression_percent:
        status = "PASS"
        reason = f"{interval_text} is entirely below the {threshold} threshold"
    else:
        status = "INCONCLUSIVE"
        reason = (
            f"{interval_text} includes the {threshold} threshold; more repetitions would narrow it"
        )
    return RegressionDecision(
        status=status,
        regression_percent=regression(ratio),
        regression_percent_low=bound_low,
        regression_percent_high=bound_high,
        ratio=ratio,
        ratio_low=low,
        ratio_high=high,
        degrees_of_freedom=degrees_of_freedom,
        reason=reason,
    )


@dataclass(frozen=True)
class RepeatedRun:
    path: Path
    result: BenchmarkResult
    value: float | None
    record: int | None = None  # line number within a JSONL file

    @property
    def label(self) -> str:
        return str(self.path) if self.record is None else f"{self.path}:{self.record}"


LATENCY_FAMILIES = {
    "ttft": "TTFT",
    "tpot": "TPOT",
    "itl": "ITL",
    "e2el": "End-to-end latency",
}
LATENCY_STATISTICS = {"mean": "mean", "median": "median", "p95": "p95", "p99": "p99"}

# metric name -> (label, lower is better)
METRICS: dict[str, tuple[str, bool]] = {
    "output_token_throughput": ("Output token throughput", False),
    "total_token_throughput": ("Total token throughput", False),
    "request_throughput": ("Request throughput", False),
    "texts_per_second": ("Texts per second", False),
    **{
        f"{family}_{statistic}_ms": (f"{label} {statistic_label}", True)
        for family, label in LATENCY_FAMILIES.items()
        for statistic, statistic_label in LATENCY_STATISTICS.items()
    },
    "requests_successful": ("Successful requests", False),
    "requests_failed": ("Failed requests", True),
}


def metric_choices() -> tuple[str, ...]:
    return tuple(METRICS)


def _metric_value(result: BenchmarkResult, metric: str) -> float | None:
    metrics = result.metrics
    if metric == "requests_successful":
        return metrics.requests.successful
    if metric == "requests_failed":
        return metrics.requests.failed
    if metric.endswith("_ms"):
        family, statistic = metric.removesuffix("_ms").rsplit("_", 1)
        return getattr(getattr(metrics, f"{family}_ms"), statistic)
    return getattr(metrics, metric)


def load_repeated_runs(
    directory: Path,
    metric: str,
    *,
    format: str = "auto",
    options: SourceOptions | None = None,
) -> tuple[list[RepeatedRun], LoadedDirectory]:
    if metric not in METRICS:
        raise QualityInputError(f"Unknown metric '{metric}'. Choose one of: {', '.join(METRICS)}")
    loaded = _load(directory, format, options)
    return _runs(loaded, metric), loaded


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
            index for index, value in enumerate(numbers) if abs(value - median) > 3 * mad
        )
    return [runs[index].label for index in sorted(flagged)]


def _cv_percent(mean: float, standard_deviation: float | None) -> float | None:
    if standard_deviation is None:
        return None
    if standard_deviation == 0:
        return 0.0  # identical repetitions, including all zeros, are perfectly stable
    return standard_deviation / mean * 100


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
            "cv_percent": _cv_percent(mean, standard_deviation),
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

    label, lower_is_better = METRICS[metric]
    baseline_stats = summarize_runs(baseline)
    candidate_stats = summarize_runs(candidate)

    # Validity checks. Any failure means there is no verdict, whatever the numbers say.
    environment_reasons = _validate_environment(baseline, candidate)
    reasons = list(environment_reasons)
    notes: list[str] = []
    failed_requests = [
        (side, run.label)
        for side, runs in (("baseline", baseline), ("candidate", candidate))
        for run in runs
        if run.result.failed_requests_reported
        and (run.result.metrics.requests.failed is None or run.result.metrics.requests.failed > 0)
    ]
    if failed_requests:
        reasons.append("one or more repetitions recorded failed or unavailable requests")
    unreported = sorted(
        {
            run.result.source_format
            for run in (*baseline, *candidate)
            if not run.result.failed_requests_reported
        }
    )
    if unreported:
        notes.append(
            f"{', '.join(unreported)} results do not record failed requests, so failures "
            "could not be checked"
        )
    for name in ("gpu", "model"):
        if any(
            getattr(run.result.environment, name) == UNSPECIFIED for run in (*baseline, *candidate)
        ):
            notes.append(
                f"the {name} is not recorded in the results, so it could not be compared; "
                f"pass --{name} to record it"
            )
    enough_repetitions = True
    for side, stats in (("baseline", baseline_stats), ("candidate", candidate_stats)):
        if stats["missing_values"]:
            reasons.append(
                f"{label} was missing or non-finite in {stats['missing_values']} "
                f"of {stats['count']} {side} repetition(s)"
            )
        if stats["count"] < thresholds.min_repetitions:
            enough_repetitions = False
            reasons.append(
                f"{side} has {stats['count']} repetition(s); at least "
                f"{thresholds.min_repetitions} are required"
            )

    stable = {"baseline": True, "candidate": True}
    if thresholds.stability_cv_percent is not None:
        for side, stats in (("baseline", baseline_stats), ("candidate", candidate_stats)):
            cv = stats["cv_percent"]
            if cv is not None and cv > thresholds.stability_cv_percent:
                stable[side] = False
                reasons.append(
                    f"{side} CV {cv:.2f}% exceeded the {thresholds.stability_cv_percent:g}% "
                    "stability ceiling"
                )

    # The interval is reported whenever it can be computed, even when a validity check
    # failed, so the numbers appear next to the reason there is no verdict.
    decision: RegressionDecision | None = None
    if len(baseline_stats["values"]) >= 2 and len(candidate_stats["values"]) >= 2:
        decision = decide_regression(
            baseline_stats["values"],
            candidate_stats["values"],
            lower_is_better=lower_is_better,
            thresholds=thresholds,
        )

    if reasons or decision is None:
        status = "INCONCLUSIVE"
        if decision is not None:
            reasons.append(f"for reference, {decision.reason}")
    else:
        status = decision.status
        if status != "PASS":
            reasons.append(decision.reason)

    if decision is not None and decision.regression_percent <= -thresholds.regression_percent:
        notes.append(
            f"the candidate is {-decision.regression_percent:.2f}% better than the baseline, "
            "more than the regression threshold; check that both sides measured the same "
            "workload"
        )

    baseline_mean = baseline_stats["mean"]
    candidate_mean = candidate_stats["mean"]
    delta_percent = None
    if baseline_mean and candidate_mean is not None:
        delta_percent = (candidate_mean - baseline_mean) / baseline_mean * 100

    if reasons:
        explanation = "; ".join(reasons)
    else:
        explanation = decision.reason if decision is not None else ""
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "result": status,
        "metric": metric,
        "metric_label": label,
        "lower_is_better": lower_is_better,
        "environment": {
            "baseline": baseline[0].result.metadata_dict(),
            "candidate": candidate[0].result.metadata_dict(),
        },
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "delta_percent": delta_percent,
        "regression_percent": _finite(decision.regression_percent) if decision else None,
        "confidence_interval": _interval_dict(decision, thresholds.confidence),
        "thresholds": {
            "regression_percent": thresholds.regression_percent,
            "confidence": thresholds.confidence,
            "min_repetitions": thresholds.min_repetitions,
            "stability_cv_percent": thresholds.stability_cv_percent,
        },
        "measurement_quality": {
            "enough_repetitions": enough_repetitions,
            "baseline_stable": stable["baseline"],
            "candidate_stable": stable["candidate"],
            "failed_requests": bool(failed_requests),
            "environment_valid": not environment_reasons,
            "outliers_included": True,
        },
        "explanation": explanation,
        "reasons": reasons,
        "notes": notes,
        "baseline_files": [run.label for run in baseline],
        "candidate_files": [run.label for run in candidate],
    }


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _interval_dict(decision: RegressionDecision | None, confidence: float) -> dict[str, Any]:
    if decision is None:
        return {"level": confidence, "available": False}
    return {
        "level": confidence,
        "available": True,
        "method": "Welch t interval on the log ratio of means",
        "regression_percent_low": _finite(decision.regression_percent_low),
        "regression_percent_high": _finite(decision.regression_percent_high),
        "ratio": _finite(decision.ratio),
        "ratio_low": _finite(decision.ratio_low),
        "ratio_high": _finite(decision.ratio_high),
        "degrees_of_freedom": decision.degrees_of_freedom,
    }


def _load(directory: Path, format: str, options: SourceOptions | None) -> LoadedDirectory:
    try:
        return load_directory(directory, format, options)
    except BenchmarkResultError as exc:
        raise QualityInputError(str(exc)) from exc


def _runs(loaded: LoadedDirectory, metric: str) -> list[RepeatedRun]:
    return [
        RepeatedRun(item.path, item.result, _metric_value(item.result, metric), item.record)
        for item in loaded.records
    ]


def combine_results(results: list[str]) -> str:
    """FAIL if any metric fails, else INCONCLUSIVE if any is, else PASS."""
    if "FAIL" in results:
        return "FAIL"
    if "INCONCLUSIVE" in results:
        return "INCONCLUSIVE"
    return "PASS"


def load_and_compare_directories(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    metric: str | list[str] = "output_token_throughput",
    thresholds: QualityThresholds = QualityThresholds(),
    format: str = "auto",
    options: SourceOptions | None = None,
) -> dict[str, Any]:
    """Compare one metric, or several.

    With one metric the report is that metric's comparison. With several, the report
    has the combined `result` and one full comparison per metric under `comparisons`.
    """
    metrics = [metric] if isinstance(metric, str) else list(dict.fromkeys(metric))
    if not metrics:
        raise QualityInputError("At least one metric is required")
    unknown = [name for name in metrics if name not in METRICS]
    if unknown:
        raise QualityInputError(
            f"Unknown metric '{unknown[0]}'. Choose one of: {', '.join(METRICS)}"
        )
    baseline_source = _load(baseline_dir, format, options)
    candidate_source = _load(candidate_dir, format, options)
    source = {
        "baseline": {"format": baseline_source.format, "ignored_files": baseline_source.ignored},
        "candidate": {"format": candidate_source.format, "ignored_files": candidate_source.ignored},
    }
    ignored = len(baseline_source.ignored) + len(candidate_source.ignored)
    ignored_note = (
        f"{ignored} JSON file(s) or line(s) in the input directories were not read; "
        "they are listed under input.ignored_files in --json output"
    )

    comparisons = [
        compare_repeated_runs(
            _runs(baseline_source, name),
            _runs(candidate_source, name),
            metric=name,
            thresholds=thresholds,
        )
        for name in metrics
    ]
    if len(comparisons) == 1:
        report = comparisons[0]
        report["input"] = source
        if ignored:
            report["notes"].append(ignored_note)
        return report

    notes: list[str] = []
    for comparison in comparisons:
        for note in comparison["notes"]:
            if note.startswith("the candidate is "):
                note = f"{comparison['metric']}: {note}"
            if note not in notes:
                notes.append(note)
    if ignored:
        notes.append(ignored_note)
    notes.append(
        f"{len(metrics)} metrics were checked; with no real change, the chance that at "
        f"least one FAILs is up to {len(metrics)} times the per-metric rate"
    )
    reasons = [
        f"{comparison['metric']}: {reason}"
        for comparison in comparisons
        if comparison["result"] != "PASS"
        for reason in comparison["reasons"]
    ]
    result = combine_results([comparison["result"] for comparison in comparisons])
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "result": result,
        "metrics": metrics,
        "comparisons": comparisons,
        "thresholds": comparisons[0]["thresholds"],
        "environment": comparisons[0]["environment"],
        "input": source,
        "reasons": reasons,
        "notes": notes,
        "explanation": "; ".join(reasons) if reasons else "every metric passed",
    }


def load_json_result(path: Path) -> dict[str, Any]:
    """Small helper used by integrations that need the machine-readable report."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
