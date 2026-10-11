from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from inference_doctor.benchmark_results import BenchmarkResult


class ComparisonStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    UNSTABLE = "UNSTABLE"


@dataclass(frozen=True)
class ComparisonThresholds:
    latency_warn_percent: float = 5.0
    latency_fail_percent: float = 10.0
    throughput_warn_percent: float = 5.0
    throughput_fail_percent: float = 10.0
    max_failed_request_increase: int = 0
    stability_tolerance_percent: float = 5.0

    def validate(self) -> None:
        values = {
            "latency_warn_percent": self.latency_warn_percent,
            "latency_fail_percent": self.latency_fail_percent,
            "throughput_warn_percent": self.throughput_warn_percent,
            "throughput_fail_percent": self.throughput_fail_percent,
            "max_failed_request_increase": self.max_failed_request_increase,
            "stability_tolerance_percent": self.stability_tolerance_percent,
        }
        if any(value < 0 for value in values.values()):
            raise ValueError("Comparison thresholds must be non-negative")
        if self.latency_fail_percent < self.latency_warn_percent:
            raise ValueError(
                "Latency fail threshold must be greater than or equal "
                "to the warn threshold"
            )
        if self.throughput_fail_percent < self.throughput_warn_percent:
            raise ValueError(
                "Throughput fail threshold must be greater than or equal "
                "to the warn threshold"
            )


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    label: str
    unit: str
    baseline: float | None
    candidate: float | None
    absolute_delta: float | None
    percentage_delta: float | None
    status: ComparisonStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "label": self.label,
            "unit": self.unit,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "absolute_delta": self.absolute_delta,
            "percentage_delta": self.percentage_delta,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class BenchmarkComparison:
    baseline_metadata: dict[str, Any]
    candidate_metadata: dict[str, Any]
    thresholds: ComparisonThresholds
    metrics: list[MetricComparison]
    overall_status: ComparisonStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "overall_status": self.overall_status.value,
            "baseline": self.baseline_metadata,
            "candidate": self.candidate_metadata,
            "thresholds": {
                "latency_warn_percent": (
                    self.thresholds.latency_warn_percent
                ),
                "latency_fail_percent": (
                    self.thresholds.latency_fail_percent
                ),
                "throughput_warn_percent": (
                    self.thresholds.throughput_warn_percent
                ),
                "throughput_fail_percent": (
                    self.thresholds.throughput_fail_percent
                ),
                "max_failed_request_increase": (
                    self.thresholds.max_failed_request_increase
                ),
                "stability_tolerance_percent": (
                    self.thresholds.stability_tolerance_percent
                ),
            },
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


def _percentage_delta(
    baseline: float,
    candidate: float,
) -> float | None:
    if baseline == 0:
        return None
    return round(((candidate - baseline) / baseline) * 100, 6)


def _compare_percent_metric(
    *,
    metric: str,
    label: str,
    unit: str,
    baseline: float | None,
    candidate: float | None,
    lower_is_better: bool,
    warn_percent: float,
    fail_percent: float,
    same_version: bool,
    stability_tolerance_percent: float,
) -> MetricComparison:
    if baseline is None or candidate is None:
        return MetricComparison(
            metric=metric,
            label=label,
            unit=unit,
            baseline=baseline,
            candidate=candidate,
            absolute_delta=None,
            percentage_delta=None,
            status=ComparisonStatus.WARN,
        )

    absolute_delta = candidate - baseline
    percentage_delta = _percentage_delta(baseline, candidate)

    if same_version:
        if percentage_delta is None:
            stable = candidate == baseline
        else:
            stable = abs(percentage_delta) <= stability_tolerance_percent
        status = (
            ComparisonStatus.PASS
            if stable
            else ComparisonStatus.UNSTABLE
        )
    elif percentage_delta is None:
        if candidate == baseline:
            regression_percent = 0.0
        elif lower_is_better:
            regression_percent = float("inf")
        else:
            regression_percent = float("-inf")
    else:
        regression_percent = (
            percentage_delta
            if lower_is_better
            else -percentage_delta
        )

        if regression_percent >= fail_percent:
            status = ComparisonStatus.FAIL
        elif regression_percent >= warn_percent:
            status = ComparisonStatus.WARN
        else:
            status = ComparisonStatus.PASS

    return MetricComparison(
        metric=metric,
        label=label,
        unit=unit,
        baseline=baseline,
        candidate=candidate,
        absolute_delta=absolute_delta,
        percentage_delta=percentage_delta,
        status=status,
    )


def _compare_failed_requests(
    baseline: float | None,
    candidate: float | None,
    max_increase: int,
    same_version: bool,
) -> MetricComparison:
    if baseline is None or candidate is None:
        return MetricComparison(
            metric="requests_failed",
            label="Failed requests",
            unit="requests",
            baseline=baseline,
            candidate=candidate,
            absolute_delta=None,
            percentage_delta=None,
            status=ComparisonStatus.WARN,
        )

    absolute_delta = candidate - baseline
    if same_version and absolute_delta != 0:
        status = ComparisonStatus.UNSTABLE
    elif absolute_delta > max_increase:
        status = ComparisonStatus.FAIL
    elif absolute_delta > 0:
        status = ComparisonStatus.WARN
    else:
        status = ComparisonStatus.PASS

    return MetricComparison(
        metric="requests_failed",
        label="Failed requests",
        unit="requests",
        baseline=baseline,
        candidate=candidate,
        absolute_delta=absolute_delta,
        percentage_delta=_percentage_delta(baseline, candidate),
        status=status,
    )


def compare_benchmarks(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    thresholds: ComparisonThresholds,
    *,
    same_version: bool = False,
) -> BenchmarkComparison:
    thresholds.validate()
    metrics: list[MetricComparison] = []

    latency_metrics = [
        (
            "ttft_mean_ms",
            "TTFT mean (ms)",
            baseline.metrics.ttft_ms.mean,
            candidate.metrics.ttft_ms.mean,
        ),
        (
            "ttft_median_ms",
            "TTFT median (ms)",
            baseline.metrics.ttft_ms.median,
            candidate.metrics.ttft_ms.median,
        ),
        (
            "ttft_p95_ms",
            "TTFT p95 (ms)",
            baseline.metrics.ttft_ms.p95,
            candidate.metrics.ttft_ms.p95,
        ),
        (
            "tpot_mean_ms",
            "TPOT mean (ms)",
            baseline.metrics.tpot_ms.mean,
            candidate.metrics.tpot_ms.mean,
        ),
        (
            "tpot_median_ms",
            "TPOT median (ms)",
            baseline.metrics.tpot_ms.median,
            candidate.metrics.tpot_ms.median,
        ),
        (
            "tpot_p95_ms",
            "TPOT p95 (ms)",
            baseline.metrics.tpot_ms.p95,
            candidate.metrics.tpot_ms.p95,
        ),
    ]
    for metric, label, baseline_value, candidate_value in latency_metrics:
        metrics.append(
            _compare_percent_metric(
                metric=metric,
                label=label,
                unit="ms",
                baseline=baseline_value,
                candidate=candidate_value,
                lower_is_better=True,
                warn_percent=thresholds.latency_warn_percent,
                fail_percent=thresholds.latency_fail_percent,
                same_version=same_version,
                stability_tolerance_percent=(
                    thresholds.stability_tolerance_percent
                ),
            )
        )

    throughput_metrics = [
        (
            "output_token_throughput",
            "Output tok/s",
            "tokens/s",
            baseline.metrics.output_token_throughput,
            candidate.metrics.output_token_throughput,
        ),
        (
            "request_throughput",
            "Requests/s",
            "requests/s",
            baseline.metrics.request_throughput,
            candidate.metrics.request_throughput,
        ),
        (
            "requests_successful",
            "Successful requests",
            "requests",
            baseline.metrics.requests.successful,
            candidate.metrics.requests.successful,
        ),
    ]
    for (
        metric,
        label,
        unit,
        baseline_value,
        candidate_value,
    ) in throughput_metrics:
        metrics.append(
            _compare_percent_metric(
                metric=metric,
                label=label,
                unit=unit,
                baseline=baseline_value,
                candidate=candidate_value,
                lower_is_better=False,
                warn_percent=thresholds.throughput_warn_percent,
                fail_percent=thresholds.throughput_fail_percent,
                same_version=same_version,
                stability_tolerance_percent=(
                    thresholds.stability_tolerance_percent
                ),
            )
        )

    metrics.append(
        _compare_failed_requests(
            baseline.metrics.requests.failed,
            candidate.metrics.requests.failed,
            thresholds.max_failed_request_increase,
            same_version,
        )
    )

    if any(metric.status == ComparisonStatus.UNSTABLE for metric in metrics):
        overall_status = ComparisonStatus.UNSTABLE
    elif any(metric.status == ComparisonStatus.FAIL for metric in metrics):
        overall_status = ComparisonStatus.FAIL
    else:
        overall_status = ComparisonStatus.PASS

    return BenchmarkComparison(
        baseline_metadata=baseline.metadata_dict(),
        candidate_metadata=candidate.metadata_dict(),
        thresholds=thresholds,
        metrics=metrics,
        overall_status=overall_status,
    )
