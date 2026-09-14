from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BenchmarkResultError(ValueError):
    pass


class IncomparableBenchmarkError(BenchmarkResultError):
    pass


@dataclass(frozen=True)
class BenchmarkEnvironment:
    gpu: str
    vllm_version: str | None
    model: str
    image: str | None


@dataclass(frozen=True)
class LatencyStats:
    mean: float | None
    median: float | None
    p95: float | None


@dataclass(frozen=True)
class RequestCounts:
    successful: float | None
    failed: float | None


@dataclass(frozen=True)
class BenchmarkMetrics:
    ttft_ms: LatencyStats
    tpot_ms: LatencyStats
    output_token_throughput: float | None
    request_throughput: float | None
    requests: RequestCounts
    texts_per_second: float | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: str
    environment: BenchmarkEnvironment
    benchmark_parameters: dict[str, Any]
    metrics: BenchmarkMetrics

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "gpu": self.environment.gpu,
            "vllm_version": self.environment.vllm_version,
            "model": self.environment.model,
            "image": self.environment.image,
            "benchmark_parameters": self.benchmark_parameters,
        }


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkResultError(f"'{field}' must be a JSON object")
    return value


def _require_string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkResultError(
            f"'environment.{field}' must be a non-empty string"
        )
    return value


def _optional_string(mapping: dict[str, Any], field: str) -> str | None:
    value = mapping.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkResultError(
            f"'environment.{field}' must be a string or null"
        )
    return value


def _metric_number(
    mapping: dict[str, Any],
    key: str,
    field: str,
) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkResultError(
            f"'{field}' must be a non-negative number or null"
        )

    number = float(value)
    if not math.isfinite(number):
        return None
    if number < 0:
        raise BenchmarkResultError(
            f"'{field}' must be a non-negative number or null"
        )
    return number


def _latency_stats(
    metrics: dict[str, Any],
    field: str,
) -> LatencyStats:
    raw = metrics.get(field)
    if raw is None:
        raw = {}
    values = _require_mapping(raw, f"metrics.{field}")
    return LatencyStats(
        mean=_metric_number(values, "mean", f"metrics.{field}.mean"),
        median=_metric_number(
            values,
            "median",
            f"metrics.{field}.median",
        ),
        p95=_metric_number(values, "p95", f"metrics.{field}.p95"),
    )


def parse_normalized_benchmark(payload: Any) -> BenchmarkResult:
    root = _require_mapping(payload, "root")
    schema_version = root.get("schema_version")
    if schema_version != "1":
        raise BenchmarkResultError(
            "'schema_version' must be '1' for normalized benchmark results"
        )

    environment_raw = _require_mapping(
        root.get("environment"),
        "environment",
    )
    parameters = _require_mapping(
        root.get("benchmark_parameters"),
        "benchmark_parameters",
    )
    if not parameters:
        raise BenchmarkResultError(
            "'benchmark_parameters' must not be empty"
        )
    metrics = _require_mapping(root.get("metrics"), "metrics")

    requests_raw = metrics.get("requests")
    if requests_raw is None:
        requests_raw = {}
    requests = _require_mapping(requests_raw, "metrics.requests")

    return BenchmarkResult(
        schema_version=schema_version,
        environment=BenchmarkEnvironment(
            gpu=_require_string(environment_raw, "gpu"),
            vllm_version=_optional_string(
                environment_raw,
                "vllm_version",
            ),
            model=_require_string(environment_raw, "model"),
            image=_optional_string(environment_raw, "image"),
        ),
        benchmark_parameters=parameters,
        metrics=BenchmarkMetrics(
            ttft_ms=_latency_stats(metrics, "ttft_ms"),
            tpot_ms=_latency_stats(metrics, "tpot_ms"),
            output_token_throughput=_metric_number(
                metrics,
                "output_token_throughput",
                "metrics.output_token_throughput",
            ),
            request_throughput=_metric_number(
                metrics,
                "request_throughput",
                "metrics.request_throughput",
            ),
            requests=RequestCounts(
                successful=_metric_number(
                    requests,
                    "successful",
                    "metrics.requests.successful",
                ),
                failed=_metric_number(
                    requests,
                    "failed",
                    "metrics.requests.failed",
                ),
            ),
            texts_per_second=_metric_number(
                metrics,
                "texts_per_second",
                "metrics.texts_per_second",
            ),
        ),
    )


def load_normalized_benchmark(path: Path) -> BenchmarkResult:
    try:
        with path.open(encoding="utf-8") as result_file:
            payload = json.load(result_file)
    except OSError as exc:
        raise BenchmarkResultError(f"Could not read '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkResultError(
            f"Invalid JSON in '{path}': {exc.msg}"
        ) from exc

    return parse_normalized_benchmark(payload)


def ensure_comparable(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
) -> None:
    differences: list[str] = []

    if baseline.environment.gpu != candidate.environment.gpu:
        differences.append(
            "environment.gpu "
            f"({baseline.environment.gpu!r} != "
            f"{candidate.environment.gpu!r})"
        )
    if baseline.environment.model != candidate.environment.model:
        differences.append(
            "environment.model "
            f"({baseline.environment.model!r} != "
            f"{candidate.environment.model!r})"
        )
    if baseline.benchmark_parameters != candidate.benchmark_parameters:
        differences.append("benchmark_parameters")

    if differences:
        joined = ", ".join(differences)
        raise IncomparableBenchmarkError(
            f"Benchmark runs are incomparable: {joined} differ"
        )
