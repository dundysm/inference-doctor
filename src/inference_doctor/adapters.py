"""Read saved results from common LLM benchmark tools as BenchmarkResult records.

Supported formats:

    normalized  Inference Doctor's own schema (docs/benchmark-format.md)
    vllm        `vllm bench serve --save-result` (also --append-result JSONL)
    aiperf      NVIDIA AIPerf profile_export_aiperf.json, and GenAI-Perf's
                profile_export_genai_perf.json, which has the same layout
    guidellm    GuideLLM benchmarks.json; one benchmark of the sweep is selected
    sglang      SGLang bench_serving JSONL (one run per line)

One file, or one line of a JSONL file, is one repetition. Field names were checked
against each tool's source on the main branch in October 2026.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference_doctor.benchmark_results import (
    BenchmarkEnvironment,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkResultError,
    LatencyStats,
    RequestCounts,
    parse_normalized_benchmark,
)

FORMATS = ("auto", "normalized", "vllm", "aiperf", "guidellm", "sglang")
NORMALIZED_PATTERNS = ("normalized.json", "*.normalized.json", "repetition-*.json")
UNSPECIFIED = "unspecified"

# Files that sit next to results but are not one repetition.
_SKIPPED_SUFFIXES = (".pytorch.json", "_aggregate.json", "_collated.json")
_SKIPPED_NAMES = {"summary.json", "run.json"}


@dataclass(frozen=True)
class SourceOptions:
    """Values that the result files may not carry themselves."""

    gpu: str | None = None
    model: str | None = None
    guidellm_benchmark: int | None = None


@dataclass(frozen=True)
class LoadedRecord:
    path: Path
    record: int | None  # line number in a JSONL file, or None for a whole file
    result: BenchmarkResult

    @property
    def label(self) -> str:
        return str(self.path) if self.record is None else f"{self.path}:{self.record}"


@dataclass
class LoadedDirectory:
    format: str
    records: list[LoadedRecord]
    ignored: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small helpers


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _get(mapping: Any, *path: str | int) -> Any:
    current = mapping
    for key in path:
        if isinstance(key, int):
            if not isinstance(current, list) or len(current) <= key:
                return None
        elif not isinstance(current, dict):
            return None
        current = current[key] if isinstance(key, int) else current.get(key)
    return current


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return text or None


def _environment(
    *,
    gpu: Any,
    model: Any,
    options: SourceOptions,
    vllm_version: Any = None,
    image: Any = None,
) -> BenchmarkEnvironment:
    return BenchmarkEnvironment(
        gpu=_text(gpu) or options.gpu or UNSPECIFIED,
        vllm_version=_text(vllm_version),
        model=_text(model) or options.model or UNSPECIFIED,
        image=_text(image),
    )


def _parameters(values: dict[str, Any]) -> dict[str, Any]:
    kept = {key: value for key, value in values.items() if value is not None}
    for key, value in kept.items():
        if isinstance(value, float) and not math.isfinite(value):
            kept[key] = str(value)  # request_rate may be inf
    if not kept:
        raise BenchmarkResultError("the result records no benchmark parameters")
    return kept


# ---------------------------------------------------------------------------
# vLLM and SGLang: flat dicts with mean_/median_/pNN_ prefixed keys in ms


def _flat_latency(record: dict[str, Any], name: str) -> LatencyStats:
    return LatencyStats(
        mean=_number(record.get(f"mean_{name}_ms")),
        median=_number(record.get(f"median_{name}_ms", record.get(f"p50_{name}_ms"))),
        p95=_number(record.get(f"p95_{name}_ms")),
        p99=_number(record.get(f"p99_{name}_ms")),
    )


def _from_vllm(record: dict[str, Any], options: SourceOptions) -> BenchmarkResult:
    # `--metadata k=v` adds top-level string keys; gpu, vllm_version and image are
    # read from there when present. A user `label` is not a workload parameter.
    failed = record.get("failed")
    return BenchmarkResult(
        schema_version="vllm",
        environment=_environment(
            gpu=record.get("gpu"),
            model=record.get("model_id"),
            options=options,
            vllm_version=record.get("vllm_version"),
            image=record.get("image"),
        ),
        benchmark_parameters=_parameters(
            {
                key: record.get(key)
                for key in (
                    "backend",
                    "endpoint_type",
                    "tokenizer_id",
                    "num_prompts",
                    "request_rate",
                    "burstiness",
                    "max_concurrency",
                    "ramp_up_strategy",
                    "ramp_up_start_rps",
                    "ramp_up_end_rps",
                )
            }
        ),
        metrics=BenchmarkMetrics(
            ttft_ms=_flat_latency(record, "ttft"),
            tpot_ms=_flat_latency(record, "tpot"),
            itl_ms=_flat_latency(record, "itl"),
            e2el_ms=_flat_latency(record, "e2el"),
            output_token_throughput=_number(record.get("output_throughput")),
            total_token_throughput=_number(record.get("total_token_throughput")),
            request_throughput=_number(record.get("request_throughput")),
            requests=RequestCounts(
                successful=_number(record.get("completed")), failed=_number(failed)
            ),
        ),
        source_format="vllm",
        failed_requests_reported="failed" in record,
    )


def _from_sglang(record: dict[str, Any], options: SourceOptions) -> BenchmarkResult:
    # SGLang writes no failed-request count. With --output-details, `errors` holds one
    # entry per request and an empty string means it succeeded.
    errors = record.get("errors")
    failed: float | None = None
    if isinstance(errors, list):
        failed = float(sum(1 for error in errors if error))
    server_info = record.get("server_info")
    return BenchmarkResult(
        schema_version="sglang",
        environment=_environment(
            gpu=None,
            model=_get(server_info, "model_path"),
            options=options,
        ),
        benchmark_parameters=_parameters(
            {
                key: record.get(key)
                for key in (
                    "backend",
                    "dataset_name",
                    "request_rate",
                    "max_concurrency",
                    "sharegpt_output_len",
                    "random_input_len",
                    "random_output_len",
                    "random_range_ratio",
                )
            }
        ),
        metrics=BenchmarkMetrics(
            ttft_ms=_flat_latency(record, "ttft"),
            tpot_ms=_flat_latency(record, "tpot"),
            itl_ms=_flat_latency(record, "itl"),
            e2el_ms=_flat_latency(record, "e2e_latency"),
            output_token_throughput=_number(record.get("output_throughput")),
            total_token_throughput=_number(record.get("total_throughput")),
            request_throughput=_number(record.get("request_throughput")),
            requests=RequestCounts(successful=_number(record.get("completed")), failed=failed),
        ),
        source_format="sglang",
        failed_requests_reported=failed is not None,
    )


# ---------------------------------------------------------------------------
# AIPerf and GenAI-Perf: each metric is a top-level {unit, avg, p50, p95, p99, ...}

_TIME_UNITS_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0, "sec": 1000.0}


def _aiperf_latency(record: dict[str, Any], tag: str) -> LatencyStats:
    stats = record.get(tag)
    if not isinstance(stats, dict):
        return LatencyStats()
    unit = str(stats.get("unit", "ms")).lower()
    scale = _TIME_UNITS_TO_MS.get(unit)
    if scale is None:
        raise BenchmarkResultError(f"'{tag}' has unit '{unit}', expected a time unit")

    def scaled(key: str) -> float | None:
        value = _number(stats.get(key))
        return None if value is None else value * scale

    return LatencyStats(
        mean=scaled("avg"), median=scaled("p50"), p95=scaled("p95"), p99=scaled("p99")
    )


def _aiperf_value(record: dict[str, Any], tag: str) -> float | None:
    return _number(_get(record, tag, "avg"))


def _from_aiperf(record: dict[str, Any], options: SourceOptions) -> BenchmarkResult:
    successful = _aiperf_value(record, "request_count")
    failed = _aiperf_value(record, "error_request_count")
    completed = _aiperf_value(record, "completed_request_count")
    reported = failed is not None
    if failed is None and completed is not None and successful is not None:
        failed, reported = max(completed - successful, 0.0), True

    config = record.get("input_config") or {}
    model = _get(config, "models", "items", 0, "name") or _get(config, "model_names", 0)
    gpu = None
    for endpoint in (_get(record, "telemetry_data", "endpoints") or {}).values():
        for device in (_get(endpoint, "gpus") or {}).values():
            gpu = gpu or _get(device, "gpu_name")
    phases = [
        {key: phase.get(key) for key in ("name", "type", "concurrency", "rate", "requests")}
        for phase in (_get(config, "phases") or [])
        if isinstance(phase, dict)
    ]
    return BenchmarkResult(
        schema_version="aiperf",
        environment=_environment(gpu=gpu, model=model, options=options),
        benchmark_parameters=_parameters(
            {
                "endpoint_type": _get(config, "endpoint", "type"),
                "streaming": _get(config, "endpoint", "streaming"),
                "phases": phases or None,
                # GenAI-Perf keeps load settings at the top of input_config.
                **{
                    key: config.get(key)
                    for key in (
                        "concurrency",
                        "request_rate",
                        "request_count",
                        "num_dataset_entries",
                        "synthetic_input_tokens_mean",
                        "output_tokens_mean",
                    )
                    if isinstance(config, dict)
                },
            }
        ),
        metrics=BenchmarkMetrics(
            ttft_ms=_aiperf_latency(record, "time_to_first_token"),
            # AIPerf's inter_token_latency is the per-request average gap after the
            # first token, which is what vLLM calls TPOT.
            tpot_ms=_aiperf_latency(record, "inter_token_latency"),
            e2el_ms=_aiperf_latency(record, "request_latency"),
            output_token_throughput=_aiperf_value(record, "output_token_throughput"),
            total_token_throughput=_aiperf_value(record, "total_token_throughput"),
            request_throughput=_aiperf_value(record, "request_throughput"),
            requests=RequestCounts(successful=successful, failed=failed),
        ),
        source_format="aiperf",
        failed_requests_reported=reported,
    )


# ---------------------------------------------------------------------------
# GuideLLM: benchmarks[] of a sweep, each metric split by request status


def _guidellm_stats(metrics: dict[str, Any], name: str, scale: float = 1.0) -> LatencyStats:
    stats = _get(metrics, name, "successful")
    if not isinstance(stats, dict):
        return LatencyStats()

    def scaled(*path: str) -> float | None:
        value = _number(_get(stats, *path))
        return None if value is None else value * scale

    return LatencyStats(
        mean=scaled("mean"),
        median=scaled("median"),
        p95=scaled("percentiles", "p95"),
        p99=scaled("percentiles", "p99"),
    )


def _guidellm_description(benchmark: Any) -> str:
    strategy = _get(benchmark, "config", "strategy") or {}
    parts = [str(strategy.get("type_", "?"))]
    for key in ("rate", "streams", "max_concurrency"):
        if strategy.get(key) is not None:
            parts.append(f"{key}={strategy[key]}")
    return " ".join(parts)


def _from_guidellm(record: dict[str, Any], options: SourceOptions) -> BenchmarkResult:
    benchmarks = record.get("benchmarks") or []
    if options.guidellm_benchmark is None:
        if len(benchmarks) != 1:
            listing = "; ".join(
                f"{index}: {_guidellm_description(benchmark)}"
                for index, benchmark in enumerate(benchmarks)
            )
            raise BenchmarkResultError(
                f"the GuideLLM file has {len(benchmarks)} benchmarks; choose one with "
                f"--guidellm-benchmark INDEX ({listing})"
            )
        index = 0
    else:
        index = options.guidellm_benchmark
        if not 0 <= index < len(benchmarks):
            raise BenchmarkResultError(
                f"--guidellm-benchmark {index} is out of range; the file has "
                f"{len(benchmarks)} benchmark(s)"
            )
    benchmark = benchmarks[index]
    metrics = benchmark.get("metrics") or {}
    totals = metrics.get("request_totals") or {}
    strategy = _get(benchmark, "config", "strategy") or {}
    return BenchmarkResult(
        schema_version="guidellm",
        environment=_environment(
            gpu=None,
            model=_get(benchmark, "config", "backend", "model")
            or _get(record, "config", "spec", "backend", "model"),
            options=options,
        ),
        benchmark_parameters=_parameters(
            {
                "strategy": {
                    key: value
                    for key, value in strategy.items()
                    if key in {"type_", "rate", "streams", "max_concurrency"} and value is not None
                }
                or None,
                "data": _get(record, "config", "spec", "data"),
                "backend": _get(benchmark, "config", "backend", "kind"),
            }
        ),
        metrics=BenchmarkMetrics(
            ttft_ms=_guidellm_stats(metrics, "time_to_first_token_ms"),
            tpot_ms=_guidellm_stats(metrics, "time_per_output_token_ms"),
            itl_ms=_guidellm_stats(metrics, "inter_token_latency_ms"),
            # request_latency is in seconds, unlike the *_ms metrics.
            e2el_ms=_guidellm_stats(metrics, "request_latency", scale=1000.0),
            output_token_throughput=_number(
                _get(metrics, "output_tokens_per_second", "successful", "mean")
            ),
            total_token_throughput=_number(
                _get(metrics, "tokens_per_second", "successful", "mean")
            ),
            request_throughput=_number(_get(metrics, "requests_per_second", "successful", "mean")),
            # Incomplete requests were cut off by the end of a time-bounded run; they
            # are not failures.
            requests=RequestCounts(
                successful=_number(totals.get("successful")),
                failed=_number(totals.get("errored")),
            ),
        ),
        source_format="guidellm",
        failed_requests_reported="errored" in totals,
    )


# ---------------------------------------------------------------------------
# Detection and loading

Converter = Callable[[dict[str, Any], SourceOptions], BenchmarkResult]


def _from_normalized(record: dict[str, Any], options: SourceOptions) -> BenchmarkResult:
    return parse_normalized_benchmark(record)


CONVERTERS: dict[str, Converter] = {
    "normalized": _from_normalized,
    "vllm": _from_vllm,
    "aiperf": _from_aiperf,
    "guidellm": _from_guidellm,
    "sglang": _from_sglang,
}


def detect_format(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    if "schema_version" in record and "environment" in record and "metrics" in record:
        return "normalized"
    if isinstance(record.get("benchmarks"), list) and (
        "guidellm_version" in (record.get("metadata") or {})
        or any(isinstance(b, dict) and "type_" in b for b in record["benchmarks"])
    ):
        return "guidellm"
    if "aiperf_version" in record or any(
        isinstance(record.get(tag), dict) and "avg" in record[tag]
        for tag in ("time_to_first_token", "request_throughput", "request_latency")
    ):
        return "aiperf"
    if "completed" in record and ("request_throughput" in record or "mean_ttft_ms" in record):
        if "model_id" in record or "endpoint_type" in record or "failed" in record:
            return "vllm"
        if "server_info" in record or "dataset_name" in record or "total_throughput" in record:
            return "sglang"
    return None


def _read_records(path: Path) -> Iterator[tuple[int | None, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkResultError(f"Could not read '{path}': {exc}") from exc
    try:
        yield None, json.loads(text)
        return
    except json.JSONDecodeError as exc:
        if path.suffix != ".jsonl" and "Extra data" not in exc.msg:
            raise BenchmarkResultError(f"Invalid JSON in '{path}': {exc.msg}") from exc
    # JSON Lines: vLLM --append-result and SGLang write one run per line.
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkResultError(
                    f"Invalid JSON on line {number} of '{path}': {exc.msg}"
                ) from exc


def _is_normalized_name(path: Path) -> bool:
    name = path.name
    return (
        name == "normalized.json"
        or name.endswith(".normalized.json")
        or (name.startswith("repetition-") and name.endswith(".json"))
    )


def _is_skipped(path: Path) -> bool:
    name = path.name.lower()
    return name in _SKIPPED_NAMES or "aggregate" in name or name.endswith(_SKIPPED_SUFFIXES)


def load_directory(
    directory: Path,
    format: str = "auto",
    options: SourceOptions | None = None,
) -> LoadedDirectory:
    """Load every repetition under `directory`, searched recursively.

    With format "auto", files named like normalized results are used if there are
    any, exactly as before adapters existed. Otherwise every .json and .jsonl file is
    read and its format detected; files in no known format are listed as ignored.
    """
    if format not in FORMATS:
        raise BenchmarkResultError(
            f"Unknown format '{format}'. Choose one of: {', '.join(FORMATS)}"
        )
    if not directory.is_dir():
        raise BenchmarkResultError(f"Repeated-run input is not a directory: '{directory}'")
    options = options or SourceOptions()

    every = sorted(
        path
        for pattern in ("*.json", "*.jsonl")
        for path in directory.rglob(pattern)
        if path.is_file()
    )
    named = [path for path in every if _is_normalized_name(path) and not _is_skipped(path)]
    if format == "normalized" or (format == "auto" and named):
        candidates, wanted = named, "normalized"
    else:
        candidates = [path for path in every if not _is_skipped(path)]
        wanted = format

    records: list[LoadedRecord] = []
    ignored: list[str] = [str(path) for path in every if path not in candidates]
    formats_seen: set[str] = set()
    for path in candidates:
        for number, record in _read_records(path):
            detected = detect_format(record)
            label = str(path) if number is None else f"{path}:{number}"
            if detected is None or (wanted != "auto" and detected != wanted):
                ignored.append(label)
                continue
            try:
                result = CONVERTERS[detected](record, options)
            except BenchmarkResultError as exc:
                raise BenchmarkResultError(f"{label}: {exc}") from exc
            formats_seen.add(detected)
            records.append(LoadedRecord(path, number, result))

    if len(formats_seen) > 1:
        raise BenchmarkResultError(
            f"'{directory}' mixes result formats ({', '.join(sorted(formats_seen))}); "
            "keep one format per directory"
        )
    if not records:
        message = f"No benchmark results found under '{directory}'."
        if format in ("auto", "normalized"):
            message += (
                f" Normalized results must be named {', '.join(NORMALIZED_PATTERNS)};"
                " results from vLLM, AIPerf, GuideLLM or SGLang are detected in any"
                " .json or .jsonl file."
            )
        else:
            message += f" No .json or .jsonl file there is in the {format} format."
        if ignored:
            shown = ", ".join(Path(item).name for item in ignored[:3])
            more = f" and {len(ignored) - 3} more" if len(ignored) > 3 else ""
            message += f" Ignored {len(ignored)} other JSON file(s): {shown}{more}."
        raise BenchmarkResultError(message)
    return LoadedDirectory(format=formats_seen.pop(), records=records, ignored=ignored)
