from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable


class GpuIdentityError(ValueError):
    pass


def parse_gpu_metadata(output: str, expected_gpu: str) -> dict[str, str]:
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {len(rows)}"
        )

    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 4:
        raise RuntimeError("Unexpected nvidia-smi metadata output")

    name, gpu_uuid, driver_version, memory_mib = parts
    if "RTX 4090" not in name or "RTX 4090" not in expected_gpu:
        raise RuntimeError(f"Expected an RTX 4090, found {name}")
    if not gpu_uuid:
        raise RuntimeError("nvidia-smi did not report a GPU UUID")

    return {
        "gpu": name,
        "gpu_uuid": gpu_uuid,
        "driver_version": driver_version,
        "gpu_memory_mib": memory_mib,
    }


def collect_gpu_metadata(expected_gpu: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_metadata(result.stdout, expected_gpu)


def collect_runtime_metadata() -> dict[str, Any]:
    runpod_environment = {
        key: value
        for key in (
            "RUNPOD_POD_ID",
            "RUNPOD_DC_ID",
            "RUNPOD_GPU_COUNT",
            "CUDA_VISIBLE_DEVICES",
        )
        if (value := os.environ.get(key)) is not None
    }
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "runpod_environment": runpod_environment,
    }


def require_matching_gpu_uuid(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    uuids: dict[str, str] = {}
    for label, payload in (
        ("baseline", baseline),
        ("candidate", candidate),
    ):
        environment = payload.get("environment")
        if not isinstance(environment, dict):
            raise GpuIdentityError(
                f"{label} environment metadata is missing"
            )
        gpu_uuid = environment.get("gpu_uuid")
        if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
            raise GpuIdentityError(
                f"{label} environment.gpu_uuid must be a non-empty string"
            )
        uuids[label] = gpu_uuid.strip()

    if uuids["baseline"] != uuids["candidate"]:
        raise GpuIdentityError(
            "GPU UUID mismatch: baseline "
            f"{uuids['baseline']!r}, candidate {uuids['candidate']!r}"
        )
    return uuids["baseline"]


def finite_numbers(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def percentile(values: Iterable[Any], quantile: float) -> float | None:
    numbers = sorted(finite_numbers(values))
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]

    position = (len(numbers) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return numbers[lower]
    fraction = position - lower
    return numbers[lower] + (
        (numbers[upper] - numbers[lower]) * fraction
    )


def summarize(values: Iterable[Any]) -> dict[str, float | None]:
    numbers = finite_numbers(values)
    if not numbers:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p95": percentile(numbers, 0.95),
    }


def prompts_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, allow_nan=False)
        destination.write("\n")


def request_metrics(
    *,
    ttft_ms: float | None,
    total_latency_ms: float,
    output_tokens: int | None,
) -> dict[str, float | None]:
    decode_duration_ms = None
    tpot_ms = None
    decode_tokens_per_second = None

    if ttft_ms is not None:
        decode_duration_ms = max(total_latency_ms - ttft_ms, 0.0)

    if (
        decode_duration_ms is not None
        and decode_duration_ms > 0
        and output_tokens is not None
        and output_tokens > 1
    ):
        decoded_tokens = output_tokens - 1
        tpot_ms = decode_duration_ms / decoded_tokens
        decode_tokens_per_second = decoded_tokens / (
            decode_duration_ms / 1000
        )

    return {
        "decode_duration_ms": decode_duration_ms,
        "tpot_ms": tpot_ms,
        "decode_tokens_per_second": decode_tokens_per_second,
    }


def normalized_result(
    *,
    environment: dict[str, Any],
    benchmark_parameters: dict[str, Any],
    records: list[dict[str, Any]],
    measured_duration_seconds: float,
) -> dict[str, Any]:
    successful = [record for record in records if record.get("success")]
    failed_count = len(records) - len(successful)
    decode_rates = finite_numbers(
        record.get("decode_tokens_per_second") for record in successful
    )

    return {
        "schema_version": "1",
        "environment": environment,
        "benchmark_parameters": benchmark_parameters,
        "metrics": {
            "ttft_ms": summarize(
                record.get("ttft_ms") for record in successful
            ),
            "tpot_ms": summarize(
                record.get("tpot_ms") for record in successful
            ),
            "output_token_throughput": (
                statistics.fmean(decode_rates) if decode_rates else None
            ),
            "request_throughput": (
                len(successful) / measured_duration_seconds
                if measured_duration_seconds > 0
                else None
            ),
            "requests": {
                "successful": len(successful),
                "failed": failed_count,
            },
        },
    }
