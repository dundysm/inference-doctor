from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable


class ExperimentError(ValueError):
    pass


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, allow_nan=False)
        destination.write("\n")


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
    return numbers[lower] + (numbers[upper] - numbers[lower]) * fraction


def summarize(values: Iterable[Any]) -> dict[str, float | None]:
    numbers = finite_numbers(values)
    return {
        "mean": statistics.fmean(numbers) if numbers else None,
        "median": statistics.median(numbers) if numbers else None,
        "p95": percentile(numbers, 0.95),
    }


def sample_standard_deviation(values: Iterable[Any]) -> float | None:
    numbers = finite_numbers(values)
    return statistics.stdev(numbers) if len(numbers) >= 2 else None


def coefficient_of_variation(values: Iterable[Any]) -> float | None:
    numbers = finite_numbers(values)
    if len(numbers) < 2:
        return None
    mean = statistics.fmean(numbers)
    if mean == 0:
        return None
    return statistics.stdev(numbers) / mean


def selected_indices(manifest: dict[str, Any]) -> list[int]:
    selection = manifest["selection"]
    if selection["kind"] != "contiguous_row_range":
        raise ExperimentError("Unsupported dataset selection kind")
    start = int(selection["start"])
    count = int(selection["count"])
    if start < 0 or count <= 0:
        raise ExperimentError("Dataset selection must be a positive range")
    return list(range(start, start + count))


def selected_index_sha256(indices: list[int]) -> str:
    encoded = json.dumps(indices, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_value(row: Any, candidates: list[str]) -> Any:
    if not isinstance(row, dict):
        return None
    for key in candidates:
        if row.get(key) is not None:
            return row[key]
    return None


def load_documents(
    manifest: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    revision = manifest.get("revision")
    resolved_commit_sha = manifest.get("resolved_commit_sha")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision.lower())
        or resolved_commit_sha != revision
    ):
        raise ExperimentError(
            "Dataset manifest must use a matching immutable 40-character commit SHA"
        )
    indices = selected_indices(manifest)
    expected_hash = manifest.get("selected_index_sha256")
    actual_hash = selected_index_sha256(indices)
    if expected_hash and actual_hash != expected_hash:
        raise ExperimentError("Dataset index manifest hash does not match")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ExperimentError(
            "The benchmark environment must include the datasets package"
        ) from exc

    data_files = manifest.get("data_files")
    if data_files:
        # The frozen MTEB snapshot contains corpus, query, and qrels Parquet
        # files with different schemas. Load only the corpus file selected by
        # the experiment manifest.
        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            split=manifest.get("data_files_split", manifest["split"]),
        )
    else:
        dataset = load_dataset(
            manifest["dataset"],
            split=manifest["split"],
            revision=manifest["revision"],
        )
    if len(dataset) < max(indices) + 1:
        raise ExperimentError("Dataset has fewer rows than the manifest")
    dataset_fingerprint = getattr(dataset, "_fingerprint", None)
    expected_fingerprint = manifest.get("datasets_fingerprint")
    if expected_fingerprint and dataset_fingerprint != expected_fingerprint:
        raise ExperimentError("Dataset fingerprint does not match the manifest")

    documents: list[dict[str, str]] = []
    id_candidates = manifest["document_id_field_candidates"]
    text_candidates = manifest["text_field_candidates"]
    for index in indices:
        row = dataset[index]
        document_id = _first_value(row, id_candidates)
        text = _first_value(row, text_candidates)
        if document_id is None:
            document_id = str(index)
        if not isinstance(text, str) or not text:
            raise ExperimentError(f"Row {index} has no usable text field")
        documents.append({"id": str(document_id), "text": text})

    selected_document_ids_sha256 = hashlib.sha256(
        json.dumps(
            [document["id"] for document in documents],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_document_ids_hash = manifest.get("selected_document_ids_sha256")
    if expected_document_ids_hash and selected_document_ids_sha256 != expected_document_ids_hash:
        raise ExperimentError("Selected document ID hash does not match the manifest")

    return documents, {
        "dataset_revision": revision,
        "resolved_commit_sha": resolved_commit_sha,
        "datasets_fingerprint": dataset_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "selected_indices": {
            "start": indices[0],
            "count": len(indices),
            "index_sha256": actual_hash,
        },
        "selected_document_ids_sha256": selected_document_ids_sha256,
    }


def make_batches(
    documents: list[dict[str, str]],
    batch_size: int,
) -> list[list[dict[str, str]]]:
    if len(documents) % batch_size:
        raise ExperimentError(
            "Selected document count is not divisible by batch size"
        )
    return [
        documents[start : start + batch_size]
        for start in range(0, len(documents), batch_size)
    ]


def collect_gpu_metadata() -> dict[str, Any]:
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
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ExperimentError(f"Expected one visible GPU, found {len(rows)}")
    values = [part.strip() for part in rows[0].split(",")]
    if len(values) != 4:
        raise ExperimentError("Unexpected nvidia-smi output")
    name, gpu_uuid, driver, memory_mib = values
    if "H100" not in name:
        raise ExperimentError(f"Expected H100, found {name}")
    return {
        "gpu": name,
        "gpu_uuid": gpu_uuid,
        "driver_version": driver,
        "gpu_memory_mib": memory_mib,
        "nvidia_smi": result.stdout.strip(),
    }


def collect_runtime_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def require_matching_gpu_uuid(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    baseline_uuid = baseline.get("gpu_uuid")
    candidate_uuid = candidate.get("gpu_uuid")
    if not baseline_uuid or not candidate_uuid:
        raise ExperimentError("Both GPU UUIDs are required")
    if baseline_uuid != candidate_uuid:
        raise ExperimentError(
            f"GPU UUID mismatch: {baseline_uuid!r} versus {candidate_uuid!r}"
        )
    return baseline_uuid


def normalized_repetition(
    *,
    environment: dict[str, Any],
    benchmark_parameters: dict[str, Any],
    concurrency_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "vllm-52630-1",
        "environment": environment,
        "benchmark_parameters": benchmark_parameters,
        "metrics": {
            "texts_per_second": {
                level: result["texts_per_second"]
                for level, result in concurrency_results.items()
            },
            "request_latency_ms": {
                level: result["latency_ms"]["median"]
                for level, result in concurrency_results.items()
            },
            "requests": {
                "successful": sum(
                    result["successful_requests"]
                    for result in concurrency_results.values()
                ),
                "failed": sum(
                    result["failed_requests"]
                    for result in concurrency_results.values()
                ),
            },
        },
    }
