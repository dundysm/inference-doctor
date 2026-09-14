from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from common import collect_gpu_metadata, load_documents, load_json, make_batches, write_json

try:
    from inference_doctor.telemetry import TelemetrySampler
except ImportError:  # pragma: no cover - only relevant to an uninstalled harness
    TelemetrySampler = None  # type: ignore[assignment,misc]

LEGACY_DIR = Path(__file__).resolve().parents[1] / "vllm-52630"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

from benchmark_pooling import send_batch  # noqa: E402


RequestSender = Callable[[int, list[dict[str, str]], int, str], Awaitable[dict[str, Any]]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cycle_batch_indices(batch_count: int, request_count: int) -> list[int]:
    if batch_count <= 0 or request_count < 0:
        raise ValueError("batch_count must be positive and request_count non-negative")
    return [index % batch_count for index in range(request_count)]


async def run_window(
    *,
    client: httpx.AsyncClient,
    server_url: str,
    model: str,
    batches: list[list[dict[str, str]]],
    headers: dict[str, str],
    duration_seconds: float,
    concurrency: int,
    phase: str,
    request_sender: RequestSender | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    if request_sender is None:
        async def request_sender(
            batch_index: int,
            batch: list[dict[str, str]],
            _cycle_index: int,
            _phase: str,
        ) -> dict[str, Any]:
            return await send_batch(
                client,
                server_url=server_url,
                model=model,
                batch_index=batch_index,
                batch=batch,
                headers=headers,
            )

    started_at = utc_now()
    started_clock = clock()
    deadline = started_clock + duration_seconds
    pending: set[asyncio.Task[dict[str, Any]]] = set()
    records: list[dict[str, Any]] = []
    next_request = 0

    async def invoke(batch_number: int) -> dict[str, Any]:
        batch_index = batch_number % len(batches)
        cycle_index = batch_number // len(batches)
        record = await request_sender(
            batch_index,
            batches[batch_index],
            cycle_index,
            phase,
        )
        record["phase"] = phase
        record["cycle_index"] = cycle_index
        record["batch_index"] = batch_index
        return record

    while pending or clock() < deadline:
        while len(pending) < concurrency and clock() < deadline:
            pending.add(asyncio.create_task(invoke(next_request)))
            next_request += 1
        if not pending:
            await asyncio.sleep(0)
            continue
        done, pending = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED,
        )
        records.extend(task.result() for task in done)

    ended_at = utc_now()
    return {
        "phase": phase,
        "configured_duration_seconds": duration_seconds,
        "elapsed_seconds": max(clock() - started_clock, 0.0),
        "started_at": started_at,
        "ended_at": ended_at,
        "requests_started": next_request,
        "requests": records,
    }


def summarize_window(window: dict[str, Any]) -> dict[str, Any]:
    records = window["requests"]
    successful = [record for record in records if record.get("success")]
    failed = [record for record in records if not record.get("success")]
    successful_texts = sum(int(record.get("texts_returned") or 0) for record in successful)
    duration = float(window["configured_duration_seconds"])
    return {
        "phase": window["phase"],
        "configured_duration_seconds": duration,
        "elapsed_seconds": window["elapsed_seconds"],
        "started_at": window["started_at"],
        "ended_at": window["ended_at"],
        "requests_started": window["requests_started"],
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "successful_texts": successful_texts,
        "texts_per_second": successful_texts / duration,
        "request_throughput": len(successful) / duration,
        "median_request_latency_ms": _percentile(
            [float(record["latency_ms"]) for record in records], 0.5
        ),
        "p95_request_latency_ms": _percentile(
            [float(record["latency_ms"]) for record in records], 0.95
        ),
        "server_errors": [record.get("error") for record in failed if record.get("error")],
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def package_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("vllm", "torch", "datasets", "httpx"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    try:
        import torch

        result["torch_cuda"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except (ImportError, RuntimeError):
        result["torch_cuda"] = None
        result["cuda_available"] = None
        result["cuda_device"] = None
    return result


async def run_block(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    manifest_path = args.manifest
    manifest = load_json(manifest_path)
    documents, dataset_metadata = load_documents(manifest)
    batches = make_batches(documents, config["dataset"]["batch_size"])
    if len(documents) != 4096 or len(batches) != 128:
        raise SystemExit("Frozen workload must contain 4096 documents in 128 batches")

    gpu = collect_gpu_metadata()
    headers = {}
    timeout = httpx.Timeout(args.timeout_seconds)
    sampler = None
    telemetry_output = getattr(args, "telemetry_output", None)
    if telemetry_output:
        if TelemetrySampler is None:
            raise SystemExit("Telemetry requested but inference-doctor is not importable")
        sampler = TelemetrySampler(getattr(args, "telemetry_interval_seconds", 5.0))
        sampler.start()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            health = await client.get(f"{args.server_url.rstrip('/')}/health")
            health.raise_for_status()

            warmup = await run_window(
                client=client,
                server_url=args.server_url,
                model=config["server"]["model_name"],
                batches=batches,
                headers=headers,
                duration_seconds=args.warmup_seconds,
                concurrency=config["dataset"]["concurrency"],
                phase="warmup",
            )
            measurement = await run_window(
                client=client,
                server_url=args.server_url,
                model=config["server"]["model_name"],
                batches=batches,
                headers=headers,
                duration_seconds=args.measurement_seconds,
                concurrency=config["dataset"]["concurrency"],
                phase="measurement",
            )
    finally:
        if sampler is not None:
            sampler.stop()
            sampler.write(telemetry_output)

    return {
        "schema_version": "inference-ci-reliability-001-raw-1",
        "complete": True,
        "block": {
            "order": args.block_order,
            "variant": args.variant,
            "version_label": args.version_label,
            "gpu_uuid_before": args.gpu_uuid_before,
            "gpu_uuid_at_client_start": gpu["gpu_uuid"],
            "concurrency": config["dataset"]["concurrency"],
            "warmup_seconds": args.warmup_seconds,
            "measurement_seconds": args.measurement_seconds,
        },
        "environment": {
            "gpu": gpu,
            "packages": package_metadata(),
        },
        "benchmark_parameters": {
            "model": config["model"],
            "dataset": {**config["dataset"], **dataset_metadata},
            "batch_count": len(batches),
            "request_shape": "32 documents per /v1/embeddings request",
            "server": config["server"],
        },
        "warmup": {
            **summarize_window(warmup),
            "requests": warmup["requests"],
        },
        "measurement": {
            **summarize_window(measurement),
            "requests": measurement["requests"],
        },
    }


def parse_args() -> argparse.Namespace:
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:8043")
    parser.add_argument("--version-label", required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--block-order", type=int, required=True)
    parser.add_argument("--gpu-uuid-before", required=True)
    parser.add_argument("--warmup-seconds", type=float, default=300.0)
    parser.add_argument("--measurement-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=experiment_dir / "experiment.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--telemetry-output", type=Path)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest is None:
        config = load_json(args.config)
        args.manifest = (args.config.parent / config["dataset"]["dataset_manifest"]).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw = asyncio.run(run_block(args))
    write_json(args.output_dir / "raw.json", raw)
    write_json(args.output_dir / "measurement.json", raw["measurement"])
    print(json.dumps(raw["measurement"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
