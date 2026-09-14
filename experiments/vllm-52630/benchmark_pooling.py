from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from harness_common import (
    ExperimentError,
    collect_gpu_metadata,
    collect_runtime_metadata,
    load_documents,
    load_json,
    make_batches,
    normalized_repetition,
    summarize,
    write_json,
)


def package_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("vllm", "torch", "transformers", "datasets", "httpx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        packages["torch_cuda"] = torch.version.cuda
        packages["cuda_available"] = str(torch.cuda.is_available())
        packages["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except (ImportError, RuntimeError):
        packages["torch_cuda"] = None
        packages["cuda_available"] = None
        packages["cuda_device"] = None
    return packages


def _payload_bytes(texts: list[str], model: str) -> int:
    payload = json.dumps(
        {"model": model, "input": texts},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(payload)


async def send_batch(
    client: httpx.AsyncClient,
    *,
    server_url: str,
    model: str,
    batch_index: int,
    batch: list[dict[str, str]],
    headers: dict[str, str],
    record_payload: bool = True,
) -> dict[str, Any]:
    texts = [document["text"] for document in batch]
    started = time.perf_counter()
    record: dict[str, Any] = {
        "batch_index": batch_index,
        "document_ids": [document["id"] for document in batch],
        "num_texts": len(texts),
        "text_characters": sum(len(text) for text in texts),
        "payload_bytes": _payload_bytes(texts, model),
        "request_sha256": hashlib.sha256(
            "\n".join(texts).encode("utf-8")
        ).hexdigest(),
        "success": False,
        "status_code": None,
        "latency_ms": None,
        "texts_returned": 0,
        "error": None,
    }
    try:
        response = await client.post(
            f"{server_url.rstrip('/')}/v1/embeddings",
            json={"model": model, "input": texts},
            headers=headers,
        )
        record["status_code"] = response.status_code
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        record["texts_returned"] = len(data)
        record["success"] = len(data) == len(texts)
        if not record["success"]:
            record["error"] = (
                f"Expected {len(texts)} embeddings, received {len(data)}"
            )
    except (httpx.HTTPError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["latency_ms"] = (time.perf_counter() - started) * 1000
    if not record_payload:
        record.pop("document_ids", None)
    return record


async def run_warmup(
    client: httpx.AsyncClient,
    *,
    server_url: str,
    model: str,
    batches: list[list[dict[str, str]]],
    headers: dict[str, str],
) -> dict[str, Any]:
    records = []
    for index, batch in enumerate(batches):
        records.append(
            await send_batch(
                client,
                server_url=server_url,
                model=model,
                batch_index=index,
                batch=batch,
                headers=headers,
                record_payload=False,
            )
        )
    return {
        "batches": len(records),
        "successful_requests": sum(record["success"] for record in records),
        "failed_requests": sum(not record["success"] for record in records),
        "records": records,
    }


async def run_concurrency_point(
    client: httpx.AsyncClient,
    *,
    server_url: str,
    model: str,
    batches: list[list[dict[str, str]]],
    concurrency: int,
    headers: dict[str, str],
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(index: int, batch: list[dict[str, str]]) -> dict[str, Any]:
        async with semaphore:
            return await send_batch(
                client,
                server_url=server_url,
                model=model,
                batch_index=index,
                batch=batch,
                headers=headers,
            )

    started = time.perf_counter()
    records = await asyncio.gather(
        *(limited(index, batch) for index, batch in enumerate(batches))
    )
    duration = time.perf_counter() - started
    successful = [record for record in records if record["success"]]
    failed = [record for record in records if not record["success"]]
    texts_processed = sum(record["texts_returned"] for record in successful)
    errors = [record["error"] for record in failed if record.get("error")]
    return {
        "concurrency": concurrency,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_texts_processed": texts_processed,
        "wall_clock_duration_seconds": duration,
        "request_throughput": len(successful) / duration if duration else None,
        "texts_per_second": texts_processed / duration if duration else None,
        "latency_ms": summarize(record["latency_ms"] for record in records),
        "server_errors": errors,
        "requests": records,
    }


async def run_benchmark(
    *,
    server_url: str,
    model: str,
    batches: list[list[dict[str, str]]],
    warmup_batches: int,
    concurrency_levels: list[int],
    headers: dict[str, str],
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        health = await client.get(f"{server_url.rstrip('/')}/health")
        health.raise_for_status()
        warmup_batches_payload = batches[:warmup_batches]
        results: dict[str, dict[str, Any]] = {}
        for concurrency in concurrency_levels:
            warmup = await run_warmup(
                client,
                server_url=server_url,
                model=model,
                batches=warmup_batches_payload,
                headers=headers,
            )
            point = await run_concurrency_point(
                client,
                server_url=server_url,
                model=model,
                batches=batches,
                concurrency=concurrency,
                headers=headers,
            )
            point["warmup"] = warmup
            results[str(concurrency)] = point
    return results, {"health_status_code": health.status_code}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:8043")
    parser.add_argument("--version-label", required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiment.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("dataset_manifest.json"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    variant = args.variant
    if variant is None:
        variant = next(
            name
            for name in ("baseline", "candidate")
            if config[name]["label"] == args.version_label
        )
    if config[variant]["label"] != args.version_label:
        raise SystemExit("Version label does not match the selected variant")
    if args.repetition < 1:
        raise SystemExit("Repetition must be positive")

    documents, dataset_metadata = load_documents(manifest)
    dataset_config = config["dataset"]
    if dataset_config["batch_size"] != 32:
        raise SystemExit("This experiment requires exactly 32 documents per request")
    batches = make_batches(documents, dataset_config["batch_size"])
    api_key = os.environ.get(args.api_key_env)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    environment = {
        "variant": variant,
        "version_label": args.version_label,
        "vllm_version": config[variant]["vllm_version"],
        "torch_version_expected": config[variant]["torch_version"],
        "model": config["model"],
        "server_url": args.server_url,
        "packages": package_metadata(),
        "gpu": collect_gpu_metadata(),
        "runtime": collect_runtime_metadata(),
    }
    parameters = {
        "experiment": config["experiment"],
        "repetition": args.repetition,
        "model": config["model"],
        "server": config["server"],
        "dataset": {**dataset_config, **dataset_metadata},
        "batch_count": len(batches),
        "request_shape": "32 documents per /v1/embeddings request",
        "concurrency": dataset_config["concurrency"],
        "warmup_batches": dataset_config["warmup_batches"],
    }
    try:
        concurrency_results, client_metadata = asyncio.run(
            run_benchmark(
                server_url=args.server_url,
                model=config["server"]["model_name"],
                batches=batches,
                warmup_batches=dataset_config["warmup_batches"],
                concurrency_levels=dataset_config["concurrency"],
                headers=headers,
                timeout_seconds=args.timeout_seconds,
            )
        )
    except (httpx.HTTPError, ExperimentError) as exc:
        raise SystemExit(f"Benchmark failed: {exc}") from exc

    raw = {
        "schema_version": "vllm-52630-raw-1",
        "complete": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "benchmark_parameters": parameters,
        "client": client_metadata,
        "concurrency_results": concurrency_results,
    }
    normalized = normalized_repetition(
        environment=environment,
        benchmark_parameters=parameters,
        concurrency_results=concurrency_results,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "raw.json", raw)
    write_json(args.output_dir / "normalized.json", normalized)
    print(json.dumps(normalized, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
