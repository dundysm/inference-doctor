from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from harness_common import (
    collect_gpu_metadata,
    collect_runtime_metadata,
    load_json,
    normalized_result,
    prompts_sha256,
    request_metrics,
    write_json,
)


def resolve_variant_metadata(
    config: dict[str, Any],
    version_label: str,
    variant: str | None,
    image: str | None,
) -> tuple[str, str]:
    variants = {
        name: config[name]
        for name in ("baseline", "candidate")
    }
    if variant is None:
        matches = [
            name
            for name, metadata in variants.items()
            if metadata["vllm_version"] == version_label
        ]
        if len(matches) != 1:
            raise ValueError(
                "--variant is required when --version-label does not "
                "uniquely match the experiment manifest"
            )
        variant = matches[0]

    if variant not in variants:
        raise ValueError("--variant must be 'baseline' or 'candidate'")
    expected_version = variants[variant]["vllm_version"]
    if version_label != expected_version:
        raise ValueError(
            f"{variant} requires version label {expected_version!r}, "
            f"got {version_label!r}"
        )
    return variant, image or variants[variant]["image"]


def verify_server(server_url: str) -> None:
    response = httpx.get(
        f"{server_url.rstrip('/')}/health",
        timeout=10.0,
    )
    response.raise_for_status()


def run_streaming_request(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    prompt: dict[str, Any],
    seed: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at: float | None = None
    output_parts: list[str] = []
    usage: dict[str, Any] | None = None
    status_code: int | None = None

    try:
        with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": prompt["messages"],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            status_code = response.status_code
            response.raise_for_status()

            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue

                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]

                for choice in chunk.get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output_parts.append(content)

        finished = time.perf_counter()
        total_latency_ms = (finished - started) * 1000
        ttft_ms = (
            (first_token_at - started) * 1000
            if first_token_at is not None
            else None
        )
        output_tokens_raw = (
            usage.get("completion_tokens") if usage is not None else None
        )
        output_tokens = (
            int(output_tokens_raw)
            if isinstance(output_tokens_raw, (int, float))
            else None
        )
        derived = request_metrics(
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
        )

        return {
            "success": True,
            "status_code": status_code,
            "error": None,
            "ttft_ms": ttft_ms,
            "total_latency_ms": total_latency_ms,
            "output_tokens": output_tokens,
            **derived,
            "usage": usage,
            "output_text": "".join(output_parts),
        }
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        finished = time.perf_counter()
        return {
            "success": False,
            "status_code": status_code,
            "error": f"{type(exc).__name__}: {exc}",
            "ttft_ms": None,
            "total_latency_ms": (finished - started) * 1000,
            "output_tokens": None,
            "decode_duration_ms": None,
            "tpot_ms": None,
            "decode_tokens_per_second": None,
            "usage": usage,
            "output_text": "".join(output_parts),
        }


def build_benchmark_parameters(
    *,
    prompt_path: Path,
    prompt_count: int,
    seeds: list[int],
    request_config: dict[str, Any],
    serve_args: list[str],
    repetitions: int,
) -> dict[str, Any]:
    return {
        "endpoint": "/v1/chat/completions",
        "prompts_sha256": prompts_sha256(prompt_path),
        "prompt_count": prompt_count,
        "seeds": seeds,
        "max_tokens": request_config["max_tokens"],
        "temperature": request_config["temperature"],
        "top_p": request_config["top_p"],
        "concurrency": request_config["concurrency"],
        "stream": request_config["stream"],
        "warmup_requests_per_repetition": request_config[
            "warmup_requests_per_repetition"
        ],
        "repetitions": repetitions,
        "serve_args": serve_args,
    }


def run_repetition(
    *,
    base_url: str,
    variant: str,
    repetition: int,
    environment: dict[str, Any],
    prompt_path: Path,
    prompts: list[dict[str, Any]],
    seeds: list[int],
    request_config: dict[str, Any],
    serve_args: list[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    if request_config.get("concurrency") != 1:
        raise ValueError("This experiment requires concurrency=1")
    if len(prompts) != 10:
        raise ValueError("This experiment requires exactly 10 prompts")
    if len(seeds) != 3:
        raise ValueError("This experiment requires exactly 3 seeds")

    repetition_dir = output_dir / f"repetition-{repetition:03d}"
    repetition_dir.mkdir(parents=True, exist_ok=False)
    benchmark_parameters = build_benchmark_parameters(
        prompt_path=prompt_path,
        prompt_count=len(prompts),
        seeds=seeds,
        request_config=request_config,
        serve_args=serve_args,
        repetitions=1,
    )
    raw_path = repetition_dir / "raw.json"
    normalized_path = repetition_dir / "normalized.json"
    raw_payload = {
        "schema_version": "1",
        "variant": variant,
        "repetition": repetition,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "environment": environment,
        "benchmark_parameters": benchmark_parameters,
        "warmup": None,
        "measured_duration_seconds": 0.0,
        "requests": [],
    }
    write_json(raw_path, raw_payload)
    client = httpx.Client(timeout=httpx.Timeout(1200.0))

    try:
        warmup = run_streaming_request(
            client,
            base_url=base_url,
            model=environment["model"],
            prompt=prompts[0],
            seed=seeds[0],
            max_tokens=request_config["max_tokens"],
            temperature=request_config["temperature"],
            top_p=request_config["top_p"],
        )
        warmup["excluded_from_measurements"] = True
        raw_payload["warmup"] = warmup
        write_json(raw_path, raw_payload)
        if not warmup["success"]:
            raise RuntimeError(f"Warmup request failed: {warmup['error']}")

        records: list[dict[str, Any]] = []
        measured_started = time.perf_counter()
        sequence = 0
        for prompt in prompts:
            for seed in seeds:
                sequence += 1
                record = run_streaming_request(
                    client,
                    base_url=base_url,
                    model=environment["model"],
                    prompt=prompt,
                    seed=seed,
                    max_tokens=request_config["max_tokens"],
                    temperature=request_config["temperature"],
                    top_p=request_config["top_p"],
                )
                record.update(
                    {
                        "sequence": sequence,
                        "prompt_id": prompt["id"],
                        "seed": seed,
                    }
                )
                records.append(record)
                raw_payload["requests"] = records
                raw_payload["measured_duration_seconds"] = (
                    time.perf_counter() - measured_started
                )
                write_json(raw_path, raw_payload)
                print(
                    f"{variant} repetition {repetition}: "
                    f"{sequence:02d}/30 {prompt['id']} seed={seed} "
                    f"success={record['success']}"
                )
        measured_duration_seconds = time.perf_counter() - measured_started
    finally:
        client.close()

    raw_payload["complete"] = True
    raw_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    raw_payload["measured_duration_seconds"] = measured_duration_seconds
    raw_payload["requests"] = records
    write_json(raw_path, raw_payload)
    normalized = normalized_result(
        environment=environment,
        benchmark_parameters=benchmark_parameters,
        records=records,
        measured_duration_seconds=measured_duration_seconds,
    )

    write_json(normalized_path, normalized)
    return raw_path, normalized_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        "--base-url",
        dest="server_url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--version-label",
        "--vllm-version",
        dest="version_label",
        required=True,
    )
    parser.add_argument("--variant")
    parser.add_argument("--image")
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiment.json"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).with_name("prompts.json"),
    )
    args = parser.parse_args(argv)
    if args.repetition is not None and args.repetitions is not None:
        parser.error("--repetition and --repetitions are mutually exclusive")
    if args.repetition is not None and args.repetition < 1:
        parser.error("--repetition must be at least 1")
    if args.repetitions is not None and args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    prompts = load_json(args.prompts)
    variant, image = resolve_variant_metadata(
        config,
        args.version_label,
        args.variant,
        args.image,
    )
    verify_server(args.server_url)
    gpu_metadata = collect_gpu_metadata(config["gpu"])
    environment = {
        **gpu_metadata,
        "vllm_version": args.version_label,
        "model": config["model"],
        "image": image,
        "runtime": collect_runtime_metadata(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "environment.json", environment)

    if args.repetition is not None:
        repetitions = [args.repetition]
    else:
        repetition_count = args.repetitions or config["repetitions"]
        repetitions = list(range(1, repetition_count + 1))

    for repetition in repetitions:
        run_repetition(
            base_url=args.server_url,
            variant=variant,
            repetition=repetition,
            environment=environment,
            prompt_path=args.prompts,
            prompts=prompts,
            seeds=config["seeds"],
            request_config=config["request"],
            serve_args=config["serve_args"],
            output_dir=args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
