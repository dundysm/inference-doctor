from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from aggregate import aggregate_raw_runs, render_summary
from benchmark_client import run_repetition
from harness_common import load_json, write_json
from harness_common import (
    collect_runtime_metadata,
    parse_gpu_metadata,
    require_matching_gpu_uuid,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
CONTAINER_NAME = "inference-doctor-vllm-48035"
CACHE_VOLUME = "inference-doctor-vllm-48035-hf-cache"


def run_command(
    command: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def verify_gpu(expected_gpu: str) -> dict[str, str]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_gpu_metadata(result.stdout, expected_gpu)


def ensure_container_name_is_free() -> None:
    result = run_command(
        ["docker", "inspect", CONTAINER_NAME],
        check=False,
    )
    if result.returncode == 0:
        raise RuntimeError(
            f"Docker container name '{CONTAINER_NAME}' is already in use"
        )


def wait_for_health(timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        running = run_command(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                CONTAINER_NAME,
            ],
            check=False,
        )
        if running.returncode != 0 or running.stdout.strip() != "true":
            raise RuntimeError("vLLM container stopped before becoming healthy")
        try:
            response = httpx.get(
                "http://127.0.0.1:8000/health",
                timeout=2.0,
            )
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(
        f"vLLM did not become healthy within {timeout_seconds} seconds"
    )


def build_docker_command(
    image: str,
    model: str,
    serve_args: list[str],
) -> list[str]:
    return [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        CONTAINER_NAME,
        "--gpus",
        "device=0",
        "--ipc=host",
        "--publish",
        "127.0.0.1:8000:8000",
        "--volume",
        f"{CACHE_VOLUME}:/root/.cache/huggingface",
        image,
        "--model",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "1",
        *serve_args,
    ]


def start_server(
    *,
    image: str,
    model: str,
    serve_args: list[str],
    output_dir: Path,
    health_timeout_seconds: int,
) -> tuple[subprocess.Popen[str], Any, list[str], dict[str, Any]]:
    ensure_container_name_is_free()
    run_command(["docker", "pull", image], capture_output=False)
    image_id = run_command(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    ).stdout.strip()
    repo_digests_output = run_command(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{json .RepoDigests}}",
            image,
        ]
    ).stdout.strip()
    image_metadata = {
        "id": image_id,
        "repo_digests": json.loads(repo_digests_output),
    }
    launch_command = build_docker_command(image, model, serve_args)
    run_command(launch_command)

    log_handle = (output_dir / "server.log").open(
        "w",
        encoding="utf-8",
    )
    log_process = subprocess.Popen(
        ["docker", "logs", "--follow", CONTAINER_NAME],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_health(health_timeout_seconds)
    except Exception:
        stop_server(log_process, log_handle)
        raise
    return log_process, log_handle, launch_command, image_metadata


def stop_server(log_process: subprocess.Popen[str], log_handle: Any) -> None:
    run_command(
        ["docker", "stop", "--time", "30", CONTAINER_NAME],
        check=False,
    )
    try:
        log_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        log_process.terminate()
        log_process.wait(timeout=5)
    log_handle.close()


def run_variant(
    *,
    name: str,
    variant: dict[str, str],
    config: dict[str, Any],
    prompts: list[dict[str, Any]],
    prompt_path: Path,
    output_root: Path,
    health_timeout_seconds: int,
    gpu_metadata: dict[str, str],
) -> Path:
    variant_dir = output_root / name
    variant_dir.mkdir(parents=True, exist_ok=False)
    log_process, log_handle, launch_command, image_metadata = start_server(
        image=variant["image"],
        model=config["model"],
        serve_args=config["serve_args"],
        output_dir=variant_dir,
        health_timeout_seconds=health_timeout_seconds,
    )

    environment = {
        **gpu_metadata,
        "vllm_version": variant["vllm_version"],
        "model": config["model"],
        "image": variant["image"],
        "runtime": collect_runtime_metadata(),
    }
    write_json(
        variant_dir / "server.json",
        {
            "image": variant["image"],
            "resolved_image": image_metadata,
            "launch_command": launch_command,
            "healthy_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    try:
        for repetition in range(1, config["repetitions"] + 1):
            run_repetition(
                base_url="http://127.0.0.1:8000",
                variant=name,
                repetition=repetition,
                environment=environment,
                prompt_path=prompt_path,
                prompts=prompts,
                seeds=config["seeds"],
                request_config=config["request"],
                serve_args=config["serve_args"],
                output_dir=variant_dir,
            )
    finally:
        stop_server(log_process, log_handle)

    raw_paths = sorted(variant_dir.glob("repetition-*/raw.json"))
    raw_runs = [load_json(path) for path in raw_paths]
    aggregate = aggregate_raw_runs(
        raw_runs,
        source_files=[str(path.relative_to(output_root)) for path in raw_paths],
    )
    aggregate_path = variant_dir / "aggregate.normalized.json"
    write_json(aggregate_path, aggregate)
    print(f"\n{name} aggregate\n{render_summary(aggregate)}\n")
    return aggregate_path


def compare_files(
    *,
    name: str,
    baseline: Path,
    candidate: Path,
    comparison_config: dict[str, Any],
    output_dir: Path,
    same_version: bool = False,
) -> str:
    common = [
        sys.executable,
        "-m",
        "inference_doctor.cli",
        "compare",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--latency-warn-percent",
        str(comparison_config["latency_warn_percent"]),
        "--latency-fail-percent",
        str(comparison_config["latency_fail_percent"]),
        "--throughput-warn-percent",
        str(comparison_config["throughput_warn_percent"]),
        "--throughput-fail-percent",
        str(comparison_config["throughput_fail_percent"]),
        "--max-failed-request-increase",
        str(comparison_config["max_failed_request_increase"]),
        "--stability-tolerance-percent",
        str(comparison_config.get("stability_tolerance_percent", 5.0)),
    ]
    if same_version:
        common.append("--same-version")
    json_result = run_command([*common, "--json"], check=False)
    if json_result.returncode not in (0, 1):
        raise RuntimeError(
            f"Comparison '{name}' failed: {json_result.stderr.strip()}"
        )
    payload = json.loads(json_result.stdout)
    write_json(output_dir / f"{name}.json", payload)

    terminal_result = run_command(common, check=False)
    (output_dir / f"{name}.txt").write_text(
        terminal_result.stdout,
        encoding="utf-8",
    )
    print(f"\n{name}\n{terminal_result.stdout}")
    return payload["overall_status"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/<UTC timestamp> under this experiment.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Overrides the manifest default of 3; must be at least 2.",
    )
    parser.add_argument(
        "--health-timeout-seconds",
        type=int,
        default=1200,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(EXPERIMENT_DIR / "experiment.json")
    prompts = load_json(EXPERIMENT_DIR / "prompts.json")
    if args.repetitions is not None:
        config["repetitions"] = args.repetitions
    if config["repetitions"] < 2:
        raise SystemExit("At least two repetitions are required for controls")

    output_root = args.output_dir or (
        EXPERIMENT_DIR
        / "results"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    baseline_gpu = verify_gpu(config["gpu"])
    run_metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "experiment": config,
        "baseline_host": baseline_gpu,
    }
    write_json(output_root / "run.json", run_metadata)

    baseline_aggregate = run_variant(
        name="baseline",
        variant=config["baseline"],
        config=config,
        prompts=prompts,
        prompt_path=EXPERIMENT_DIR / "prompts.json",
        output_root=output_root,
        health_timeout_seconds=args.health_timeout_seconds,
        gpu_metadata=baseline_gpu,
    )
    candidate_gpu = verify_gpu(config["gpu"])
    run_metadata["candidate_host"] = candidate_gpu
    write_json(output_root / "run.json", run_metadata)
    candidate_aggregate = run_variant(
        name="candidate",
        variant=config["candidate"],
        config=config,
        prompts=prompts,
        prompt_path=EXPERIMENT_DIR / "prompts.json",
        output_root=output_root,
        health_timeout_seconds=args.health_timeout_seconds,
        gpu_metadata=candidate_gpu,
    )

    comparisons_dir = output_root / "comparisons"
    comparisons_dir.mkdir()
    baseline_control = compare_files(
        name="baseline-vs-baseline",
        baseline=output_root / "baseline/repetition-001/normalized.json",
        candidate=output_root / "baseline/repetition-002/normalized.json",
        comparison_config=config["comparison"],
        output_dir=comparisons_dir,
        same_version=True,
    )
    candidate_control = compare_files(
        name="candidate-vs-candidate",
        baseline=output_root / "candidate/repetition-001/normalized.json",
        candidate=output_root / "candidate/repetition-002/normalized.json",
        comparison_config=config["comparison"],
        output_dir=comparisons_dir,
        same_version=True,
    )
    matching_gpu_uuid = require_matching_gpu_uuid(
        load_json(baseline_aggregate),
        load_json(candidate_aggregate),
    )
    cross_version = compare_files(
        name="baseline-vs-candidate",
        baseline=baseline_aggregate,
        candidate=candidate_aggregate,
        comparison_config=config["comparison"],
        output_dir=comparisons_dir,
    )

    summary = {
        "baseline_control": baseline_control,
        "candidate_control": candidate_control,
        "baseline_vs_candidate": cross_version,
        "gpu_uuid": matching_gpu_uuid,
        "cross_version_valid": (
            baseline_control == "PASS"
            and candidate_control == "PASS"
        ),
        "historical_regression_reproduced": (
            baseline_control == "PASS"
            and candidate_control == "PASS"
            and cross_version == "FAIL"
        ),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))

    if baseline_control != "PASS" or candidate_control != "PASS":
        print("A same-version control failed; cross-version result is invalid.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
