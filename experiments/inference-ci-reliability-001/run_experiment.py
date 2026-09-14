from __future__ import annotations

import argparse
import json
import os
import posixpath
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis import bootstrap_mean_difference, regression_decision, summarize_measurements
from common import (
    ExperimentError,
    collect_gpu_metadata,
    load_json,
    write_json,
)

LEGACY_DIR = Path(__file__).resolve().parents[1] / "vllm-52630"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

from run_same_gpu import (  # noqa: E402
    capture_gpu_cleanup,
    start_server,
    stop_server,
    variant_python_executable,
    verify_no_vllm_process,
    wait_for_health,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
SCHEDULE = (
    "baseline",
    "candidate",
    "candidate",
    "baseline",
    "baseline",
    "candidate",
    "candidate",
    "baseline",
    "baseline",
    "candidate",
    "candidate",
    "baseline",
)


def validate_schedule(schedule: list[str] | tuple[str, ...]) -> None:
    if tuple(schedule) != SCHEDULE:
        raise ExperimentError("Schedule must be exactly ABBA repeated three times")
    if schedule.count("baseline") != 6 or schedule.count("candidate") != 6:
        raise ExperimentError("ABBA schedule must contain six blocks per variant")


def validate_block_gpu_uuid(
    expected_uuid: str,
    before: dict[str, Any],
    at_client_start: str,
    after: dict[str, Any],
) -> None:
    observed = {
        before.get("gpu_uuid"),
        at_client_start,
        after.get("gpu_uuid"),
    }
    if observed != {expected_uuid}:
        raise ExperimentError(
            "GPU UUID changed during block: "
            f"expected {expected_uuid!r}, observed {sorted(observed)!r}"
        )


def run_block(
    *,
    order: int,
    variant_name: str,
    variant: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    expected_gpu_uuid: str | None,
    health_timeout: int,
    telemetry_dir: Path | None = None,
) -> tuple[dict[str, Any], str]:
    before = collect_gpu_metadata()
    if expected_gpu_uuid and before["gpu_uuid"] != expected_gpu_uuid:
        raise ExperimentError(
            f"GPU UUID changed before block {order}: "
            f"expected {expected_gpu_uuid}, got {before['gpu_uuid']}"
        )
    expected_gpu_uuid = expected_gpu_uuid or before["gpu_uuid"]

    block_dir = output_root / "blocks" / f"{order:02d}-{variant_name}"
    log_path = output_root / "server-logs" / f"{order:02d}-{variant_name}.log"
    server_url = f"http://127.0.0.1:{config['server']['port']}"
    process, command = start_server(
        variant=variant,
        config=config,
        log_path=log_path,
    )
    try:
        wait_for_health(server_url, process, health_timeout)
        benchmark_command = [
            variant_python_executable(variant),
            str(EXPERIMENT_DIR / "benchmark_block.py"),
            "--server-url",
            server_url,
            "--version-label",
            variant["label"],
            "--variant",
            variant_name,
            "--block-order",
            str(order),
            "--gpu-uuid-before",
            before["gpu_uuid"],
            "--warmup-seconds",
            str(config["warmup_seconds"]),
            "--measurement-seconds",
            str(config["measurement_seconds"]),
            "--output-dir",
            str(block_dir),
            "--config",
            str(EXPERIMENT_DIR / "experiment.json"),
        ]
        if telemetry_dir is not None:
            benchmark_command.extend(
                [
                    "--telemetry-output",
                    str(telemetry_dir / f"{order:02d}-{variant_name}.json"),
                ]
            )
        subprocess.run(benchmark_command, check=True)
    finally:
        stop_server(process)

    write_json(
        block_dir / "server.json",
        {
            "launch_command": command,
            "server_log": str(log_path),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "server_url": server_url,
            "gpu_uuid_before": before["gpu_uuid"],
        },
    )
    verify_no_vllm_process()
    capture_gpu_cleanup(block_dir / "after-stop-gpu-memory.txt")
    after = collect_gpu_metadata()
    raw_path = block_dir / "raw.json"
    raw = load_json(raw_path)
    validate_block_gpu_uuid(
        expected_gpu_uuid,
        before,
        raw["block"]["gpu_uuid_at_client_start"],
        after,
    )
    raw["block"]["gpu_uuid_after"] = after["gpu_uuid"]
    raw["block"]["expected_gpu_uuid"] = expected_gpu_uuid
    raw["block"]["benchmark_command"] = benchmark_command
    write_json(raw_path, raw)
    return raw, expected_gpu_uuid


def aggregate_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    failed_requests = sum(block["measurement"]["failed_requests"] for block in blocks)
    successful_texts = sum(block["measurement"]["successful_texts"] for block in blocks)
    return summarize_measurements(
        [block["measurement"]["texts_per_second"] for block in blocks],
        failed_requests=failed_requests,
        successful_texts=successful_texts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "experiment.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--health-timeout", type=int, default=900)
    parser.add_argument(
        "--telemetry-dir",
        type=Path,
        help="Optional directory for one telemetry JSON file per block.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    schedule = tuple(config["schedule"])
    validate_schedule(schedule)
    if config["dataset"]["concurrency"] != 32:
        raise ExperimentError("This reliability experiment only supports concurrency 32")
    if args.output_dir.exists():
        raise ExperimentError("Output directory must not already exist")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now(timezone.utc).isoformat()
    write_json(
        args.output_dir / "run.json",
        {
            "experiment": config["experiment"],
            "question": config["question"],
            "official_prior_result": {
                "status": "INVALID",
                "baseline_control": "UNSTABLE",
                "candidate_control": "UNSTABLE",
                "historical_regression_reproduced": False,
            },
            "schedule": list(schedule),
            "config": config,
            "started_at": started_at,
        },
    )

    blocks_by_variant = {"baseline": [], "candidate": []}
    gpu_uuid: str | None = None
    for order, variant_name in enumerate(schedule, start=1):
        raw, gpu_uuid = run_block(
            order=order,
            variant_name=variant_name,
            variant=config[variant_name],
            config=config,
            output_root=args.output_dir,
            expected_gpu_uuid=gpu_uuid,
            health_timeout=args.health_timeout,
            telemetry_dir=args.telemetry_dir,
        )
        blocks_by_variant[variant_name].append(raw)

    baseline = aggregate_blocks(blocks_by_variant["baseline"])
    candidate = aggregate_blocks(blocks_by_variant["candidate"])
    decision = regression_decision(
        baseline,
        candidate,
        max_cv_percent=config["stability"]["max_cv_percent"],
        regression_threshold_percent=config["regression"]["threshold_percent"],
    )
    bootstrap = bootstrap_mean_difference(
        baseline["values"],
        candidate["values"],
        samples=config["bootstrap"]["samples"],
        seed=config["bootstrap"]["seed"],
    )
    summary = {
        "schema_version": "inference-ci-reliability-001-summary-1",
        "experiment": config["experiment"],
        "question": config["question"],
        "gpu_uuid": gpu_uuid,
        "schedule": list(schedule),
        "baseline": baseline,
        "candidate": candidate,
        "decision": decision,
        "bootstrap_supporting_evidence": bootstrap,
        "no_outlier_removal": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
