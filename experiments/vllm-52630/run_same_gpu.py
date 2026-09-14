from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
import posixpath
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from aggregate import aggregate_repetitions, render_summary
from compare_results import compare_results, render_table
from harness_common import (
    ExperimentError,
    collect_gpu_metadata,
    collect_runtime_metadata,
    load_json,
    require_matching_gpu_uuid,
    write_json,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent


def variant_python_executable(variant: dict[str, Any]) -> str:
    """Return the environment Python used for that variant's benchmark client."""
    return posixpath.join(variant["environment_path"], "bin", "python")


def wait_for_health(server_url: str, process: subprocess.Popen[str], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ExperimentError("vLLM server exited before becoming healthy")
        try:
            response = httpx.get(f"{server_url}/health", timeout=3.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise ExperimentError(f"vLLM did not become healthy within {timeout}s")


def start_server(
    *,
    variant: dict[str, Any],
    config: dict[str, Any],
    log_path: Path,
) -> tuple[subprocess.Popen[str], list[str]]:
    executable = str(Path(variant["environment_path"]) / "bin" / "vllm")
    command = [executable, "serve", config["model"], *config["server"]["args"]]
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        command.extend(["--api-key", api_key])
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": "/workspace/hf-cache",
            "VLLM_CACHE_ROOT": variant["compile_cache"],
            "TORCHINDUCTOR_CACHE_DIR": variant["torchinductor_cache"],
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
        text=True,
    )
    process._inference_doctor_log_handle = log_handle  # type: ignore[attr-defined]
    return process, command


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=45)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    log_handle = getattr(process, "_inference_doctor_log_handle", None)
    if log_handle:
        log_handle.close()


def verify_no_vllm_process() -> None:
    result = subprocess.run(
        ["pgrep", "-af", "vllm( |$)"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [
        line
        for line in result.stdout.splitlines()
        if "pgrep -af vllm" not in line
    ]
    if lines:
        raise ExperimentError("vLLM processes remain after server shutdown")


def capture_gpu_cleanup(path: Path) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    path.write_text(result.stdout, encoding="utf-8")


def run_variant(
    *,
    name: str,
    variant: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    repetitions: int,
    health_timeout: int,
    python_executable: str,
) -> tuple[Path, dict[str, Any]]:
    variant_dir = output_root / name
    variant_dir.mkdir(parents=True, exist_ok=False)
    freeze = subprocess.run(
        [
            variant_python_executable(variant),
            "-m",
            "pip",
            "freeze",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (variant_dir / "packages.freeze.txt").write_text(
        freeze.stdout, encoding="utf-8"
    )
    log_path = variant_dir / "server.log"
    server_url = f"http://127.0.0.1:{config['server']['port']}"
    process, command = start_server(
        variant=variant,
        config=config,
        log_path=log_path,
    )
    try:
        wait_for_health(server_url, process, health_timeout)
        write_json(
            variant_dir / "server.json",
            {
                "launch_command": command,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "server_url": server_url,
            },
        )
        for repetition in range(1, repetitions + 1):
            repetition_dir = variant_dir / f"repetition-{repetition:03d}"
            subprocess.run(
                [
                    python_executable,
                    str(EXPERIMENT_DIR / "benchmark_pooling.py"),
                    "--server-url",
                    server_url,
                    "--version-label",
                    variant["label"],
                    "--variant",
                    name,
                    "--repetition",
                    str(repetition),
                    "--output-dir",
                    str(repetition_dir),
                ],
                check=True,
            )
    finally:
        stop_server(process)
    verify_no_vllm_process()
    capture_gpu_cleanup(variant_dir / "after-stop-gpu-memory.txt")
    raw_paths = sorted(variant_dir.glob("repetition-*/raw.json"))
    if len(raw_paths) != repetitions:
        raise ExperimentError("Not all repetitions produced raw evidence")
    aggregate = aggregate_repetitions(
        [load_json(path) for path in raw_paths],
        source_files=[str(path.relative_to(variant_dir)) for path in raw_paths],
        stability_tolerance_percent=config["comparison"]["stability_tolerance_percent"],
    )
    aggregate_path = variant_dir / "aggregate.json"
    write_json(aggregate_path, aggregate)
    print(f"\n{name}\n{render_summary(aggregate)}")
    return aggregate_path, aggregate


def run_comparison(
    *,
    name: str,
    baseline_path: Path,
    candidate_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    same_version: bool,
) -> str:
    result = compare_results(
        load_json(baseline_path),
        load_json(candidate_path),
        same_version=same_version,
        **{
            key: config["comparison"][key]
            for key in (
                "stability_tolerance_percent",
                "throughput_warn_percent",
                "throughput_fail_percent",
            )
        },
    )
    write_json(output_dir / f"{name}.json", result)
    (output_dir / f"{name}.txt").write_text(
        render_table(result), encoding="utf-8"
    )
    print(f"\n{name}\n{render_table(result)}")
    return result["overall_status"]


def run_stability_control(
    *,
    name: str,
    variant_dir: Path,
    repetitions: int,
    config: dict[str, Any],
    output_dir: Path,
) -> str:
    """Require every pair of repetitions to satisfy the stability tolerance."""
    statuses: dict[str, str] = {}
    for first, second in combinations(range(1, repetitions + 1), 2):
        pair_name = f"{name}-repetition-{first:03d}-vs-{second:03d}"
        statuses[pair_name] = run_comparison(
            name=pair_name,
            baseline_path=variant_dir / f"repetition-{first:03d}/normalized.json",
            candidate_path=variant_dir / f"repetition-{second:03d}/normalized.json",
            config=config,
            output_dir=output_dir,
            same_version=True,
        )
    overall = "PASS" if all(status == "PASS" for status in statuses.values()) else "UNSTABLE"
    write_json(
        output_dir / f"{name}.json",
        {"overall_status": overall, "pair_statuses": statuses},
    )
    return overall


def historical_regression_is_reproduced(
    *,
    baseline_control: str,
    candidate_control: str,
    cross_result: dict[str, Any],
    primary_concurrency: int,
    fail_threshold_percent: float,
) -> bool:
    if baseline_control != "PASS" or candidate_control != "PASS":
        return False
    metric_name = f"texts_per_second[{primary_concurrency}]"
    metric = next(
        (item for item in cross_result["metrics"] if item["metric"] == metric_name),
        None,
    )
    percentage_delta = None if metric is None else metric.get("percentage_delta")
    return (
        percentage_delta is not None
        and percentage_delta <= -fail_threshold_percent
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--health-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "experiment.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 3:
        raise SystemExit("This experiment requires at least 3 repetitions")
    config = load_json(args.config)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    baseline_gpu = collect_gpu_metadata()
    write_json(
        output_root / "run.json",
        {
            "experiment": config,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "baseline_gpu": baseline_gpu,
            "runtime": collect_runtime_metadata(),
        },
    )
    baseline_aggregate_path, _ = run_variant(
        name="baseline",
        variant=config["baseline"],
        config=config,
        output_root=output_root,
        repetitions=args.repetitions,
        health_timeout=args.health_timeout_seconds,
        python_executable=variant_python_executable(config["baseline"]),
    )
    candidate_gpu = collect_gpu_metadata()
    require_matching_gpu_uuid(baseline_gpu, candidate_gpu)
    run_metadata = load_json(output_root / "run.json")
    run_metadata["candidate_gpu"] = candidate_gpu
    write_json(output_root / "run.json", run_metadata)
    candidate_aggregate_path, _ = run_variant(
        name="candidate",
        variant=config["candidate"],
        config=config,
        output_root=output_root,
        repetitions=args.repetitions,
        health_timeout=args.health_timeout_seconds,
        python_executable=variant_python_executable(config["candidate"]),
    )

    comparisons_dir = output_root / "comparisons"
    comparisons_dir.mkdir()
    baseline_control = run_stability_control(
        name="baseline-vs-baseline",
        variant_dir=output_root / "baseline",
        repetitions=args.repetitions,
        config=config,
        output_dir=comparisons_dir,
    )
    candidate_control = run_stability_control(
        name="candidate-vs-candidate",
        variant_dir=output_root / "candidate",
        repetitions=args.repetitions,
        config=config,
        output_dir=comparisons_dir,
    )
    cross_status = run_comparison(
        name="baseline-vs-candidate",
        baseline_path=baseline_aggregate_path,
        candidate_path=candidate_aggregate_path,
        config=config,
        output_dir=comparisons_dir,
        same_version=False,
    )
    cross_result = load_json(comparisons_dir / "baseline-vs-candidate.json")
    primary_concurrency = int(config["primary_concurrency"])
    summary = {
        "baseline_control": baseline_control,
        "candidate_control": candidate_control,
        "baseline_vs_candidate": cross_status,
        "gpu_uuid": baseline_gpu["gpu_uuid"],
        "cross_version_valid": baseline_control == "PASS" and candidate_control == "PASS",
        "primary_concurrency": primary_concurrency,
        "primary_metric": next(
            metric
            for metric in cross_result["metrics"]
            if metric["metric"] == f"texts_per_second[{primary_concurrency}]"
        ),
        "historical_regression_reproduced": historical_regression_is_reproduced(
            baseline_control=baseline_control,
            candidate_control=candidate_control,
            cross_result=cross_result,
            primary_concurrency=primary_concurrency,
            fail_threshold_percent=config["comparison"]["throughput_fail_percent"],
        ),
        "baseline_best_stable_plateau": load_json(baseline_aggregate_path)["metrics"]["best_stable_plateau"],
        "candidate_best_stable_plateau": load_json(candidate_aggregate_path)["metrics"]["best_stable_plateau"],
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["cross_version_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
