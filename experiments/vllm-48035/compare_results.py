from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from aggregate import aggregate_raw_runs, render_summary
from harness_common import (
    GpuIdentityError,
    load_json,
    require_matching_gpu_uuid,
    write_json,
)


def prepare_aggregate(input_dir: Path) -> tuple[Path, list[Path]]:
    raw_paths = sorted(input_dir.glob("repetition-*/raw.json"))
    if len(raw_paths) < 2:
        raise ValueError(
            f"At least two complete repetitions are required under {input_dir}"
        )

    raw_runs = [load_json(path) for path in raw_paths]
    normalized_paths: list[Path] = []
    for raw_path, raw_run in zip(raw_paths, raw_runs, strict=True):
        normalized = aggregate_raw_runs(
            [raw_run],
            source_files=[str(raw_path.relative_to(input_dir))],
        )
        normalized_path = raw_path.with_name("normalized.json")
        write_json(normalized_path, normalized)
        normalized_paths.append(normalized_path)

    aggregate = aggregate_raw_runs(
        raw_runs,
        source_files=[str(path.relative_to(input_dir)) for path in raw_paths],
    )
    aggregate_path = input_dir / "aggregate.normalized.json"
    write_json(aggregate_path, aggregate)
    print(f"\n{input_dir.name} aggregate\n{render_summary(aggregate)}")
    return aggregate_path, normalized_paths


def run_comparison(
    *,
    name: str,
    baseline: Path,
    candidate: Path,
    comparison_config: dict[str, Any],
    output_dir: Path,
    same_version: bool = False,
) -> str:
    command = [
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
        command.append("--same-version")
    json_result = subprocess.run(
        [*command, "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if json_result.returncode not in (0, 1):
        raise RuntimeError(
            f"Comparison {name!r} failed: {json_result.stderr.strip()}"
        )
    payload = json.loads(json_result.stdout)
    write_json(output_dir / f"{name}.json", payload)

    terminal_result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if terminal_result.returncode not in (0, 1):
        raise RuntimeError(
            f"Comparison {name!r} failed: {terminal_result.stderr.strip()}"
        )
    (output_dir / f"{name}.txt").write_text(
        terminal_result.stdout,
        encoding="utf-8",
    )
    print(f"\n{name}\n{terminal_result.stdout}")
    return payload["overall_status"]


def run_offline_comparisons(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline_aggregate, baseline_repetitions = prepare_aggregate(baseline_dir)
    candidate_aggregate, candidate_repetitions = prepare_aggregate(
        candidate_dir
    )

    baseline_payload = load_json(baseline_aggregate)
    candidate_payload = load_json(candidate_aggregate)
    gpu_uuid = require_matching_gpu_uuid(
        baseline_payload,
        candidate_payload,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_config = config["comparison"]
    baseline_control = run_comparison(
        name="baseline-vs-baseline",
        baseline=baseline_repetitions[0],
        candidate=baseline_repetitions[1],
        comparison_config=comparison_config,
        output_dir=output_dir,
        same_version=True,
    )
    candidate_control = run_comparison(
        name="candidate-vs-candidate",
        baseline=candidate_repetitions[0],
        candidate=candidate_repetitions[1],
        comparison_config=comparison_config,
        output_dir=output_dir,
        same_version=True,
    )
    cross_version = run_comparison(
        name="baseline-vs-candidate",
        baseline=baseline_aggregate,
        candidate=candidate_aggregate,
        comparison_config=comparison_config,
        output_dir=output_dir,
    )

    controls_pass = (
        baseline_control == "PASS" and candidate_control == "PASS"
    )
    summary = {
        "gpu_uuid": gpu_uuid,
        "gpu_uuid_match": True,
        "baseline_control": baseline_control,
        "candidate_control": candidate_control,
        "baseline_vs_candidate": cross_version,
        "cross_version_valid": controls_pass,
        "historical_regression_reproduced": (
            controls_pass and cross_version == "FAIL"
        ),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiment.json"),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        summary = run_offline_comparisons(
            baseline_dir=args.baseline_dir,
            candidate_dir=args.candidate_dir,
            output_dir=args.output_dir,
            config=load_json(args.config),
        )
    except GpuIdentityError as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_dir / "summary.json",
            {
                "gpu_uuid_match": False,
                "cross_version_valid": False,
                "error": str(exc),
            },
        )
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_dir / "summary.json",
            {
                "cross_version_valid": False,
                "error": str(exc),
            },
        )
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    return 0 if summary["cross_version_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
