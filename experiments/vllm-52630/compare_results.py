from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from harness_common import ExperimentError, load_json, write_json


def _critical_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    parameters = payload["benchmark_parameters"]
    return {
        key: parameters[key]
        for key in (
            "experiment",
            "model",
            "server",
            "dataset",
            "batch_count",
            "request_shape",
            "concurrency",
            "warmup_batches",
        )
    }


def _percent_delta(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return ((candidate - baseline) / baseline) * 100


def _point_value(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("mean")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _metric(
    *,
    name: str,
    baseline: float | None,
    candidate: float | None,
    same_version: bool,
    tolerance: float,
    warn: float,
    fail: float,
) -> dict[str, Any]:
    if baseline is None or candidate is None:
        status = "WARN"
        absolute = None
        percentage = None
    else:
        absolute = candidate - baseline
        percentage = _percent_delta(baseline, candidate)
        if same_version:
            status = (
                "PASS"
                if percentage is not None and abs(percentage) <= tolerance
                else "UNSTABLE"
            )
        elif percentage is None:
            status = "WARN"
        else:
            regression = -percentage
            if regression >= fail:
                status = "FAIL"
            elif regression >= warn:
                status = "WARN"
            else:
                status = "PASS"
    return {
        "metric": name,
        "baseline": baseline,
        "candidate": candidate,
        "absolute_delta": absolute,
        "percentage_delta": percentage,
        "status": status,
    }


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    same_version: bool = False,
    stability_tolerance_percent: float = 5.0,
    throughput_warn_percent: float = 5.0,
    throughput_fail_percent: float = 10.0,
) -> dict[str, Any]:
    if _critical_parameters(baseline) != _critical_parameters(candidate):
        raise ExperimentError("Benchmark parameters are not comparable")
    baseline_uuid = baseline.get("environment", {}).get("gpu", {}).get("gpu_uuid")
    candidate_uuid = candidate.get("environment", {}).get("gpu", {}).get("gpu_uuid")
    if baseline_uuid and candidate_uuid and baseline_uuid != candidate_uuid:
        raise ExperimentError("GPU UUIDs differ between compared results")

    metrics: list[dict[str, Any]] = []
    baseline_texts = baseline["metrics"]["texts_per_second"]
    candidate_texts = candidate["metrics"]["texts_per_second"]
    for level in sorted(baseline_texts, key=int):
        metrics.append(
            _metric(
                name=f"texts_per_second[{level}]",
                baseline=_point_value(baseline_texts.get(level)),
                candidate=_point_value(candidate_texts.get(level)),
                same_version=same_version,
                tolerance=stability_tolerance_percent,
                warn=throughput_warn_percent,
                fail=throughput_fail_percent,
            )
        )

    baseline_requests = baseline["metrics"]["requests"]
    candidate_requests = candidate["metrics"]["requests"]
    failed_delta = candidate_requests["failed"] - baseline_requests["failed"]
    metrics.append(
        {
            "metric": "failed_requests",
            "baseline": baseline_requests["failed"],
            "candidate": candidate_requests["failed"],
            "absolute_delta": failed_delta,
            "percentage_delta": _percent_delta(
                baseline_requests["failed"], candidate_requests["failed"]
            ),
            "status": (
                "UNSTABLE"
                if same_version and failed_delta != 0
                else "FAIL"
                if not same_version and failed_delta > 0
                else "PASS"
            ),
        }
    )
    statuses = {metric["status"] for metric in metrics}
    if "UNSTABLE" in statuses:
        overall = "UNSTABLE"
    elif "FAIL" in statuses:
        overall = "FAIL"
    else:
        overall = "PASS"
    return {
        "schema_version": "vllm-52630-comparison-1",
        "overall_status": overall,
        "same_version": same_version,
        "thresholds": {
            "stability_tolerance_percent": stability_tolerance_percent,
            "throughput_warn_percent": throughput_warn_percent,
            "throughput_fail_percent": throughput_fail_percent,
        },
        "baseline": baseline.get("environment", {}),
        "candidate": candidate.get("environment", {}),
        "metrics": metrics,
    }


def render_table(result: dict[str, Any]) -> str:
    lines = [
        f"Overall: {result['overall_status']}",
        "Metric                         Baseline       Candidate      Delta       Status",
        "-" * 86,
    ]
    for metric in result["metrics"]:
        baseline = metric["baseline"]
        candidate = metric["candidate"]
        delta = metric["percentage_delta"]
        lines.append(
            f"{metric['metric']:<30} "
            f"{str(baseline)[:13]:>13} "
            f"{str(candidate)[:13]:>13} "
            f"{('N/A' if delta is None else f'{delta:+.2f}%'):>10} "
            f"{metric['status']:>8}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--same-version", action="store_true")
    parser.add_argument("--stability-tolerance-percent", type=float, default=5.0)
    parser.add_argument("--throughput-warn-percent", type=float, default=5.0)
    parser.add_argument("--throughput-fail-percent", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = compare_results(
            load_json(args.baseline),
            load_json(args.candidate),
            same_version=args.same_version,
            stability_tolerance_percent=args.stability_tolerance_percent,
            throughput_warn_percent=args.throughput_warn_percent,
            throughput_fail_percent=args.throughput_fail_percent,
        )
    except (ExperimentError, KeyError) as exc:
        raise SystemExit(f"Comparison failed: {exc}") from exc
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2) if args.json else render_table(result))
    return 1 if result["overall_status"] in {"FAIL", "UNSTABLE"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
