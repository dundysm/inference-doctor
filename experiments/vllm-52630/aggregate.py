from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from harness_common import (
    ExperimentError,
    coefficient_of_variation,
    finite_numbers,
    load_json,
    sample_standard_deviation,
    summarize,
    write_json,
)


def _comparison_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(parameters)
    result.pop("repetition", None)
    return result


def _metric_stats(values: list[Any]) -> dict[str, Any]:
    numbers = finite_numbers(values)
    summary = summarize(numbers)
    return {
        **summary,
        "sample_standard_deviation": sample_standard_deviation(numbers),
        "coefficient_of_variation": coefficient_of_variation(numbers),
        "repetition_values": numbers,
    }


def aggregate_repetitions(
    raw_runs: list[dict[str, Any]],
    *,
    source_files: list[str] | None = None,
    stability_tolerance_percent: float = 5.0,
) -> dict[str, Any]:
    if not raw_runs:
        raise ExperimentError("At least one repetition is required")
    first = raw_runs[0]
    if first.get("complete") is not True:
        raise ExperimentError("First repetition is incomplete")
    base_parameters = _comparison_parameters(first["benchmark_parameters"])
    base_environment = first["environment"]
    levels = sorted(first["concurrency_results"], key=int)
    for index, run in enumerate(raw_runs, start=1):
        if run.get("complete") is not True:
            raise ExperimentError(f"Repetition {index} is incomplete")
        if _comparison_parameters(run["benchmark_parameters"]) != base_parameters:
            raise ExperimentError(f"Repetition {index} parameters differ")
        if run["environment"] != base_environment:
            raise ExperimentError(f"Repetition {index} environment differs")
        if sorted(run["concurrency_results"], key=int) != levels:
            raise ExperimentError(f"Repetition {index} concurrency levels differ")

    texts_stats = {
        level: _metric_stats(
            [run["concurrency_results"][level]["texts_per_second"] for run in raw_runs]
        )
        for level in levels
    }
    request_latency_median_stats = {
        level: _metric_stats(
            [
                run["concurrency_results"][level]["latency_ms"]["median"]
                for run in raw_runs
            ]
        )
        for level in levels
    }
    request_latency_p95_stats = {
        level: _metric_stats(
            [
                run["concurrency_results"][level]["latency_ms"]["p95"]
                for run in raw_runs
            ]
        )
        for level in levels
    }
    stable_levels = [
        level
        for level in levels
        if (
            texts_stats[level]["coefficient_of_variation"] is not None
            and texts_stats[level]["coefficient_of_variation"]
            <= stability_tolerance_percent / 100
        )
    ]
    best_level = max(
        stable_levels,
        key=lambda level: texts_stats[level]["mean"],
        default=None,
    )
    best_observed_level = max(
        levels,
        key=lambda level: texts_stats[level]["mean"]
        if texts_stats[level]["mean"] is not None
        else float("-inf"),
        default=None,
    )
    successful = sum(
        run["concurrency_results"][level]["successful_requests"]
        for run in raw_runs
        for level in levels
    )
    failed = sum(
        run["concurrency_results"][level]["failed_requests"]
        for run in raw_runs
        for level in levels
    )

    result = {
        "schema_version": "vllm-52630-aggregate-1",
        "environment": base_environment,
        "benchmark_parameters": {
            **base_parameters,
            "repetitions": len(raw_runs),
        },
        "metrics": {
            "texts_per_second": texts_stats,
            "request_latency_median_ms": request_latency_median_stats,
            "request_latency_p95_ms": request_latency_p95_stats,
            "requests": {"successful": successful, "failed": failed},
            "best_stable_plateau": (
                {
                    "concurrency": int(best_level),
                    "texts_per_second": texts_stats[best_level]["mean"],
                    "stability_tolerance_percent": stability_tolerance_percent,
                }
                if best_level is not None
                else None
            ),
            "best_observed": (
                {
                    "concurrency": int(best_observed_level),
                    "texts_per_second": texts_stats[best_observed_level]["mean"],
                }
                if best_observed_level is not None
                else None
            ),
        },
        "aggregation": {
            "repetitions": len(raw_runs),
            "source_files": source_files or [],
            "stability_tolerance_percent": stability_tolerance_percent,
        },
    }
    return result


def render_summary(result: dict[str, Any]) -> str:
    lines = [f"Repetitions: {result['aggregation']['repetitions']}"]
    for level in sorted(result["metrics"]["texts_per_second"], key=int):
        throughput = result["metrics"]["texts_per_second"][level]
        latency_median = result["metrics"]["request_latency_median_ms"][level]
        latency_p95 = result["metrics"]["request_latency_p95_ms"][level]
        lines.append(
            f"Concurrency {level}: texts/s mean {throughput['mean']:.2f}, "
            f"median {throughput['median']:.2f}, CV "
            f"{throughput['coefficient_of_variation']:.4f}; "
            f"request latency median/run median {latency_median['median']:.2f} ms, "
            f"p95/run median {latency_p95['median']:.2f} ms"
        )
    plateau = result["metrics"]["best_stable_plateau"]
    lines.append(
        "Best stable plateau: "
        + ("N/A" if plateau is None else f"concurrency {plateau['concurrency']}, {plateau['texts_per_second']:.2f} texts/s")
    )
    requests = result["metrics"]["requests"]
    lines.append(f"Requests: successful {requests['successful']}, failed {requests['failed']}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stability-tolerance-percent", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.input_dir.glob("repetition-*/raw.json"))
    if not paths:
        raise SystemExit(f"No repetition-*/raw.json files under {args.input_dir}")
    result = aggregate_repetitions(
        [load_json(path) for path in paths],
        source_files=[str(path.relative_to(args.input_dir)) for path in paths],
        stability_tolerance_percent=args.stability_tolerance_percent,
    )
    write_json(args.output, result)
    print(render_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
