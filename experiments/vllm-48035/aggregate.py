from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from harness_common import (
    finite_numbers,
    load_json,
    normalized_result,
    write_json,
)


def aggregate_raw_runs(
    raw_runs: list[dict[str, Any]],
    source_files: list[str] | None = None,
) -> dict[str, Any]:
    if not raw_runs:
        raise ValueError("At least one raw repetition is required")

    first = raw_runs[0]
    environment = first["environment"]
    base_parameters = first["benchmark_parameters"]

    for index, run in enumerate(raw_runs, start=1):
        if run.get("complete") is not True:
            raise ValueError(f"Repetition {index} is incomplete")
        if run["environment"] != environment:
            raise ValueError(
                f"Repetition {index} has different environment metadata"
            )
        if run["benchmark_parameters"] != base_parameters:
            raise ValueError(
                f"Repetition {index} has different benchmark parameters"
            )

    all_records = [
        record
        for run in raw_runs
        for record in run.get("requests", [])
    ]
    total_duration = sum(
        finite_numbers(
            run.get("measured_duration_seconds") for run in raw_runs
        )
    )
    parameters = dict(base_parameters)
    parameters["repetitions"] = len(raw_runs)
    result = normalized_result(
        environment=environment,
        benchmark_parameters=parameters,
        records=all_records,
        measured_duration_seconds=total_duration,
    )

    request_decode_rates = finite_numbers(
        record.get("decode_tokens_per_second")
        for record in all_records
        if record.get("success")
    )
    run_decode_means: list[float] = []
    for run in raw_runs:
        rates = finite_numbers(
            record.get("decode_tokens_per_second")
            for record in run.get("requests", [])
            if record.get("success")
        )
        if rates:
            run_decode_means.append(statistics.fmean(rates))

    run_mean = (
        statistics.fmean(run_decode_means)
        if run_decode_means
        else None
    )
    run_stddev = (
        statistics.stdev(run_decode_means)
        if len(run_decode_means) >= 2
        else None
    )
    run_cv = (
        run_stddev / run_mean
        if run_stddev is not None and run_mean not in (None, 0)
        else None
    )

    result["aggregation"] = {
        "repetitions": len(raw_runs),
        "source_files": source_files or [],
        "decode_tokens_per_second": {
            "mean": (
                statistics.fmean(request_decode_rates)
                if request_decode_rates
                else None
            ),
            "median": (
                statistics.median(request_decode_rates)
                if request_decode_rates
                else None
            ),
        },
        "run_to_run_decode_throughput": {
            "repetition_means": run_decode_means,
            "mean": run_mean,
            "sample_standard_deviation": run_stddev,
            "coefficient_of_variation": run_cv,
        },
    }
    return result


def render_summary(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    aggregation = result["aggregation"]
    decode = aggregation["decode_tokens_per_second"]
    run_to_run = aggregation["run_to_run_decode_throughput"]

    def display(value: float | None, digits: int = 2) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    lines = [
        f"Repetitions: {aggregation['repetitions']}",
        (
            "Decode tok/s: mean "
            f"{display(decode['mean'])}, median {display(decode['median'])}"
        ),
        (
            "TTFT ms: mean "
            f"{display(metrics['ttft_ms']['mean'])}, median "
            f"{display(metrics['ttft_ms']['median'])}, p95 "
            f"{display(metrics['ttft_ms']['p95'])}"
        ),
        (
            "TPOT ms: mean "
            f"{display(metrics['tpot_ms']['mean'])}, median "
            f"{display(metrics['tpot_ms']['median'])}, p95 "
            f"{display(metrics['tpot_ms']['p95'])}"
        ),
        (
            "Requests: successful "
            f"{metrics['requests']['successful']}, failed "
            f"{metrics['requests']['failed']}"
        ),
        (
            "Run decode tok/s: mean "
            f"{display(run_to_run['mean'])}, sample stddev "
            f"{display(run_to_run['sample_standard_deviation'])}, CV "
            f"{display(run_to_run['coefficient_of_variation'], 4)}"
        ),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_paths = sorted(args.input_dir.glob("repetition-*/raw.json"))
    if not raw_paths:
        raise SystemExit(
            f"No repetition-*/raw.json files found under {args.input_dir}"
        )
    runs = [load_json(path) for path in raw_paths]
    result = aggregate_raw_runs(
        runs,
        source_files=[
            str(path.relative_to(args.input_dir)) for path in raw_paths
        ],
    )
    write_json(args.output, result)
    print(render_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
