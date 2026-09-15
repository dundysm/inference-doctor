from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import httpx
import typer

from inference_doctor.benchmark_results import (
    BenchmarkResultError,
    ensure_comparable,
    load_normalized_benchmark,
)
from inference_doctor.collectors.prometheus import build_snapshot
from inference_doctor.comparison import (
    ComparisonStatus,
    ComparisonThresholds,
    compare_benchmarks,
)
from inference_doctor.comparison_report import render_comparison
from inference_doctor.diagnostics import diagnose
from inference_doctor.quality import (
    QualityInputError,
    QualityThresholds,
    load_and_compare_directories,
    metric_choices,
)
from inference_doctor.quality_report import render_quality_report
from inference_doctor.report import render_report


app = typer.Typer(
    help="Gate LLM inference performance regressions and diagnose vLLM bottlenecks."
)


@app.callback()
def main() -> None:
    pass


@app.command(name="diagnose")
def diagnose_command(
    prometheus: str = typer.Option(..., "--prometheus"),
    window: str = typer.Option("15m", "--window"),
    ttft_slo_ms: float | None = typer.Option(
        None,
        "--ttft-slo-ms",
    ),
    tpot_slo_ms: float | None = typer.Option(
        None,
        "--tpot-slo-ms",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
    ),
) -> None:
    try:
        snapshot = build_snapshot(
            prometheus_url=prometheus,
            window=window,
            ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms,
        )
    except (httpx.HTTPError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    findings = diagnose(snapshot)

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "snapshot": asdict(snapshot),
                    "findings": [
                        finding.to_dict()
                        for finding in findings
                    ],
                },
                indent=2,
            )
        )
        return

    render_report(snapshot, findings)


@app.command(name="compare")
def compare_command(
    baseline: Path = typer.Option(
        ...,
        "--baseline",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    candidate: Path = typer.Option(
        ...,
        "--candidate",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    latency_warn_percent: float = typer.Option(
        5.0,
        "--latency-warn-percent",
    ),
    latency_fail_percent: float = typer.Option(
        10.0,
        "--latency-fail-percent",
    ),
    throughput_warn_percent: float = typer.Option(
        5.0,
        "--throughput-warn-percent",
    ),
    throughput_fail_percent: float = typer.Option(
        10.0,
        "--throughput-fail-percent",
    ),
    max_failed_request_increase: int = typer.Option(
        0,
        "--max-failed-request-increase",
    ),
    stability_tolerance_percent: float = typer.Option(
        5.0,
        "--stability-tolerance-percent",
        help="Maximum absolute variation allowed for same-version controls.",
    ),
    same_version: bool = typer.Option(
        False,
        "--same-version",
        help="Evaluate the pair as a same-version stability control.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        thresholds = ComparisonThresholds(
            latency_warn_percent=latency_warn_percent,
            latency_fail_percent=latency_fail_percent,
            throughput_warn_percent=throughput_warn_percent,
            throughput_fail_percent=throughput_fail_percent,
            max_failed_request_increase=max_failed_request_increase,
            stability_tolerance_percent=stability_tolerance_percent,
        )
        thresholds.validate()
        baseline_result = load_normalized_benchmark(baseline)
        candidate_result = load_normalized_benchmark(candidate)
        ensure_comparable(baseline_result, candidate_result)
        comparison = compare_benchmarks(
            baseline_result,
            candidate_result,
            thresholds,
            same_version=same_version,
        )
    except (BenchmarkResultError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc

    if json_output:
        typer.echo(json.dumps(comparison.to_dict(), indent=2))
    else:
        render_comparison(comparison)

    if comparison.overall_status in (
        ComparisonStatus.FAIL,
        ComparisonStatus.UNSTABLE,
    ):
        raise typer.Exit(1)


@app.command(name="compare-runs")
def compare_runs_command(
    baseline: Path = typer.Option(
        ...,
        "--baseline",
        exists=True,
        file_okay=False,
        readable=True,
        help="Directory containing repeated normalized baseline runs.",
    ),
    candidate: Path = typer.Option(
        ...,
        "--candidate",
        exists=True,
        file_okay=False,
        readable=True,
        help="Directory containing repeated normalized candidate runs.",
    ),
    metric: str = typer.Option(
        "output_token_throughput",
        "--metric",
        help="Metric to compare: " + ", ".join(metric_choices()),
    ),
    stability_cv_percent: float = typer.Option(
        5.0,
        "--stability-cv-percent",
        help="Maximum allowed CV for each repeated run set.",
    ),
    regression_percent: float = typer.Option(
        10.0,
        "--regression-percent",
        help="Regression threshold for a stable candidate.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = load_and_compare_directories(
            baseline,
            candidate,
            metric=metric,
            thresholds=QualityThresholds(
                stability_cv_percent=stability_cv_percent,
                regression_percent=regression_percent,
            ),
        )
    except (BenchmarkResultError, QualityInputError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(3) from exc

    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        render_quality_report(report)

    exit_codes = {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}
    raise typer.Exit(exit_codes[report["result"]])


if __name__ == "__main__":
    app()
