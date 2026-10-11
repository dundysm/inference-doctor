"""Command-line interface.

Exit codes are the same for every command:

    0  PASS
    1  FAIL: a regression was measured
    2  INCONCLUSIVE (compare-runs) or UNSTABLE (compare): no decision, the
       measurement could not be trusted
    3  error: no decision was made because of bad arguments, unreadable or
       invalid input, an unreachable Prometheus, or an internal error

Set INFERENCE_DOCTOR_DEBUG=1 to see the traceback of an internal error.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import typer
from typer.core import TyperGroup

from inference_doctor import __version__
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

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_NO_DECISION = 2
EXIT_ERROR = 3


def _is_click_error(exc: BaseException, name: str) -> bool:
    # typer raises click's exceptions, or its vendored copy of them in newer releases.
    return any(cls.__name__ == name for cls in type(exc).__mro__)


class _Group(TyperGroup):
    """Report usage errors and internal errors with exit code 3.

    click uses 2 for usage errors and an uncaught exception exits 1. Both would
    read as a verdict (INCONCLUSIVE or FAIL) to a CI job.
    """

    def make_context(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().make_context(*args, **kwargs)
        except Exception as exc:
            if _is_click_error(exc, "UsageError"):
                exc.exit_code = EXIT_ERROR  # type: ignore[attr-defined]
            raise

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except (typer.Exit, typer.Abort):
            raise
        except Exception as exc:
            if _is_click_error(exc, "ClickException"):
                exc.exit_code = EXIT_ERROR  # type: ignore[attr-defined]
                raise
            if os.environ.get("INFERENCE_DOCTOR_DEBUG"):
                raise
            typer.echo(
                f"inference-doctor: internal error: {type(exc).__name__}: {exc}\n"
                "  (set INFERENCE_DOCTOR_DEBUG=1 for the full traceback)",
                err=True,
            )
            raise typer.Exit(EXIT_ERROR) from exc


app = typer.Typer(
    cls=_Group,
    help="Gate LLM inference performance regressions and diagnose vLLM bottlenecks.",
    pretty_exceptions_show_locals=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"inference-doctor {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Exit codes: 0 PASS, 1 FAIL, 2 INCONCLUSIVE or UNSTABLE, 3 error (no decision)."""


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
        raise typer.Exit(EXIT_ERROR) from exc

    findings = diagnose(snapshot)

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "snapshot": asdict(snapshot),
                    "findings": [finding.to_dict() for finding in findings],
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
        raise typer.Exit(EXIT_ERROR) from exc

    if json_output:
        typer.echo(json.dumps(comparison.to_dict(), indent=2))
    else:
        render_comparison(comparison)

    if comparison.overall_status == ComparisonStatus.FAIL:
        raise typer.Exit(EXIT_FAIL)
    if comparison.overall_status == ComparisonStatus.UNSTABLE:
        raise typer.Exit(EXIT_NO_DECISION)


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
    regression_percent: float = typer.Option(
        10.0,
        "--regression-percent",
        help="FAIL when the whole confidence interval for the regression is at or past this "
        "percentage; PASS when the whole interval is below it.",
    ),
    confidence: float = typer.Option(
        0.90,
        "--confidence",
        help="Two-sided confidence level of the interval (0.90 makes each bound a one-sided "
        "95% bound).",
    ),
    min_repetitions: int = typer.Option(
        3,
        "--min-repetitions",
        help="Repetitions required on each side. 5 or more is recommended.",
    ),
    stability_cv_percent: float | None = typer.Option(
        None,
        "--stability-cv-percent",
        help="Optional ceiling on each side's CV. Off by default: the confidence interval "
        "already widens with noise.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = load_and_compare_directories(
            baseline,
            candidate,
            metric=metric,
            thresholds=QualityThresholds(
                regression_percent=regression_percent,
                confidence=confidence,
                min_repetitions=min_repetitions,
                stability_cv_percent=stability_cv_percent,
            ),
        )
    except (BenchmarkResultError, QualityInputError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc

    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        render_quality_report(report)

    exit_codes = {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL, "INCONCLUSIVE": EXIT_NO_DECISION}
    raise typer.Exit(exit_codes[report["result"]])


if __name__ == "__main__":
    app()
