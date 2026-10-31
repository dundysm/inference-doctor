"""Command-line interface.

Exit codes are the same for every command:

    0  PASS
    1  FAIL: a regression was measured
    2  INCONCLUSIVE, or UNSTABLE for compare: no decision, the measurement could
       not be trusted
    3  error: no decision was made because of bad arguments, unreadable or
       invalid input, an unreachable Prometheus, or an internal error

diagnose exits 1 when it reports a finding at or above --fail-on (critical by
default), and 0 otherwise.

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
from inference_doctor.adapters import FORMATS, SourceOptions, load_directory
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


FAIL_ON = {"critical": ("critical",), "warning": ("critical", "warning"), "never": ()}


@app.command(
    name="diagnose",
    help="(Experimental) Read a live vLLM server's Prometheus metrics and report queue, "
    "prefill, decode and KV-cache pressure. Read-only.",
)
def diagnose_command(
    prometheus: str = typer.Option(..., "--prometheus", help="Prometheus base URL."),
    window: str = typer.Option("15m", "--window", help="Query window, such as 5m or 1h."),
    ttft_slo_ms: float | None = typer.Option(
        None, "--ttft-slo-ms", help="Time-to-first-token p95 SLO in milliseconds."
    ),
    tpot_slo_ms: float | None = typer.Option(
        None, "--tpot-slo-ms", help="Time-per-output-token p95 SLO in milliseconds."
    ),
    fail_on: str = typer.Option(
        "critical",
        "--fail-on",
        help="Exit 1 when a finding of this severity or worse is reported: critical, "
        "warning, or never.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
    ),
) -> None:
    if fail_on not in FAIL_ON:
        typer.echo(
            f"Error: --fail-on must be one of {', '.join(FAIL_ON)}, not '{fail_on}'", err=True
        )
        raise typer.Exit(EXIT_ERROR)
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
    else:
        render_report(snapshot, findings)

    if any(finding.severity.value in FAIL_ON[fail_on] for finding in findings):
        raise typer.Exit(EXIT_FAIL)


@app.command(
    name="compare",
    help="(Deprecated) Compare one normalized result file with another. A single run per "
    "side cannot measure noise; use compare-runs.",
)
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
    typer.echo(
        "inference-doctor compare: deprecated. One run per side cannot measure benchmark "
        "noise; use compare-runs with repeated runs. compare will be removed in a future "
        "release.",
        err=True,
    )
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
    if comparison.overall_status in (ComparisonStatus.UNSTABLE, ComparisonStatus.INCONCLUSIVE):
        raise typer.Exit(EXIT_NO_DECISION)


METRIC_HELP = "Metric to compare; repeat to gate on several (FAIL if any fails). Choices: " + (
    ", ".join(metric_choices())
)


@app.command(name="compare-runs")
def compare_runs_command(
    baseline: Path = typer.Option(
        ...,
        "--baseline",
        exists=True,
        file_okay=False,
        readable=True,
        help="Directory of repeated baseline results, one file or JSONL line per run.",
    ),
    candidate: Path = typer.Option(
        ...,
        "--candidate",
        exists=True,
        file_okay=False,
        readable=True,
        help="Directory of repeated candidate results, one file or JSONL line per run.",
    ),
    metric: list[str] = typer.Option(
        ["output_token_throughput"],
        "--metric",
        help=METRIC_HELP,
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
    input_format: str = typer.Option(
        "auto",
        "--format",
        help="Result format: " + ", ".join(FORMATS) + ". auto detects it per file.",
    ),
    gpu: str | None = typer.Option(
        None, "--gpu", help="GPU name to record when the results do not include one."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Model name to record when the results do not include one."
    ),
    guidellm_benchmark: int | None = typer.Option(
        None,
        "--guidellm-benchmark",
        help="Which benchmark of a GuideLLM sweep to compare, by index.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        report = load_and_compare_directories(
            baseline,
            candidate,
            metric=metric,
            format=input_format,
            options=SourceOptions(gpu=gpu, model=model, guidellm_benchmark=guidellm_benchmark),
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


@app.command(
    name="cost",
    help="(Experimental) Turn vLLM/SGLang Prometheus scrapes or benchmark results into "
    "$/1M tokens, estimated $ wasted, and ranked fixes. --demo runs on bundled samples.",
)
def cost_command(
    metrics: list[Path] = typer.Option(
        [], "--metrics", help="Saved /metrics scrape(s); give two for a measured window."
    ),
    url: str | None = typer.Option(None, "--url", help="Live /metrics URL; scraped twice."),
    runs: Path | None = typer.Option(None, "--runs", help="Directory of benchmark results."),
    seconds: float | None = typer.Option(None, "--seconds", help="Window or uptime in seconds."),
    gpu: str = typer.Option("H100-SXM", "--gpu", help="GPU type from the price table."),
    gpus: int = typer.Option(1, "--gpus", help="GPU count."),
    price: float | None = typer.Option(None, "--price", help="Your $/GPU-hour (overrides table)."),
    provider: str | None = typer.Option(None, "--provider", help="Price table column."),
    max_num_seqs: int = typer.Option(256, "--max-num-seqs", help="Server's --max-num-seqs."),
    replicas: int = typer.Option(1, "--replicas", help="Engine replicas behind the metrics."),
    tp: int = typer.Option(1, "--tp", help="Tensor-parallel size (right-size rounds to it)."),
    api_model: str | None = typer.Option(None, "--api-model", help="Hosted API to compare."),
    demo: bool = typer.Option(False, "--demo", help="Use bundled sample metrics."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    import time
    from importlib import resources

    from inference_doctor import cost as costmod
    from inference_doctor import cost_metrics as cm

    try:
        if demo:
            base = resources.files("inference_doctor").joinpath("data", "cost_demo")
            texts = [base.joinpath(n).read_text() for n in ("vllm-t0.prom", "vllm-t1.prom")]
            gpu, gpus, replicas, tp = "H100-SXM", 8, 4, 2
            api_model = api_model or "gpt-oss-120b"
        elif url:
            first = cm.fetch(url)
            time.sleep(seconds or 30)
            texts = [first, cm.fetch(url)]
            seconds = seconds or 30
        else:
            texts = [p.read_text() for p in metrics]
        if runs is not None:
            loaded = load_directory(runs)
            usage = costmod.usage_from_benchmarks([r.result for r in loaded.records])
        elif texts:
            snaps = [cm.snapshot_from_text(t) for t in texts[:2]]
            usage = costmod.usage_from_window(
                cm.window(snaps[0], snaps[1] if len(snaps) > 1 else None, seconds)
            )
        else:
            raise ValueError("Give --metrics, --url, --runs or --demo")
        report = costmod.analyze(
            usage, gpu, gpus, price, provider, max_num_seqs, replicas, tp, api_model
        )
    except (ValueError, OSError, BenchmarkResultError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    typer.echo(json.dumps(report.to_dict(), indent=2) if json_output else costmod.render(report))


@app.command(
    name="cost-compare",
    help="(Experimental) Cost-regression gate on $/1M output tokens at a fixed $/GPU-hour. "
    "Uses the same Welch interval as compare-runs: FAIL when the whole interval for the "
    "increase is above --max-increase-pct, PASS when it is all at or below, else INCONCLUSIVE.",
)
def cost_compare_command(
    baseline: Path = typer.Option(..., "--baseline"),
    candidate: Path = typer.Option(..., "--candidate"),
    gpu: str = typer.Option("H100-SXM", "--gpu"),
    gpus: int = typer.Option(1, "--gpus"),
    price: float | None = typer.Option(None, "--price", help="$/GPU-hour you pay."),
    max_increase_pct: float = typer.Option(5.0, "--max-increase-pct"),
    confidence: float = typer.Option(0.9, "--confidence"),
    min_repetitions: int = typer.Option(3, "--min-repetitions"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    import json as _json

    from inference_doctor import cost as costmod

    try:
        hourly = (price if price is not None else costmod.list_price(gpu)[0]) * gpus

        def per_m(d: Path) -> list[float]:
            raw = [r.result.metrics.output_token_throughput for r in load_directory(d).records]
            vals = [v for v in raw if v]
            if len(vals) < min_repetitions:
                raise ValueError(
                    f"'{d}' has {len(vals)} runs with output token throughput; "
                    f"need {min_repetitions}"
                )
            return [hourly / (v * 3600 / 1e6) for v in vals]

        gate = costmod.cost_gate(per_m(baseline), per_m(candidate), max_increase_pct, confidence)
    except (ValueError, OSError, BenchmarkResultError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    if as_json:
        typer.echo(_json.dumps({**gate.to_dict(), "hourly_usd": hourly}, indent=2))
    else:
        typer.echo(
            f"{gate.result}: $/1M output tokens {gate.baseline_mean:.4f} -> "
            f"{gate.candidate_mean:.4f} ({gate.change_percent:+.1f}%, "
            f"{gate.confidence:.0%} interval {gate.change_percent_low:+.1f}% to "
            f"{gate.change_percent_high:+.1f}%; limit +{max_increase_pct:g}%) "
            f"at ${hourly:.2f}/h"
        )
    if gate.result == "FAIL":
        raise typer.Exit(EXIT_FAIL)
    if gate.result == "INCONCLUSIVE":
        raise typer.Exit(EXIT_NO_DECISION)
