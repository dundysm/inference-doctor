from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table


def _number(value: float | None, suffix: str = "") -> str:
    return "unavailable" if value is None else f"{value:.2f}{suffix}"


def _signed(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:+.2f}%"


STYLES = {"PASS": "green", "FAIL": "red", "INCONCLUSIVE": "yellow"}


def render_quality_report(report: dict, console: Console | None = None) -> None:
    console = console or Console()
    if "comparisons" in report:
        _render_multi(report, console)
        return
    status = report["result"]
    style = STYLES[status]
    console.print("[bold]INFERENCE CI[/bold]")
    console.print(f"RESULT: [{style}]{status}[/]")
    direction = "lower is better" if report["lower_is_better"] else "higher is better"
    console.print(f"Metric: {report['metric_label']} ({report['metric']}, {direction})")
    source = report.get("input")
    if source:
        formats = {source["baseline"]["format"], source["candidate"]["format"]}
        console.print(f"Input: {' / '.join(sorted(formats))} results")

    table = Table(box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_row(
        "repetitions", str(report["baseline"]["count"]), str(report["candidate"]["count"])
    )
    for key, label, suffix in (
        ("mean", "mean", ""),
        ("median", "median", ""),
        ("sample_standard_deviation", "sample SD", ""),
        ("cv_percent", "CV", "%"),
        ("min", "min", ""),
        ("max", "max", ""),
    ):
        table.add_row(
            label,
            _number(report["baseline"][key], suffix),
            _number(report["candidate"][key], suffix),
        )
    console.print(table)

    thresholds = report["thresholds"]
    interval = report["confidence_interval"]
    console.print(
        f"Delta: {_signed(report['delta_percent'])}  |  Regression: "
        f"{_signed(report['regression_percent'])} (positive is worse)"
    )
    if interval["available"]:
        console.print(
            f"{thresholds['confidence']:.0%} confidence interval for the regression: "
            f"{_signed(interval['regression_percent_low'])}.."
            f"{_signed(interval['regression_percent_high'])}"
        )
    ceiling = thresholds["stability_cv_percent"]
    console.print(
        f"Regression threshold: {thresholds['regression_percent']:g}%  |  "
        f"Minimum repetitions: {thresholds['min_repetitions']}"
        + (f"  |  CV ceiling: {ceiling:g}%" if ceiling is not None else "")
    )

    reasons = report["reasons"]
    if reasons:
        console.print("Why:")
        for reason in reasons:
            console.print(f"- {reason}")
    else:
        console.print(f"Why: {report['explanation']}.")
    for note in report["notes"]:
        console.print(f"Note: {note}.")
    if report["baseline"]["outlier_paths"] or report["candidate"]["outlier_paths"]:
        console.print("Outlier flags are descriptive only; all observations remain included.")
    if status == "INCONCLUSIVE":
        console.print("No merge recommendation was made.")


def _render_multi(report: dict, console: Console) -> None:
    status = report["result"]
    thresholds = report["thresholds"]
    console.print("[bold]INFERENCE CI[/bold]")
    console.print(f"RESULT: [{STYLES[status]}]{status}[/]")
    formats = {report["input"]["baseline"]["format"], report["input"]["candidate"]["format"]}
    console.print(f"Input: {' / '.join(sorted(formats))} results")

    table = Table(box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Runs", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column(f"Regression, {thresholds['confidence']:.0%} CI", justify="right")
    table.add_column("Result")
    for comparison in report["comparisons"]:
        interval = comparison["confidence_interval"]
        span = (
            f"{_signed(interval['regression_percent_low'])}.."
            f"{_signed(interval['regression_percent_high'])}"
            if interval["available"]
            else "unavailable"
        )
        result = comparison["result"]
        table.add_row(
            comparison["metric"],
            f"{comparison['baseline']['count']} / {comparison['candidate']['count']}",
            _number(comparison["baseline"]["mean"]),
            _number(comparison["candidate"]["mean"]),
            span,
            f"[{STYLES[result]}]{result}[/]",
        )
    console.print(table)
    console.print(
        "Columns show means. Regression is positive when worse; threshold "
        f"{thresholds['regression_percent']:g}%  |  "
        f"Minimum repetitions: {thresholds['min_repetitions']}"
    )
    if report["reasons"]:
        console.print("Why:")
        for reason in report["reasons"]:
            console.print(f"- {reason}")
    else:
        console.print("Why: every metric passed.")
    for note in report["notes"]:
        console.print(f"Note: {note}.")
    if status == "INCONCLUSIVE":
        console.print("No merge recommendation was made.")
