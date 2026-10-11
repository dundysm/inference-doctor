from __future__ import annotations

from rich.console import Console
from rich.table import Table


def _number(value: float | None, suffix: str = "") -> str:
    return "unavailable" if value is None else f"{value:.2f}{suffix}"


def _signed(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:+.2f}%"


def render_quality_report(report: dict) -> None:
    console = Console()
    status = report["result"]
    style = {"PASS": "green", "FAIL": "red", "INCONCLUSIVE": "yellow"}[status]
    console.print("[bold]INFERENCE CI[/bold]")
    console.print(f"RESULT: [{style}]{status}[/]")
    direction = "lower is better" if report["lower_is_better"] else "higher is better"
    console.print(f"Metric: {report['metric_label']} ({report['metric']}, {direction})")

    table = Table(show_lines=False)
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
            f"{_signed(interval['regression_percent_low'])} to "
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
