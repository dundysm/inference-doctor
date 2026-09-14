from __future__ import annotations

from rich.console import Console
from rich.table import Table


def _number(value: float | None, suffix: str = "") -> str:
    return "unavailable" if value is None else f"{value:.2f}{suffix}"


def render_quality_report(report: dict) -> None:
    console = Console()
    status = report["result"]
    style = {"PASS": "green", "FAIL": "red", "INCONCLUSIVE": "yellow"}[status]
    console.print("[bold]INFERENCE CI[/bold]")
    console.print(f"RESULT: [{style}]{status}[/]")
    console.print(f"Metric: {report['metric_label']} ({report['metric']})")

    table = Table(show_lines=False)
    table.add_column("", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
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
    console.print(f"Delta: {_number(report['delta_percent'], '%')}")
    console.print(
        "Regression threshold: "
        f"{report['thresholds']['regression_percent']:g}%  |  "
        "Stability threshold: "
        f"{report['thresholds']['stability_cv_percent']:g}%"
    )

    reasons = report["reasons"]
    if reasons:
        console.print("Why:")
        for reason in reasons:
            console.print(f"- {reason}")
    else:
        console.print("Why: measurements were stable and within the regression threshold.")
    if report["baseline"]["outlier_paths"] or report["candidate"]["outlier_paths"]:
        console.print("Outlier flags are descriptive only; all observations remain included.")
    if status == "INCONCLUSIVE":
        console.print("No merge recommendation was made.")
