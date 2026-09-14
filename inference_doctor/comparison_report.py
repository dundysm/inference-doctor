from __future__ import annotations

from rich.console import Console
from rich.table import Table

from inference_doctor.comparison import (
    BenchmarkComparison,
    ComparisonStatus,
    MetricComparison,
)


console = Console()


def _format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "N/A"
    if unit == "requests":
        return f"{value:.0f}"
    return f"{value:.2f}"


def _format_delta(metric: MetricComparison) -> str:
    if metric.absolute_delta is None:
        return "N/A"
    return f"{metric.absolute_delta:+.2f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def render_comparison(comparison: BenchmarkComparison) -> None:
    status_style = {
        ComparisonStatus.PASS: "green",
        ComparisonStatus.WARN: "yellow",
        ComparisonStatus.FAIL: "red",
        ComparisonStatus.UNSTABLE: "magenta",
    }

    baseline = comparison.baseline_metadata
    candidate = comparison.candidate_metadata
    console.print(
        "[bold]INFERENCE UPGRADE GUARD[/bold]  "
        f"[{status_style[comparison.overall_status]}]"
        f"{comparison.overall_status.value}[/]"
    )
    console.print(
        f"GPU: {baseline['gpu']}  |  Model: {baseline['model']}"
    )
    console.print(
        "vLLM: "
        f"{baseline['vllm_version'] or 'N/A'} -> "
        f"{candidate['vllm_version'] or 'N/A'}"
    )
    console.print(
        "Image: "
        f"{baseline['image'] or 'N/A'} -> "
        f"{candidate['image'] or 'N/A'}"
    )

    table = Table(show_lines=False)
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Delta %", justify="right")
    table.add_column("Status", justify="center")

    for metric in comparison.metrics:
        style = status_style[metric.status]
        table.add_row(
            metric.label,
            _format_value(metric.baseline, metric.unit),
            _format_value(metric.candidate, metric.unit),
            _format_delta(metric),
            _format_percent(metric.percentage_delta),
            f"[{style}]{metric.status.value}[/{style}]",
        )

    console.print(table)
