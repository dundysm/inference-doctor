from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from inference_doctor.models import (
    DiagnosticSnapshot,
    Finding,
    Severity,
)


console = Console()


def milliseconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 1000:.1f} ms"


def render_summary(
    snapshot: DiagnosticSnapshot,
    findings: list[Finding],
) -> None:
    critical = sum(
        finding.severity == Severity.CRITICAL
        for finding in findings
    )
    warning = sum(
        finding.severity == Severity.WARNING
        for finding in findings
    )

    table = Table(show_header=False)
    table.add_row("TTFT p95", milliseconds(snapshot.ttft_p95))
    table.add_row("TPOT p95", milliseconds(snapshot.tpot_p95))
    table.add_row("Queue p95", milliseconds(snapshot.queue_time_p95))
    table.add_row("Queue mean", milliseconds(snapshot.queue_mean))
    table.add_row(
        "Prefill p95",
        milliseconds(snapshot.prefill_time_p95),
    )
    table.add_row(
        "Prompt tokens p95",
        (
            f"{snapshot.prompt_tokens_p95:.1f} tokens"
            if snapshot.prompt_tokens_p95 is not None
            else "N/A"
        ),
    )
    table.add_row(
        "Waiting requests",
        (
            f"{snapshot.requests_waiting:.0f}"
            if snapshot.requests_waiting is not None
            else "N/A"
        ),
    )
    table.add_row(
        "KV cache",
        (
            f"{snapshot.kv_cache_usage * 100:.1f}%"
            if snapshot.kv_cache_usage is not None
            else "N/A"
        ),
    )
    table.add_row(
        "Preemptions/min",
        (
            f"{snapshot.preemptions_per_min:.2f}"
            if snapshot.preemptions_per_min is not None
            else "N/A"
        ),
    )

    console.print(
        Panel(
            table,
            title="INFERENCE DOCTOR v0.2",
            subtitle=f"{critical} critical · {warning} warnings",
        )
    )


def render_finding(
    finding: Finding,
    number: int,
) -> None:
    severity_icon = {
        Severity.CRITICAL: "🔴",
        Severity.WARNING: "🟠",
        Severity.INFO: "ℹ",
    }[finding.severity]

    console.print()
    console.print(
        f"[bold]{severity_icon} P{number} — "
        f"{finding.title.upper()}[/bold]"
    )
    console.print(
        f"Confidence: "
        f"[bold]{finding.confidence.value.upper()}[/bold]"
    )

    console.print()
    console.print("[bold]Evidence[/bold]")

    for evidence in finding.evidence:
        observed = evidence.observed

        if evidence.unit == "seconds" and observed is not None:
            observed_display = f"{observed * 1000:.1f} ms"
        elif evidence.unit == "ratio" and observed is not None:
            observed_display = f"{observed * 100:.1f}%"
        elif observed is not None:
            observed_display = f"{observed:.2f}"
        else:
            observed_display = "N/A"

        console.print(
            f"  • {evidence.metric}: {observed_display}"
        )

    console.print()
    console.print("[bold]Diagnosis[/bold]")
    console.print(finding.explanation)

    console.print()
    console.print("[bold]Next experiment[/bold]")

    for experiment in finding.experiments:
        console.print(f"  • {experiment}")


def render_report(
    snapshot: DiagnosticSnapshot,
    findings: list[Finding],
) -> None:
    render_summary(snapshot, findings)

    if not findings:
        console.print()
        console.print(
            "[green]No diagnostic conditions were triggered.[/green]"
        )
        return

    for index, finding in enumerate(findings):
        render_finding(finding, index)
