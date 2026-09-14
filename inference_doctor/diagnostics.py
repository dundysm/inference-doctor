from __future__ import annotations

from inference_doctor.models import (
    Confidence,
    DiagnosticSnapshot,
    Evidence,
    Finding,
    Severity,
)


KV_PRESSURE_THRESHOLD = 0.90


def confidence_from_score(score: int) -> Confidence:
    if score >= 7:
        return Confidence.HIGH
    if score >= 4:
        return Confidence.MEDIUM
    return Confidence.LOW


def diagnose_queue_pressure(
    snapshot: DiagnosticSnapshot,
) -> Finding | None:
    if (
        snapshot.ttft_p95 is None
        or snapshot.ttft_slo is None
        or snapshot.ttft_p95 <= snapshot.ttft_slo
    ):
        return None

    score = 2
    evidence: list[Evidence] = [
        Evidence(
            metric="ttft_p95",
            observed=snapshot.ttft_p95,
            expected=snapshot.ttft_slo,
            unit="seconds",
        )
    ]

    if (
        snapshot.requests_waiting is not None
        and snapshot.requests_waiting > 0
    ):
        score += 2
        evidence.append(
            Evidence(
                metric="requests_waiting",
                observed=snapshot.requests_waiting,
                unit="requests",
            )
        )

    if (
        snapshot.kv_cache_usage is not None
        and snapshot.kv_cache_usage >= 0.85
    ):
        score += 1
        evidence.append(
            Evidence(
                metric="kv_cache_usage",
                observed=snapshot.kv_cache_usage,
                expected=0.85,
                unit="ratio",
            )
        )

    if (
        snapshot.tpot_p95 is not None
        and snapshot.tpot_slo is not None
        and snapshot.tpot_p95 <= snapshot.tpot_slo
    ):
        score += 2
        evidence.append(
            Evidence(
                metric="tpot_p95",
                observed=snapshot.tpot_p95,
                expected=snapshot.tpot_slo,
                unit="seconds",
            )
        )

    if score < 4:
        return None

    severity = (
        Severity.CRITICAL
        if score >= 7
        else Severity.WARNING
    )

    return Finding(
        id="QUEUE_PRESSURE",
        severity=severity,
        confidence=confidence_from_score(score),
        confidence_score=score,
        title="Queue / capacity pressure",
        explanation=(
            "Requests are experiencing elevated time-to-first-token "
            "latency together with evidence of scheduler or serving "
            "pressure. Decode performance may still be healthy, which "
            "suggests the bottleneck occurs before or during request "
            "admission/prefill."
        ),
        evidence=evidence,
        experiments=[
            (
                "Replay the same workload with roughly 20% more serving "
                "capacity and compare TTFT, queue time, TPOT, and throughput."
            ),
            (
                "Alternatively reduce request concurrency while keeping "
                "the model and serving configuration unchanged."
            ),
        ],
    )


def diagnose_decode_pressure(
    snapshot: DiagnosticSnapshot,
) -> Finding | None:
    if (
        snapshot.tpot_p95 is None
        or snapshot.tpot_slo is None
        or snapshot.tpot_p95 <= snapshot.tpot_slo
    ):
        return None

    score = 3

    evidence = [
        Evidence(
            metric="tpot_p95",
            observed=snapshot.tpot_p95,
            expected=snapshot.tpot_slo,
            unit="seconds",
        )
    ]

    if (
        snapshot.ttft_p95 is not None
        and snapshot.ttft_slo is not None
        and snapshot.ttft_p95 <= snapshot.ttft_slo
    ):
        score += 3
        evidence.append(
            Evidence(
                metric="ttft_p95",
                observed=snapshot.ttft_p95,
                expected=snapshot.ttft_slo,
                unit="seconds",
            )
        )

    return Finding(
        id="DECODE_PRESSURE",
        severity=Severity.WARNING,
        confidence=confidence_from_score(score),
        confidence_score=score,
        title="Decode-path pressure",
        explanation=(
            "Time-to-first-token appears healthy while inter-token "
            "latency violates the configured SLO. This pattern points "
            "toward token-generation/decode performance rather than "
            "request admission or queueing."
        ),
        evidence=evidence,
        experiments=[
            (
                "Replay an identical workload at lower concurrency and "
                "measure whether TPOT improves."
            ),
            (
                "Benchmark tensor-parallel configuration, quantization, "
                "and batch/concurrency settings independently."
            ),
        ],
    )


def diagnose_kv_pressure(
    snapshot: DiagnosticSnapshot,
) -> Finding | None:
    if (
        snapshot.kv_cache_usage is None
        or snapshot.kv_cache_usage < KV_PRESSURE_THRESHOLD
    ):
        return None

    pressure_signal = False
    score = 2

    evidence = [
        Evidence(
            metric="kv_cache_usage",
            observed=snapshot.kv_cache_usage,
            expected=KV_PRESSURE_THRESHOLD,
            unit="ratio",
        )
    ]

    if (
        snapshot.requests_waiting is not None
        and snapshot.requests_waiting > 0
    ):
        pressure_signal = True
        score += 2
        evidence.append(
            Evidence(
                metric="requests_waiting",
                observed=snapshot.requests_waiting,
                unit="requests",
            )
        )

    if (
        snapshot.preemptions_per_min is not None
        and snapshot.preemptions_per_min > 0
    ):
        pressure_signal = True
        score += 3
        evidence.append(
            Evidence(
                metric="preemptions_per_min",
                observed=snapshot.preemptions_per_min,
                unit="preemptions/min",
            )
        )

    if (
        snapshot.ttft_p95 is not None
        and snapshot.ttft_slo is not None
        and snapshot.ttft_p95 > snapshot.ttft_slo
    ):
        pressure_signal = True
        score += 1

    if not pressure_signal:
        return None

    return Finding(
        id="KV_CACHE_PRESSURE",
        severity=(
            Severity.CRITICAL
            if score >= 7
            else Severity.WARNING
        ),
        confidence=confidence_from_score(score),
        confidence_score=score,
        title="KV-cache pressure",
        explanation=(
            "High KV-cache utilization is occurring together with "
            "scheduler pressure, preemption, or degraded request latency. "
            "High cache utilization alone is not considered a problem."
        ),
        evidence=evidence,
        experiments=[
            (
                "Replay the workload with lower sequence concurrency and "
                "compare preemptions, queue depth, and TTFT."
            ),
            (
                "Then test increased KV-cache capacity while holding the "
                "workload constant."
            ),
        ],
    )


def diagnose(
    snapshot: DiagnosticSnapshot,
) -> list[Finding]:
    rules = [
        diagnose_queue_pressure,
        diagnose_decode_pressure,
        diagnose_kv_pressure,
    ]

    findings: list[Finding] = []

    for rule in rules:
        finding = rule(snapshot)
        if finding is not None:
            findings.append(finding)

    severity_order = {
        Severity.CRITICAL: 0,
        Severity.WARNING: 1,
        Severity.INFO: 2,
    }

    return sorted(
        findings,
        key=lambda finding: (
            severity_order[finding.severity],
            -finding.confidence_score,
        ),
    )
