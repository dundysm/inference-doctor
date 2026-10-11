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


# Queueing is "meaningful" when the mean time a request waits for admission is at least
# this share of TTFT p95. The mean comes from the histogram's _sum/_count, so it is not
# distorted by coarse buckets the way queue p95 is (see docs/experiment-log.md).
QUEUE_SHARE_OF_TTFT = 0.2
# Prefill explains TTFT when prefill p95 is at least this share of TTFT p95.
PREFILL_SHARE_OF_TTFT = 0.5


def _ttft_over_slo(snapshot: DiagnosticSnapshot) -> bool:
    return (
        snapshot.ttft_p95 is not None
        and snapshot.ttft_slo is not None
        and snapshot.ttft_p95 > snapshot.ttft_slo
    )


def _queue_evidence(snapshot: DiagnosticSnapshot) -> list[Evidence]:
    evidence: list[Evidence] = []
    if snapshot.requests_waiting is not None and snapshot.requests_waiting > 0:
        evidence.append(
            Evidence(metric="requests_waiting", observed=snapshot.requests_waiting, unit="requests")
        )
    if (
        snapshot.queue_mean is not None
        and snapshot.ttft_p95 is not None
        and snapshot.queue_mean >= QUEUE_SHARE_OF_TTFT * snapshot.ttft_p95
    ):
        evidence.append(
            Evidence(
                metric="queue_mean",
                observed=snapshot.queue_mean,
                expected=QUEUE_SHARE_OF_TTFT * snapshot.ttft_p95,
                unit="seconds",
            )
        )
    return evidence


def diagnose_queue_pressure(
    snapshot: DiagnosticSnapshot,
) -> Finding | None:
    # TTFT over its SLO is the symptom; queueing must be observed directly, through
    # waiting requests or a substantial mean queue time. A healthy TPOT only raises
    # confidence, it is not evidence of a queue.
    if not _ttft_over_slo(snapshot):
        return None
    queue_evidence = _queue_evidence(snapshot)
    if not queue_evidence:
        return None
    assert snapshot.ttft_p95 is not None

    score = 2 + 2 * len(queue_evidence)
    evidence: list[Evidence] = [
        Evidence(
            metric="ttft_p95",
            observed=snapshot.ttft_p95,
            expected=snapshot.ttft_slo,
            unit="seconds",
        ),
        *queue_evidence,
    ]

    if snapshot.kv_cache_usage is not None and snapshot.kv_cache_usage >= 0.85:
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
        score += 1
        evidence.append(
            Evidence(
                metric="tpot_p95",
                observed=snapshot.tpot_p95,
                expected=snapshot.tpot_slo,
                unit="seconds",
            )
        )

    severity = Severity.CRITICAL if score >= 7 else Severity.WARNING

    return Finding(
        id="QUEUE_PRESSURE",
        severity=severity,
        confidence=confidence_from_score(score),
        confidence_score=score,
        title="Queue / capacity pressure",
        explanation=(
            "Time to first token is over its SLO and requests are waiting for admission: "
            "the server has more work than it can start. Decode performance may still be "
            "healthy."
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


def diagnose_prefill_pressure(
    snapshot: DiagnosticSnapshot,
) -> Finding | None:
    # TTFT over its SLO with no sign of queueing, where prefill time accounts for most
    # of TTFT: long prompts, not admission, are the likely cause.
    if not _ttft_over_slo(snapshot) or _queue_evidence(snapshot):
        return None
    if snapshot.prefill_time_p95 is None or snapshot.ttft_p95 is None:
        return None
    if snapshot.prefill_time_p95 < PREFILL_SHARE_OF_TTFT * snapshot.ttft_p95:
        return None

    score = 4
    evidence = [
        Evidence(
            metric="ttft_p95",
            observed=snapshot.ttft_p95,
            expected=snapshot.ttft_slo,
            unit="seconds",
        ),
        Evidence(
            metric="prefill_time_p95",
            observed=snapshot.prefill_time_p95,
            expected=PREFILL_SHARE_OF_TTFT * snapshot.ttft_p95,
            unit="seconds",
        ),
    ]
    if snapshot.requests_waiting is not None:
        evidence.append(
            Evidence(metric="requests_waiting", observed=snapshot.requests_waiting, unit="requests")
        )
        score += 1
    if snapshot.prompt_tokens_p95 is not None:
        evidence.append(
            Evidence(metric="prompt_tokens_p95", observed=snapshot.prompt_tokens_p95, unit="tokens")
        )

    return Finding(
        id="PREFILL_PRESSURE",
        severity=Severity.WARNING,
        confidence=confidence_from_score(score),
        confidence_score=score,
        title="Prefill-bound time to first token",
        explanation=(
            "Time to first token is over its SLO, no requests are waiting, and prefill "
            "accounts for most of it. Long prompts, not request admission, are the likely "
            "cause; adding capacity may not help."
        ),
        evidence=evidence,
        experiments=[
            (
                "Replay the workload with shorter prompts, or the same prompts with prefix "
                "caching enabled, and compare TTFT and prefill time."
            ),
            "Compare chunked-prefill settings while holding the workload constant.",
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
    if snapshot.kv_cache_usage is None or snapshot.kv_cache_usage < KV_PRESSURE_THRESHOLD:
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

    if snapshot.requests_waiting is not None and snapshot.requests_waiting > 0:
        pressure_signal = True
        score += 2
        evidence.append(
            Evidence(
                metric="requests_waiting",
                observed=snapshot.requests_waiting,
                unit="requests",
            )
        )

    if snapshot.preemptions_per_min is not None and snapshot.preemptions_per_min > 0:
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
        severity=(Severity.CRITICAL if score >= 7 else Severity.WARNING),
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
            ("Then test increased KV-cache capacity while holding the workload constant."),
        ],
    )


def diagnose(
    snapshot: DiagnosticSnapshot,
) -> list[Finding]:
    rules = [
        diagnose_queue_pressure,
        diagnose_prefill_pressure,
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
