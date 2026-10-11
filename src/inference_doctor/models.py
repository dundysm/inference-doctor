from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DiagnosticSnapshot:
    window_seconds: int

    ttft_p95: float | None = None
    tpot_p95: float | None = None
    queue_time_p95: float | None = None
    queue_mean: float | None = None
    prefill_time_p95: float | None = None
    prompt_tokens_p95: float | None = None

    requests_running: float | None = None
    requests_waiting: float | None = None

    kv_cache_usage: float | None = None
    preemptions_per_min: float | None = None

    ttft_slo: float | None = None
    tpot_slo: float | None = None


@dataclass
class Evidence:
    metric: str
    observed: float | None
    expected: float | None = None
    unit: str | None = None


@dataclass
class Finding:
    id: str
    severity: Severity
    confidence: Confidence

    title: str
    explanation: str

    evidence: list[Evidence] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)

    confidence_score: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["severity"] = self.severity.value
        result["confidence"] = self.confidence.value
        return result
