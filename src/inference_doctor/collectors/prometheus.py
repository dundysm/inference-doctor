from __future__ import annotations

import math
import re

import httpx

from inference_doctor.models import DiagnosticSnapshot


WINDOW_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[smhd])$")


def parse_window(window: str) -> int:
    match = WINDOW_RE.match(window)

    if not match:
        raise ValueError(
            f"Invalid window '{window}'. "
            "Expected values like 30s, 5m, 1h."
        )

    value = int(match.group("value"))
    unit = match.group("unit")

    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
    }

    return value * multipliers[unit]


class PrometheusClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def query(self, promql: str) -> float | None:
        response = self.client.get(
            f"{self.base_url}/api/v1/query",
            params={"query": promql},
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("status") != "success":
            return None

        results = payload.get("data", {}).get("result", [])

        if not results:
            return None

        values: list[float] = []

        for result in results:
            value = result.get("value")

            if not value or len(value) < 2:
                continue

            try:
                scalar = float(value[1])
            except (TypeError, ValueError):
                continue

            if math.isfinite(scalar):
                values.append(scalar)

        if not values:
            return None

        return sum(values)

    def close(self) -> None:
        self.client.close()


def histogram_p95(metric: str, window: str) -> str:
    return f"""
histogram_quantile(
  0.95,
  sum by (le) (
    rate({metric}_bucket[{window}])
  )
)
""".strip()


def build_snapshot(
    prometheus_url: str,
    window: str,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float | None,
) -> DiagnosticSnapshot:
    client = PrometheusClient(prometheus_url)

    try:
        window_seconds = parse_window(window)

        ttft_p95 = client.query(
            histogram_p95(
                "vllm:time_to_first_token_seconds",
                window,
            )
        )

        tpot_p95 = client.query(
            histogram_p95(
                "vllm:inter_token_latency_seconds",
                window,
            )
        )

        queue_p95 = client.query(
            histogram_p95(
                "vllm:request_queue_time_seconds",
                window,
            )
        )

        queue_mean = client.query(
            f"""
rate(vllm:request_queue_time_seconds_sum[{window}])
/
rate(vllm:request_queue_time_seconds_count[{window}])
""".strip()
        )

        prefill_time_p95 = client.query(
            histogram_p95(
                "vllm:request_prefill_time_seconds",
                window,
            )
        )

        prompt_tokens_p95 = client.query(
            histogram_p95(
                "vllm:request_prompt_tokens",
                window,
            )
        )

        requests_running = client.query(
            "sum(vllm:num_requests_running)"
        )

        requests_waiting = client.query(
            "sum(vllm:num_requests_waiting)"
        )

        kv_cache_usage = client.query(
            "max(vllm:kv_cache_usage_perc)"
        )

        preemptions_per_second = client.query(
            f"""
sum(
  rate(
    vllm:num_preemptions_total[{window}]
  )
)
""".strip()
        )

        preemptions_per_min = (
            preemptions_per_second * 60
            if preemptions_per_second is not None
            else None
        )

        return DiagnosticSnapshot(
            window_seconds=window_seconds,
            ttft_p95=ttft_p95,
            tpot_p95=tpot_p95,
            queue_time_p95=queue_p95,
            queue_mean=queue_mean,
            prefill_time_p95=prefill_time_p95,
            prompt_tokens_p95=prompt_tokens_p95,
            requests_running=requests_running,
            requests_waiting=requests_waiting,
            kv_cache_usage=kv_cache_usage,
            preemptions_per_min=preemptions_per_min,
            ttft_slo=(
                ttft_slo_ms / 1000
                if ttft_slo_ms is not None
                else None
            ),
            tpot_slo=(
                tpot_slo_ms / 1000
                if tpot_slo_ms is not None
                else None
            ),
        )

    finally:
        client.close()
