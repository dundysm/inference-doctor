from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from inference_doctor.cli import app
from inference_doctor.diagnostics import diagnose
from inference_doctor.models import DiagnosticSnapshot
from inference_doctor.collectors.prometheus import (
    PrometheusClient,
    build_snapshot,
    histogram_p95,
    parse_window,
)


runner = CliRunner()


def test_queue_pressure():
    snapshot = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=0.812,
        ttft_slo=0.400,
        tpot_p95=0.042,
        tpot_slo=0.050,
        queue_time_p95=0.388,
        requests_waiting=18,
        kv_cache_usage=0.91,
        preemptions_per_min=0,
    )

    findings = diagnose(snapshot)
    ids = {finding.id for finding in findings}

    assert "QUEUE_PRESSURE" in ids
    assert "DECODE_PRESSURE" not in ids


def test_decode_pressure():
    snapshot = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=0.220,
        ttft_slo=0.400,
        tpot_p95=0.091,
        tpot_slo=0.050,
        queue_time_p95=0.005,
        requests_waiting=0,
        kv_cache_usage=0.60,
        preemptions_per_min=0,
    )

    findings = diagnose(snapshot)
    ids = {finding.id for finding in findings}

    assert "DECODE_PRESSURE" in ids
    assert "QUEUE_PRESSURE" not in ids


def test_queue_p95_is_observational_only():
    low_queue = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=0.812,
        ttft_slo=0.400,
        tpot_p95=0.042,
        tpot_slo=0.050,
        queue_time_p95=0.001,
    )
    high_queue = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=0.812,
        ttft_slo=0.400,
        tpot_p95=0.042,
        tpot_slo=0.050,
        queue_time_p95=4.0,
    )

    low_queue_finding = diagnose(low_queue)[0]
    high_queue_finding = diagnose(high_queue)[0]

    assert low_queue_finding.id == "QUEUE_PRESSURE"
    assert high_queue_finding.id == "QUEUE_PRESSURE"
    assert low_queue_finding.confidence_score == 4
    assert high_queue_finding.confidence_score == 4
    assert all(
        evidence.metric != "queue_time_p95"
        for evidence in high_queue_finding.evidence
    )


def test_kv_pressure():
    snapshot = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=0.650,
        ttft_slo=0.400,
        tpot_p95=0.045,
        tpot_slo=0.050,
        queue_time_p95=0.150,
        requests_waiting=9,
        kv_cache_usage=0.97,
        preemptions_per_min=4.5,
    )

    findings = diagnose(snapshot)
    ids = {finding.id for finding in findings}

    assert "KV_CACHE_PRESSURE" in ids


def test_high_kv_usage_alone_is_not_a_problem():
    snapshot = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=0.180,
        ttft_slo=0.400,
        tpot_p95=0.030,
        tpot_slo=0.050,
        queue_time_p95=0.001,
        requests_waiting=0,
        kv_cache_usage=0.96,
        preemptions_per_min=0,
    )

    findings = diagnose(snapshot)
    ids = {finding.id for finding in findings}

    assert "KV_CACHE_PRESSURE" not in ids
    assert "QUEUE_PRESSURE" not in ids


def test_no_slos_does_not_invent_latency_problem():
    snapshot = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=1.5,
        tpot_p95=0.15,
        queue_time_p95=0.4,
        requests_waiting=0,
        kv_cache_usage=0.50,
        preemptions_per_min=0,
    )

    findings = diagnose(snapshot)
    ids = {finding.id for finding in findings}

    assert "QUEUE_PRESSURE" not in ids
    assert "DECODE_PRESSURE" not in ids


def test_parse_window_seconds():
    assert parse_window("30s") == 30
    assert parse_window("15m") == 900
    assert parse_window("1h") == 3600


def test_histogram_p95_uses_prometheus_bucket_series():
    promql = histogram_p95(
        "vllm:time_to_first_token_seconds",
        "15m",
    )

    assert "histogram_quantile" in promql
    assert "rate(vllm:time_to_first_token_seconds_bucket[15m])" in promql
    assert "sum by (le)" in promql


def test_prometheus_client_sums_vector_results():
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "status": "success",
                "data": {
                    "result": [
                        {"value": [0, "2.5"]},
                        {"value": [0, "3.5"]},
                    ]
                },
            }

    class FakeHttpClient:
        def get(self, url, params):
            assert url == "http://prometheus/api/v1/query"
            assert params == {"query": "sum(vllm:num_requests_waiting)"}
            return FakeResponse()

    client = PrometheusClient("http://prometheus")
    client.client = FakeHttpClient()

    assert client.query("sum(vllm:num_requests_waiting)") == 6.0


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-Inf"])
def test_prometheus_client_returns_none_for_non_finite_scalars(value):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "status": "success",
                "data": {"result": [{"value": [0, value]}]},
            }

    class FakeHttpClient:
        def get(self, url, params):
            return FakeResponse()

    client = PrometheusClient("http://prometheus")
    client.client = FakeHttpClient()

    assert client.query("some_metric") is None


def test_build_snapshot_queries_current_vllm_preemption_counter():
    queries = []

    class FakePrometheusClient:
        def __init__(self, prometheus_url):
            pass

        def query(self, promql):
            queries.append(promql)
            return 0.0

        def close(self):
            pass

    with patch(
        "inference_doctor.collectors.prometheus.PrometheusClient",
        FakePrometheusClient,
    ):
        snapshot = build_snapshot(
            prometheus_url="http://prometheus",
            window="15m",
            ttft_slo_ms=400,
            tpot_slo_ms=50,
        )

    assert any("vllm:num_preemptions_total[15m]" in query for query in queries)
    assert any(
        "vllm:request_queue_time_seconds_sum[15m]" in query
        and "vllm:request_queue_time_seconds_count[15m]" in query
        for query in queries
    )
    assert any(
        "vllm:request_prefill_time_seconds_bucket[15m]" in query
        for query in queries
    )
    assert any(
        "vllm:request_prompt_tokens_bucket[15m]" in query
        for query in queries
    )
    assert snapshot.queue_mean == 0.0
    assert snapshot.prefill_time_p95 == 0.0
    assert snapshot.prompt_tokens_p95 == 0.0


def test_cli_exposes_documented_diagnose_command():
    result = runner.invoke(app, ["diagnose", "--help"])

    assert result.exit_code == 0
    assert "--prometheus" in result.output


def test_cli_reports_snapshot_collection_errors():
    def fail_build_snapshot(**kwargs):
        raise ValueError("Invalid window '15x'. Expected values like 30s, 5m, 1h.")

    with patch("inference_doctor.cli.build_snapshot", fail_build_snapshot):
        result = runner.invoke(
            app,
            [
                "diagnose",
                "--prometheus",
                "http://prometheus",
                "--window",
                "15x",
            ],
        )

    assert result.exit_code == 1
    assert "Invalid window" in result.output
