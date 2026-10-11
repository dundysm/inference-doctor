from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from inference_doctor.cli import app
from inference_doctor.collectors.prometheus import (
    PrometheusClient,
    build_snapshot,
    histogram_p95,
    parse_window,
)
from inference_doctor.diagnostics import diagnose
from inference_doctor.models import DiagnosticSnapshot

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
    def snapshot(queue_p95):
        return DiagnosticSnapshot(
            window_seconds=900,
            ttft_p95=0.812,
            ttft_slo=0.400,
            tpot_p95=0.042,
            tpot_slo=0.050,
            queue_time_p95=queue_p95,
            requests_waiting=5,
        )

    low, high = diagnose(snapshot(0.001))[0], diagnose(snapshot(4.0))[0]
    assert low.id == high.id == "QUEUE_PRESSURE"
    assert low.confidence_score == high.confidence_score
    assert all(evidence.metric != "queue_time_p95" for evidence in high.evidence)


def test_high_ttft_with_healthy_tpot_alone_is_not_queue_pressure():
    # The pattern of a prefill-heavy workload: no waiting requests, no queue time.
    snapshot = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=0.9,
        ttft_slo=0.4,
        tpot_p95=0.01,
        tpot_slo=0.05,
        requests_waiting=0,
        queue_mean=0.0,
        queue_time_p95=0.0,
    )
    assert diagnose(snapshot) == []


def test_mean_queue_time_is_queue_evidence():
    snapshot = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=0.9,
        ttft_slo=0.4,
        requests_waiting=0,
        queue_mean=0.3,
    )
    finding = diagnose(snapshot)[0]
    assert finding.id == "QUEUE_PRESSURE"
    assert any(evidence.metric == "queue_mean" for evidence in finding.evidence)


def test_prefill_bound_ttft_is_reported_as_prefill_not_queue():
    snapshot = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=0.9,
        ttft_slo=0.4,
        tpot_p95=0.01,
        tpot_slo=0.05,
        requests_waiting=0,
        queue_mean=0.01,
        prefill_time_p95=0.7,
        prompt_tokens_p95=7800,
    )
    findings = diagnose(snapshot)
    assert [finding.id for finding in findings] == ["PREFILL_PRESSURE"]
    assert findings[0].severity.value == "warning"


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


def _client_returning(*values):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "status": "success",
                "data": {"result": [{"value": [0, value]} for value in values]},
            }

    class FakeHttpClient:
        def get(self, url, params):
            assert url == "http://prometheus/api/v1/query"
            return FakeResponse()

    client = PrometheusClient("http://prometheus")
    client.client = FakeHttpClient()
    return client


def test_prometheus_client_reads_a_single_series():
    assert _client_returning("2.5").query("sum(vllm:num_requests_waiting)") == 2.5


def test_prometheus_client_refuses_to_add_up_several_series():
    # Two replicas at 0.2 s each used to be reported as a 0.4 s queue mean.
    with pytest.raises(ValueError, match="returned 2 series"):
        _client_returning("0.2", "0.2").query("rate(x_sum[5m]) / rate(x_count[5m])")


def test_queue_mean_query_aggregates_across_replicas():
    queries = []

    class FakePrometheusClient:
        def __init__(self, prometheus_url):
            pass

        def query(self, promql):
            queries.append(promql)
            return None

        def close(self):
            pass

    with patch("inference_doctor.collectors.prometheus.PrometheusClient", FakePrometheusClient):
        build_snapshot(prometheus_url="http://p", window="5m", ttft_slo_ms=None, tpot_slo_ms=None)
    queue_mean = next(q for q in queries if "request_queue_time_seconds_sum" in q)
    assert "sum(rate(vllm:request_queue_time_seconds_sum[5m]))" in queue_mean
    assert "sum(rate(vllm:request_queue_time_seconds_count[5m]))" in queue_mean


def _patched_snapshot(snapshot):
    return patch("inference_doctor.cli.build_snapshot", lambda **kwargs: snapshot)


def test_diagnose_exits_1_on_a_critical_finding_and_respects_fail_on():
    critical = DiagnosticSnapshot(
        window_seconds=900,
        ttft_p95=0.9,
        ttft_slo=0.4,
        tpot_p95=0.01,
        tpot_slo=0.05,
        requests_waiting=12,
        queue_mean=0.5,
        kv_cache_usage=0.9,
    )
    args = ["diagnose", "--prometheus", "http://p", "--json"]
    with _patched_snapshot(critical):
        assert runner.invoke(app, args).exit_code == 1
        assert runner.invoke(app, [*args, "--fail-on", "never"]).exit_code == 0
        assert runner.invoke(app, [*args, "--fail-on", "sometimes"]).exit_code == 3


def test_diagnose_warning_exits_0_by_default_and_1_with_fail_on_warning():
    warning = DiagnosticSnapshot(
        window_seconds=900, ttft_p95=0.2, ttft_slo=0.4, tpot_p95=0.09, tpot_slo=0.05
    )
    args = ["diagnose", "--prometheus", "http://p"]
    with _patched_snapshot(warning):
        assert runner.invoke(app, args).exit_code == 0
        assert runner.invoke(app, [*args, "--fail-on", "warning"]).exit_code == 1


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
    assert any("vllm:request_prefill_time_seconds_bucket[15m]" in query for query in queries)
    assert any("vllm:request_prompt_tokens_bucket[15m]" in query for query in queries)
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

    assert result.exit_code == 3
    assert "Invalid window" in result.output
