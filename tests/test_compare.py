import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inference_doctor.benchmark_results import (
    BenchmarkResultError,
    IncomparableBenchmarkError,
    ensure_comparable,
    load_normalized_benchmark,
    parse_normalized_benchmark,
)
from inference_doctor.cli import app
from inference_doctor.comparison import (
    ComparisonStatus,
    ComparisonThresholds,
    compare_benchmarks,
)


FIXTURES = Path(__file__).parent / "fixtures" / "compare"
BASELINE = FIXTURES / "baseline.json"
CANDIDATE = FIXTURES / "candidate_28pct_regression.json"
runner = CliRunner()


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_obvious_throughput_regression_fails():
    baseline = load_normalized_benchmark(BASELINE)
    candidate = load_normalized_benchmark(CANDIDATE)

    ensure_comparable(baseline, candidate)
    result = compare_benchmarks(
        baseline,
        candidate,
        ComparisonThresholds(),
    )
    output_throughput = next(
        metric
        for metric in result.metrics
        if metric.metric == "output_token_throughput"
    )

    assert result.overall_status == ComparisonStatus.FAIL
    assert output_throughput.absolute_delta == -280.0
    assert output_throughput.percentage_delta == pytest.approx(-28.0)
    assert output_throughput.status == ComparisonStatus.FAIL


def test_warn_band_does_not_fail_overall():
    baseline_payload = _payload(BASELINE)
    candidate_payload = copy.deepcopy(baseline_payload)
    candidate_payload["metrics"]["ttft_ms"]["p95"] = 64.0

    result = compare_benchmarks(
        parse_normalized_benchmark(baseline_payload),
        parse_normalized_benchmark(candidate_payload),
        ComparisonThresholds(),
    )
    ttft_p95 = next(
        metric
        for metric in result.metrics
        if metric.metric == "ttft_p95_ms"
    )

    assert ttft_p95.percentage_delta == pytest.approx(6.6666667)
    assert ttft_p95.status == ComparisonStatus.WARN
    assert result.overall_status == ComparisonStatus.PASS


@pytest.mark.parametrize(
    ("metric_path", "baseline_value", "candidate_value"),
    [
        (("ttft_ms", "mean"), 100.0, 115.0),
        (("ttft_ms", "mean"), 100.0, 85.0),
        (("output_token_throughput",), 100.0, 115.0),
        (("output_token_throughput",), 100.0, 85.0),
    ],
)
def test_same_version_large_variation_is_unstable(
    metric_path,
    baseline_value,
    candidate_value,
):
    baseline_payload = _payload(BASELINE)
    candidate_payload = copy.deepcopy(baseline_payload)
    baseline_target = baseline_payload["metrics"]
    candidate_target = candidate_payload["metrics"]
    for key in metric_path[:-1]:
        baseline_target = baseline_target[key]
        candidate_target = candidate_target[key]
    baseline_target[metric_path[-1]] = baseline_value
    candidate_target[metric_path[-1]] = candidate_value

    result = compare_benchmarks(
        parse_normalized_benchmark(baseline_payload),
        parse_normalized_benchmark(candidate_payload),
        ComparisonThresholds(stability_tolerance_percent=5.0),
        same_version=True,
    )

    metric = next(item for item in result.metrics if item.metric in {
        "ttft_mean_ms",
        "output_token_throughput",
    } and (
        item.baseline == baseline_value
        and item.candidate == candidate_value
    ))
    assert metric.status == ComparisonStatus.UNSTABLE
    assert result.overall_status == ComparisonStatus.UNSTABLE


def test_same_version_small_variation_is_stable():
    baseline_payload = _payload(BASELINE)
    candidate_payload = copy.deepcopy(baseline_payload)
    candidate_payload["metrics"]["ttft_ms"]["mean"] *= 1.01

    result = compare_benchmarks(
        parse_normalized_benchmark(baseline_payload),
        parse_normalized_benchmark(candidate_payload),
        ComparisonThresholds(stability_tolerance_percent=5.0),
        same_version=True,
    )

    metric = next(
        item for item in result.metrics if item.metric == "ttft_mean_ms"
    )
    assert metric.status == ComparisonStatus.PASS
    assert result.overall_status == ComparisonStatus.PASS


@pytest.mark.parametrize(
    ("metric", "baseline_value", "candidate_value", "expected"),
    [
        ("ttft_mean_ms", 100.0, 85.0, ComparisonStatus.PASS),
        ("ttft_mean_ms", 100.0, 115.0, ComparisonStatus.FAIL),
        (
            "output_token_throughput",
            100.0,
            115.0,
            ComparisonStatus.PASS,
        ),
        (
            "output_token_throughput",
            100.0,
            85.0,
            ComparisonStatus.FAIL,
        ),
    ],
)
def test_cross_version_directional_semantics_are_unchanged(
    metric,
    baseline_value,
    candidate_value,
    expected,
):
    baseline_payload = _payload(BASELINE)
    candidate_payload = copy.deepcopy(baseline_payload)
    if metric == "ttft_mean_ms":
        baseline_payload["metrics"]["ttft_ms"]["mean"] = baseline_value
        candidate_payload["metrics"]["ttft_ms"]["mean"] = candidate_value
    else:
        baseline_payload["metrics"][metric] = baseline_value
        candidate_payload["metrics"][metric] = candidate_value

    result = compare_benchmarks(
        parse_normalized_benchmark(baseline_payload),
        parse_normalized_benchmark(candidate_payload),
        ComparisonThresholds(),
    )

    comparison = next(item for item in result.metrics if item.metric == metric)
    assert comparison.status == expected
    expected_overall = (
        expected
        if expected == ComparisonStatus.FAIL
        else ComparisonStatus.PASS
    )
    assert result.overall_status == expected_overall


def test_compare_cli_same_version_unstable_exits_nonzero(tmp_path):
    baseline_payload = _payload(BASELINE)
    candidate_payload = copy.deepcopy(baseline_payload)
    candidate_payload["metrics"]["ttft_ms"]["mean"] *= 1.15
    candidate_path = tmp_path / "candidate_same_version_unstable.json"
    candidate_path.write_text(
        json.dumps(candidate_payload),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            str(BASELINE),
            "--candidate",
            str(candidate_path),
            "--same-version",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["overall_status"] == "UNSTABLE"


def test_missing_and_nonfinite_metrics_are_warn_not_nan():
    baseline_payload = _payload(BASELINE)
    candidate_payload = copy.deepcopy(baseline_payload)
    candidate_payload["metrics"]["ttft_ms"]["mean"] = float("nan")
    candidate_payload["metrics"]["tpot_ms"]["median"] = None

    result = compare_benchmarks(
        parse_normalized_benchmark(baseline_payload),
        parse_normalized_benchmark(candidate_payload),
        ComparisonThresholds(),
    )
    metrics = {metric.metric: metric for metric in result.metrics}

    assert metrics["ttft_mean_ms"].candidate is None
    assert metrics["ttft_mean_ms"].status == ComparisonStatus.WARN
    assert metrics["tpot_median_ms"].candidate is None
    assert metrics["tpot_median_ms"].percentage_delta is None


def test_new_failed_request_fails_from_zero_baseline():
    baseline = load_normalized_benchmark(BASELINE)
    candidate = load_normalized_benchmark(CANDIDATE)
    result = compare_benchmarks(
        baseline,
        candidate,
        ComparisonThresholds(
            latency_fail_percent=100.0,
            throughput_fail_percent=100.0,
        ),
    )
    failed = next(
        metric
        for metric in result.metrics
        if metric.metric == "requests_failed"
    )

    assert failed.absolute_delta == 2.0
    assert failed.percentage_delta is None
    assert failed.status == ComparisonStatus.FAIL


def test_incomparable_benchmark_parameters_are_rejected():
    baseline_payload = _payload(BASELINE)
    candidate_payload = _payload(CANDIDATE)
    candidate_payload["benchmark_parameters"]["input_tokens"] = 1024

    with pytest.raises(
        IncomparableBenchmarkError,
        match="benchmark_parameters",
    ):
        ensure_comparable(
            parse_normalized_benchmark(baseline_payload),
            parse_normalized_benchmark(candidate_payload),
        )


def test_invalid_metric_type_is_rejected():
    payload = _payload(BASELINE)
    payload["metrics"]["request_throughput"] = "fast"

    with pytest.raises(BenchmarkResultError, match="request_throughput"):
        parse_normalized_benchmark(payload)


def test_compare_cli_json_reports_failure_and_exit_one():
    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            str(BASELINE),
            "--candidate",
            str(CANDIDATE),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["overall_status"] == "FAIL"
    assert payload["baseline"]["gpu"] == "NVIDIA L4"
    assert payload["candidate"]["vllm_version"] == "0.29.0"


def test_compare_cli_same_file_passes_with_exit_zero():
    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            str(BASELINE),
            "--candidate",
            str(BASELINE),
        ],
    )

    assert result.exit_code == 0
    assert "INFERENCE UPGRADE GUARD" in result.output
    assert "PASS" in result.output


def test_compare_cli_accepts_configurable_fail_thresholds():
    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            str(BASELINE),
            "--candidate",
            str(CANDIDATE),
            "--latency-fail-percent",
            "30",
            "--throughput-fail-percent",
            "30",
            "--max-failed-request-increase",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "INFERENCE UPGRADE GUARD  PASS" in result.output


def test_compare_cli_rejects_incomparable_runs(tmp_path):
    candidate_payload = _payload(CANDIDATE)
    candidate_payload["environment"]["gpu"] = "NVIDIA A10"
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate_payload),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            str(BASELINE),
            "--candidate",
            str(candidate_path),
        ],
    )

    assert result.exit_code == 2
    assert "incomparable" in result.output
