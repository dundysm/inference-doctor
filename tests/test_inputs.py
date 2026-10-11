"""Input discovery and the reasons compare-runs gives when it cannot decide."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_doctor.benchmark_results import BenchmarkResultError, parse_normalized_benchmark
from inference_doctor.quality import (
    QualityInputError,
    discover_run_paths,
    load_and_compare_directories,
)


def _payload(**metrics) -> dict:
    return {
        "schema_version": "1",
        "environment": {"gpu": "g", "model": "m"},
        "benchmark_parameters": {"c": 1},
        "metrics": {"requests": {"successful": 100, "failed": 0}, **metrics},
    }


def _write_side(directory: Path, values, name="repetition-{:03d}.json") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(values, start=1):
        payload = _payload(output_token_throughput=value)
        (directory / name.format(index)).write_text(json.dumps(payload), encoding="utf-8")


def test_discovery_error_names_the_patterns_and_the_ignored_files(tmp_path):
    _write_side(tmp_path, [1000, 1001, 1002, 1003], name="run{}.json")
    with pytest.raises(QualityInputError) as excinfo:
        discover_run_paths(tmp_path)
    message = str(excinfo.value)
    assert "repetition-*.json" in message and "*.normalized.json" in message
    assert "Ignored 4 other JSON file(s): run1.json, run2.json, run3.json and 1 more" in message


def test_discovery_skips_aggregates_and_summaries(tmp_path):
    _write_side(tmp_path, [1000, 1001])
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "aggregate.normalized.json").write_text("{}", encoding="utf-8")
    assert [path.name for path in discover_run_paths(tmp_path)] == [
        "repetition-001.json",
        "repetition-002.json",
    ]


def test_all_zero_failed_requests_is_stable_not_inconclusive(tmp_path):
    _write_side(tmp_path / "b", [1000, 1001, 1002])
    _write_side(tmp_path / "c", [1000, 1001, 1002])
    report = load_and_compare_directories(tmp_path / "b", tmp_path / "c", metric="requests_failed")
    assert report["baseline"]["cv_percent"] == 0.0
    assert report["result"] == "PASS"


def test_missing_metric_says_missing_not_unstable(tmp_path):
    _write_side(tmp_path / "b", [1000, 1001, 1002])
    _write_side(tmp_path / "c", [1000, 1001, 1002])
    report = load_and_compare_directories(tmp_path / "b", tmp_path / "c", metric="ttft_p95_ms")
    assert report["result"] == "INCONCLUSIVE"
    assert report["reasons"] == [
        "TTFT p95 was missing or non-finite in 3 of 3 baseline repetition(s)",
        "TTFT p95 was missing or non-finite in 3 of 3 candidate repetition(s)",
    ]


def test_single_repetition_says_too_few_not_unstable(tmp_path):
    _write_side(tmp_path / "b", [1000])
    _write_side(tmp_path / "c", [1000, 1001])
    report = load_and_compare_directories(tmp_path / "b", tmp_path / "c")
    assert report["result"] == "INCONCLUSIVE"
    assert report["reasons"] == [
        "baseline has 1 repetition; at least two are required to measure stability"
    ]


def test_unstable_reason_states_the_measured_cv(tmp_path):
    _write_side(tmp_path / "b", [1000, 1300, 1000])
    _write_side(tmp_path / "c", [1000, 1001, 1002])
    report = load_and_compare_directories(tmp_path / "b", tmp_path / "c")
    assert report["reasons"] == ["baseline CV 15.75% exceeded the 5% stability threshold"]


def test_non_finite_metric_is_reported_as_missing_or_non_finite():
    payload = _payload(output_token_throughput=float("nan"))
    assert parse_normalized_benchmark(payload).metrics.output_token_throughput is None


def test_integer_schema_version_gets_a_specific_message():
    payload = _payload(output_token_throughput=1)
    payload["schema_version"] = 1
    with pytest.raises(BenchmarkResultError, match='string "1", not the number 1'):
        parse_normalized_benchmark(payload)
