from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inference_doctor.cli import app
from inference_doctor.quality import (
    compare_repeated_runs,
)
from inference_doctor import telemetry


runner = CliRunner()


def _payload(value: float, *, gpu: str = "NVIDIA H100") -> dict:
    return {
        "schema_version": "1",
        "environment": {
            "gpu": gpu,
            "vllm_version": "0.19.0",
            "model": "BAAI/bge-m3",
            "image": "test-image",
        },
        "benchmark_parameters": {
            "concurrency": 32,
            "dataset_revision": "immutable-sha",
            "request_shape": "32 documents",
        },
        "metrics": {
            "texts_per_second": value,
            "output_token_throughput": value,
            "request_throughput": value / 32,
            "ttft_ms": {"mean": 100.0, "median": 90.0, "p95": 150.0},
            "tpot_ms": {"mean": 10.0, "median": 9.0, "p95": 15.0},
            "requests": {"successful": 100, "failed": 0},
        },
    }


def _runs(values: list[float], *, version: str = "0.19.0") -> list:
    runs = []
    for index, value in enumerate(values):
        payload = _payload(value)
        payload["environment"]["vllm_version"] = version
        path = Path(f"repetition-{index + 1:03d}.json")
        from inference_doctor.quality import RepeatedRun
        from inference_doctor.benchmark_results import parse_normalized_benchmark

        runs.append(RepeatedRun(path, parse_normalized_benchmark(payload), value))
    return runs


def test_stable_without_regression_is_pass():
    report = compare_repeated_runs(
        _runs([100, 101, 99, 100, 100, 100]),
        _runs([100, 101, 99, 100, 100, 100], version="0.21.0"),
        metric="texts_per_second",
    )
    assert report["result"] == "PASS"
    assert report["baseline"]["sample_standard_deviation"] is not None


def test_stable_ten_percent_throughput_loss_is_fail():
    report = compare_repeated_runs(
        _runs([100] * 6),
        _runs([88] * 6, version="0.21.0"),
        metric="texts_per_second",
    )
    assert report["result"] == "FAIL"
    assert report["delta_percent"] == pytest.approx(-12.0)


def test_unstable_baseline_is_inconclusive():
    report = compare_repeated_runs(
        _runs([100, 115, 100, 100, 100, 100]),
        _runs([100] * 6, version="0.21.0"),
        metric="texts_per_second",
    )
    assert report["result"] == "INCONCLUSIVE"
    assert "baseline measurements" in " ".join(report["reasons"])


def test_unstable_candidate_is_inconclusive():
    report = compare_repeated_runs(
        _runs([100] * 6),
        _runs([100, 115, 100, 100, 100, 100], version="0.21.0"),
        metric="texts_per_second",
    )
    assert report["result"] == "INCONCLUSIVE"
    assert "candidate measurements" in " ".join(report["reasons"])


def test_apparent_outlier_remains_included_in_official_calculation():
    report = compare_repeated_runs(
        _runs([100, 100, 100, 100, 100, 160]),
        _runs([100] * 6, version="0.21.0"),
        metric="texts_per_second",
    )
    assert report["baseline"]["values"][-1] == 160.0
    assert report["baseline"]["count"] == 6
    assert report["baseline"]["mean"] == pytest.approx(110.0)
    assert report["measurement_quality"]["outliers_included"] is True


def test_environment_mismatch_is_inconclusive():
    baseline = _runs([100] * 3)
    candidate = _runs([100] * 3, version="0.21.0")
    candidate[1] = candidate[1].__class__(
        candidate[1].path,
        candidate[1].result.__class__(
            candidate[1].result.schema_version,
            candidate[1].result.environment.__class__(
                "NVIDIA A10",
                candidate[1].result.environment.vllm_version,
                candidate[1].result.environment.model,
                candidate[1].result.environment.image,
            ),
            candidate[1].result.benchmark_parameters,
            candidate[1].result.metrics,
        ),
        candidate[1].value,
    )
    report = compare_repeated_runs(baseline, candidate, metric="texts_per_second")
    assert report["result"] == "INCONCLUSIVE"
    assert report["measurement_quality"]["environment_valid"] is False


def _write_runs(directory: Path, values: list[float], version: str) -> None:
    directory.mkdir(parents=True)
    for index, value in enumerate(values, start=1):
        payload = _payload(value)
        payload["environment"]["vllm_version"] = version
        (directory / f"repetition-{index:03d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def test_cli_exit_codes_and_json_match_terminal(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_runs(baseline, [100] * 3, "0.19.0")
    _write_runs(candidate, [88] * 3, "0.21.0")

    json_result = runner.invoke(
        app,
        [
            "compare-runs",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--metric",
            "texts_per_second",
            "--json",
        ],
    )
    terminal_result = runner.invoke(
        app,
        [
            "compare-runs",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--metric",
            "texts_per_second",
        ],
    )

    assert json_result.exit_code == 1
    assert terminal_result.exit_code == 1
    assert json.loads(json_result.output)["result"] == "FAIL"
    assert "RESULT: FAIL" in terminal_result.output


def test_cli_inconclusive_exit_code(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_runs(baseline, [100, 130, 100], "0.19.0")
    _write_runs(candidate, [100] * 3, "0.21.0")
    result = runner.invoke(
        app,
        [
            "compare-runs",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--metric",
            "texts_per_second",
        ],
    )
    assert result.exit_code == 2
    assert "RESULT: INCONCLUSIVE" in result.output


def test_telemetry_failure_is_recorded_as_unavailable(monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi unavailable")

    monkeypatch.setattr(telemetry.subprocess, "run", fail)
    sample = telemetry.sample_once()
    assert sample["available"] is False
    assert "unavailable" in sample["error"]
