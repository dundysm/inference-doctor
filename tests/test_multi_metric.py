"""Gating on several metrics in one run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inference_doctor.cli import app
from inference_doctor.quality import (
    QualityInputError,
    combine_results,
    load_and_compare_directories,
)

runner = CliRunner()
EXAMPLE = Path(__file__).parent.parent / "examples" / "vllm-bench-serve"


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        (["PASS", "PASS"], "PASS"),
        (["PASS", "INCONCLUSIVE"], "INCONCLUSIVE"),
        (["INCONCLUSIVE", "FAIL"], "FAIL"),
        (["FAIL", "PASS"], "FAIL"),
    ],
)
def test_any_fail_fails_and_any_inconclusive_blocks_a_pass(results, expected):
    assert combine_results(results) == expected


def test_one_metric_keeps_the_single_comparison_shape():
    report = load_and_compare_directories(
        EXAMPLE / "baseline", EXAMPLE / "candidate", metric=["output_token_throughput"]
    )
    assert "comparisons" not in report
    assert report["metric"] == "output_token_throughput"
    assert report["result"] == "PASS"


def test_several_metrics_report_each_and_combine():
    report = load_and_compare_directories(
        EXAMPLE / "baseline",
        EXAMPLE / "candidate",
        metric=["output_token_throughput", "ttft_p99_ms", "output_token_throughput"],
    )
    assert report["metrics"] == ["output_token_throughput", "ttft_p99_ms"]  # duplicates dropped
    assert [c["result"] for c in report["comparisons"]] == ["PASS", "FAIL"]
    assert report["result"] == "FAIL"
    assert report["reasons"][0].startswith("ttft_p99_ms: the 90% confidence interval")
    assert any("2 metrics were checked" in note for note in report["notes"])
    assert report["input"]["baseline"]["format"] == "vllm"


def test_unknown_metric_in_the_list_is_an_input_error():
    with pytest.raises(QualityInputError, match="Unknown metric 'ttft_p42_ms'"):
        load_and_compare_directories(
            EXAMPLE / "baseline", EXAMPLE / "candidate", metric=["ttft_mean_ms", "ttft_p42_ms"]
        )


def test_cli_repeated_metric_exits_with_the_combined_result():
    args = ["compare-runs", "--baseline", str(EXAMPLE / "baseline")]
    args += ["--candidate", str(EXAMPLE / "candidate")]
    passing = runner.invoke(app, [*args, "--metric", "output_token_throughput"])
    assert passing.exit_code == 0
    both = runner.invoke(
        app, [*args, "--metric", "output_token_throughput", "--metric", "ttft_p99_ms", "--json"]
    )
    assert both.exit_code == 1
    report = json.loads(both.output)
    assert report["result"] == "FAIL"
    terminal = runner.invoke(
        app, [*args, "--metric", "output_token_throughput", "--metric", "ttft_p99_ms"]
    )
    assert "RESULT: FAIL" in terminal.output
    assert "ttft_p99_ms" in terminal.output
