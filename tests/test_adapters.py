"""Reading saved results from vLLM, AIPerf, GuideLLM and SGLang.

The record builders mirror the files each tool writes (field names checked against the
tools' source in October 2026), trimmed to the fields Inference Doctor reads plus a few
it must ignore.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inference_doctor.adapters import SourceOptions, detect_format, load_directory
from inference_doctor.benchmark_results import BenchmarkResultError
from inference_doctor.cli import app
from inference_doctor.quality import load_and_compare_directories

runner = CliRunner()


def vllm_record(throughput: float = 1341.8, **extra) -> dict:
    record = {
        "date": "20261011-142530",
        "endpoint_type": "openai-chat",
        "backend": "openai-chat",
        "label": None,
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_prompts": 1000,
        "request_rate": "inf",
        "burstiness": 1.0,
        "max_concurrency": 64,
        "duration": 95.2,
        "completed": 1000,
        "failed": 0,
        "request_throughput": 10.48,
        "output_throughput": throughput,
        "total_token_throughput": 12077.3,
        "latencies": [4.9, 5.1],
        "mean_ttft_ms": 182.4,
        "median_ttft_ms": 160.1,
        "std_ttft_ms": 70.3,
        "p99_ttft_ms": 410.9,
        "mean_tpot_ms": 38.2,
        "median_tpot_ms": 37.5,
        "p99_tpot_ms": 52.0,
        "mean_itl_ms": 37.9,
        "median_itl_ms": 35.0,
        "p99_itl_ms": 88.2,
    }
    record.update(extra)
    return record


def sglang_record(throughput: float = 1341.8, **extra) -> dict:
    record = {
        "tag": None,
        "backend": "sglang",
        "dataset_name": "random",
        "request_rate": float("inf"),
        "max_concurrency": 64,
        "random_input_len": 1024,
        "random_output_len": 128,
        "random_range_ratio": 0.0,
        "server_info": {"model_path": "meta-llama/Llama-3.1-8B-Instruct"},
        "duration": 95.2,
        "completed": 1000,
        "request_throughput": 10.48,
        "output_throughput": throughput,
        "total_throughput": 12077.1,
        "mean_e2e_latency_ms": 5010.0,
        "median_e2e_latency_ms": 4900.0,
        "p95_e2e_latency_ms": 7000.0,
        "p99_e2e_latency_ms": 8100.0,
        "mean_ttft_ms": 182.4,
        "median_ttft_ms": 160.1,
        "p95_ttft_ms": 350.7,
        "p99_ttft_ms": 410.9,
        "mean_tpot_ms": 38.2,
        "median_tpot_ms": 37.5,
        "p95_tpot_ms": 48.0,
        "p99_tpot_ms": 52.0,
        "concurrency": 52.6,
    }
    record.update(extra)
    return record


def aiperf_record(throughput: float = 1341.8, *, errors: float | None = 0.0) -> dict:
    record = {
        "schema_version": "1.5",
        "aiperf_version": "0.4.0",
        "time_to_first_token": {
            "unit": "ms",
            "avg": 182.4,
            "p50": 160.1,
            "p95": 350.7,
            "p99": 410.9,
            "std": 70.3,
            "count": 1000,
        },
        "inter_token_latency": {"unit": "ms", "avg": 37.9, "p50": 35.0, "p95": 60.1, "p99": 88.2},
        "request_latency": {"unit": "ms", "avg": 5010.0, "p50": 4900.0, "p99": 8100.0},
        "output_token_throughput": {"unit": "tokens/sec", "avg": throughput},
        "total_token_throughput": {"unit": "tokens/sec", "avg": 12077.3},
        "request_throughput": {"unit": "requests/sec", "avg": 10.48},
        "request_count": {"unit": "requests", "avg": 1000.0},
        "input_config": {
            "models": {"items": [{"name": "meta-llama/Llama-3.1-8B-Instruct"}]},
            "endpoint": {"urls": ["http://localhost:8000"], "type": "chat", "streaming": True},
            "phases": [{"name": "profiling", "type": "concurrency", "concurrency": 64}],
        },
        "telemetry_data": {
            "endpoints": {
                "localhost:9400": {"gpus": {"gpu_0": {"gpu_name": "NVIDIA H100 80GB HBM3"}}}
            }
        },
    }
    if errors is not None:
        record["error_request_count"] = {"unit": "requests", "avg": errors}
    return record


def _distribution(mean: float, p95: float, p99: float) -> dict:
    return {
        "mean": mean,
        "median": mean * 0.9,
        "std_dev": mean * 0.1,
        "percentiles": {"p50": mean * 0.9, "p95": p95, "p99": p99},
    }


def guidellm_benchmark(rate: float, throughput: float = 1341.8) -> dict:
    return {
        "type_": "generative_benchmark",
        "config": {
            "strategy": {"type_": "constant", "rate": rate, "max_concurrency": None},
            "backend": {"kind": "openai_http", "model": "meta-llama/Llama-3.1-8B-Instruct"},
        },
        "metrics": {
            "request_totals": {"successful": 998, "errored": 0, "incomplete": 2, "total": 1000},
            "requests_per_second": {"successful": _distribution(10.48, 11, 12)},
            "request_latency": {"successful": _distribution(5.01, 7.0, 8.1)},
            "time_to_first_token_ms": {"successful": _distribution(182.4, 350.7, 410.9)},
            "time_per_output_token_ms": {"successful": _distribution(38.2, 48.0, 52.0)},
            "inter_token_latency_ms": {"successful": _distribution(37.9, 60.1, 88.2)},
            "output_tokens_per_second": {"successful": _distribution(throughput, 1, 1)},
            "tokens_per_second": {"successful": _distribution(12077.3, 1, 1)},
        },
        "requests": {"successful": [{"huge": "per-request data"}]},
    }


def guidellm_file(*benchmarks: dict) -> dict:
    return {
        "metadata": {"version": 2, "guidellm_version": "0.6.0"},
        "config": {"spec": {"data": [{"kind": "synthetic_text"}]}},
        "benchmarks": list(benchmarks),
    }


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, list):
        text = "\n".join(json.dumps(item) for item in payload) + "\n"
    else:
        text = json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def test_detects_each_format():
    assert detect_format(vllm_record()) == "vllm"
    assert detect_format(sglang_record()) == "sglang"
    assert detect_format(aiperf_record()) == "aiperf"
    assert detect_format(guidellm_file(guidellm_benchmark(1))) == "guidellm"
    assert detect_format({"schema_version": "1", "environment": {}, "metrics": {}}) == "normalized"
    assert detect_format({"some": "other json"}) is None


def test_vllm_fields_map_and_label_is_not_a_parameter(tmp_path):
    _write(
        tmp_path / "openai-chat-infqps-concurrency64-Llama-3.1-8B-20261011.json",
        vllm_record(label="baseline"),
    )
    _write(
        tmp_path / "openai-chat-infqps-concurrency64-Llama-3.1-8B-20261011.pytorch.json", [{"x": 1}]
    )
    loaded = load_directory(tmp_path)
    assert loaded.format == "vllm"
    assert len(loaded.records) == 1
    result = loaded.records[0].result
    assert result.metrics.output_token_throughput == 1341.8
    assert result.metrics.ttft_ms.p99 == 410.9
    assert result.metrics.ttft_ms.p95 is None  # not written by default
    assert result.metrics.itl_ms.median == 35.0
    assert result.metrics.requests.failed == 0
    assert result.environment.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert result.environment.gpu == "unspecified"
    assert "label" not in result.benchmark_parameters
    assert result.benchmark_parameters["max_concurrency"] == 64


def test_vllm_metadata_supplies_gpu_and_version(tmp_path):
    _write(tmp_path / "run.vllm.json", vllm_record(gpu="NVIDIA L4", vllm_version="0.21.0"))
    result = load_directory(tmp_path).records[0].result
    assert result.environment.gpu == "NVIDIA L4"
    assert result.environment.vllm_version == "0.21.0"


def test_vllm_append_result_lines_are_repetitions(tmp_path):
    _write(tmp_path / "results.json", [vllm_record(1000), vllm_record(1010), vllm_record(990)])
    loaded = load_directory(tmp_path, "vllm")
    assert [item.record for item in loaded.records] == [1, 2, 3]
    assert loaded.records[2].label.endswith("results.json:3")


def test_sglang_jsonl_with_infinity_and_no_failed_count(tmp_path):
    _write(
        tmp_path / "b" / "sglang_1011_1000_1024_128.jsonl",
        [sglang_record(v) for v in (1000, 1010, 995)],
    )
    _write(
        tmp_path / "c" / "sglang_1011_1000_1024_128.jsonl",
        [sglang_record(v) for v in (990, 1000, 985)],
    )
    report = load_and_compare_directories(tmp_path / "b", tmp_path / "c")
    assert report["input"]["baseline"]["format"] == "sglang"
    assert report["result"] == "PASS"
    assert report["measurement_quality"]["failed_requests"] is False
    assert any("do not record failed requests" in note for note in report["notes"])
    assert any("gpu is not recorded" in note for note in report["notes"])


def test_sglang_errors_list_counts_failures(tmp_path):
    record = sglang_record(errors=["", "", "timeout"])
    _write(tmp_path / "run.jsonl", [record])
    result = load_directory(tmp_path).records[0].result
    assert result.metrics.requests.failed == 1
    assert result.failed_requests_reported is True
    assert result.metrics.e2el_ms.p95 == 7000.0


def test_aiperf_fields_units_and_gpu(tmp_path):
    record = aiperf_record()
    record["time_to_first_token"]["unit"] = "ns"
    record["time_to_first_token"]["avg"] = 182_400_000
    _write(tmp_path / "artifacts" / "profile_export_aiperf.json", record)
    _write(tmp_path / "artifacts" / "profile_export_aiperf_aggregate.json", {"mean": 1})
    loaded = load_directory(tmp_path)
    assert len(loaded.records) == 1
    result = loaded.records[0].result
    assert result.metrics.ttft_ms.mean == pytest.approx(182.4)
    assert result.metrics.tpot_ms.p99 == 88.2
    assert result.metrics.e2el_ms.median == 4900.0
    assert result.environment.gpu == "NVIDIA H100 80GB HBM3"
    assert result.benchmark_parameters["phases"][0]["concurrency"] == 64


def test_aiperf_failed_count_from_completed_minus_successful(tmp_path):
    record = aiperf_record(errors=None)
    record["completed_request_count"] = {"unit": "requests", "avg": 1003.0}
    _write(tmp_path / "profile_export_aiperf.json", record)
    result = load_directory(tmp_path).records[0].result
    assert result.metrics.requests.failed == 3
    assert result.failed_requests_reported is True


def test_guidellm_sweep_needs_a_benchmark_index(tmp_path):
    _write(
        tmp_path / "benchmarks.json", guidellm_file(guidellm_benchmark(5), guidellm_benchmark(10))
    )
    with pytest.raises(BenchmarkResultError, match="0: constant rate=5; 1: constant rate=10"):
        load_directory(tmp_path)
    result = load_directory(tmp_path, options=SourceOptions(guidellm_benchmark=1)).records[0].result
    assert result.benchmark_parameters["strategy"] == {"type_": "constant", "rate": 10}
    assert result.metrics.e2el_ms.mean == pytest.approx(5010.0)  # seconds converted to ms
    assert result.metrics.requests.failed == 0  # incomplete requests are not failures
    assert result.metrics.output_token_throughput == 1341.8


def test_auto_prefers_normalized_names_so_raw_files_beside_them_are_not_double_counted(tmp_path):
    normalized = {
        "schema_version": "1",
        "environment": {"gpu": "g", "model": "m"},
        "benchmark_parameters": {"c": 1},
        "metrics": {"output_token_throughput": 1000, "requests": {"successful": 1, "failed": 0}},
    }
    _write(tmp_path / "rep1" / "normalized.json", normalized)
    _write(tmp_path / "rep1" / "raw-vllm.json", vllm_record())
    loaded = load_directory(tmp_path)
    assert loaded.format == "normalized"
    assert len(loaded.records) == 1
    assert any(item.endswith("raw-vllm.json") for item in loaded.ignored)


def test_mixed_formats_in_one_directory_are_an_error(tmp_path):
    _write(tmp_path / "a.json", vllm_record())
    _write(tmp_path / "b.json", aiperf_record())
    with pytest.raises(BenchmarkResultError, match="mixes result formats"):
        load_directory(tmp_path)


def test_explicit_format_ignores_other_json(tmp_path):
    _write(tmp_path / "a.json", vllm_record())
    _write(tmp_path / "notes.json", {"hello": "world"})
    loaded = load_directory(tmp_path, "vllm")
    assert len(loaded.records) == 1
    assert loaded.ignored[0].endswith("notes.json")


def test_no_results_error_explains_both_routes(tmp_path):
    _write(tmp_path / "notes.json", {"hello": "world"})
    with pytest.raises(BenchmarkResultError, match="detected in any .json or .jsonl file"):
        load_directory(tmp_path)


def test_cli_compares_vllm_results_with_gpu_flag(tmp_path):
    for side, values in (("b", (1000, 1010, 995, 1005, 990)), ("c", (800, 810, 795, 805, 790))):
        for index, value in enumerate(values):
            _write(tmp_path / side / f"run-{index}.json", vllm_record(value))
    result = runner.invoke(
        app,
        [
            "compare-runs",
            "--baseline",
            str(tmp_path / "b"),
            "--candidate",
            str(tmp_path / "c"),
            "--metric",
            "ttft_p99_ms",
            "--gpu",
            "NVIDIA L4",
            "--json",
        ],
    )
    report = json.loads(result.output)
    assert report["result"] == "PASS"  # TTFT is identical; throughput fell
    assert report["environment"]["baseline"]["gpu"] == "NVIDIA L4"
    assert not any("gpu is not recorded" in note for note in report["notes"])

    result = runner.invoke(
        app, ["compare-runs", "--baseline", str(tmp_path / "b"), "--candidate", str(tmp_path / "c")]
    )
    assert result.exit_code == 1
    assert "RESULT: FAIL" in result.output
