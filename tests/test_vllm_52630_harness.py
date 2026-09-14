import copy
import importlib
import json
import sys
from pathlib import Path

import pytest


EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments" / "vllm-52630"
sys.path.insert(0, str(EXPERIMENT_DIR))
for _module in ("harness_common", "aggregate", "compare_results", "run_same_gpu"):
    sys.modules.pop(_module, None)

harness_common = importlib.import_module("harness_common")
aggregate = importlib.import_module("aggregate")
compare_results = importlib.import_module("compare_results")
run_same_gpu = importlib.import_module("run_same_gpu")


ENVIRONMENT = {
    "variant": "baseline",
    "version_label": "0.19.0",
    "gpu": {"gpu_uuid": "GPU-test"},
}
PARAMETERS = {
    "experiment": "vllm-52630",
    "model": "BAAI/bge-m3",
    "server": {"port": 8043},
    "dataset": {"revision": "test", "selected_indices": {"count": 4096}},
    "batch_count": 128,
    "request_shape": "32 documents per /v1/embeddings request",
    "concurrency": [16, 32, 64],
    "warmup_batches": 4,
}


def _point(value: float, latency: float = 10.0, failed: int = 0) -> dict:
    return {
        "concurrency": 32,
        "successful_requests": 128 - failed,
        "failed_requests": failed,
        "total_texts_processed": (128 - failed) * 32,
        "wall_clock_duration_seconds": (128 - failed) * 32 / value,
        "request_throughput": value / 32,
        "texts_per_second": value,
        "latency_ms": {"mean": latency, "median": latency, "p95": latency},
        "server_errors": [],
        "requests": [],
    }


def _raw(values: tuple[float, float, float], repetition: int = 1) -> dict:
    results = {
        str(level): _point(value, latency=level / 2)
        for level, value in zip((16, 32, 64), values)
    }
    return {
        "schema_version": "vllm-52630-raw-1",
        "complete": True,
        "environment": ENVIRONMENT,
        "benchmark_parameters": {**PARAMETERS, "repetition": repetition},
        "concurrency_results": results,
    }


def _write_normalized(path: Path, value: float) -> None:
    payload = harness_common.normalized_repetition(
        environment=ENVIRONMENT,
        benchmark_parameters=PARAMETERS,
        concurrency_results={"32": _point(value)},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_selection_is_deterministic_and_manifest_hash_matches():
    manifest = {
        "selection": {"kind": "contiguous_row_range", "start": 0, "count": 4096},
        "selected_index_sha256": "cb2249fa75b0f55c6f8922ce37038f8596d44b3ecf325f1ddc6d9d8a301d7e66",
    }
    indices = harness_common.selected_indices(manifest)
    assert indices == list(range(4096))
    assert harness_common.selected_index_sha256(indices) == manifest["selected_index_sha256"]


def test_batches_are_exactly_32_documents():
    documents = [{"id": str(index), "text": "x"} for index in range(4096)]
    batches = harness_common.make_batches(documents, 32)
    assert len(batches) == 128
    assert {len(batch) for batch in batches} == {32}
    assert batches[0][0]["id"] == "0"
    assert batches[-1][-1]["id"] == "4095"


def test_normalization_preserves_texts_per_second_and_latency():
    point = _point(100.0, latency=12.5)
    normalized = harness_common.normalized_repetition(
        environment=ENVIRONMENT,
        benchmark_parameters=PARAMETERS,
        concurrency_results={"32": point},
    )
    assert normalized["metrics"]["texts_per_second"]["32"] == 100.0
    assert normalized["metrics"]["request_latency_ms"]["32"] == 12.5
    assert normalized["metrics"]["requests"] == {"successful": 128, "failed": 0}


def test_aggregation_groups_concurrency_and_reports_run_statistics():
    runs = [_raw((90.0, 100.0, 80.0), 1), _raw((91.0, 101.0, 81.0), 2), _raw((89.0, 99.0, 79.0), 3)]
    runs[0]["concurrency_results"]["64"]["latency_ms"] = {"mean": 31.0, "median": 30.0, "p95": 40.0}
    runs[1]["concurrency_results"]["64"]["latency_ms"] = {"mean": 32.0, "median": 31.0, "p95": 50.0}
    runs[2]["concurrency_results"]["64"]["latency_ms"] = {"mean": 33.0, "median": 32.0, "p95": 60.0}
    result = aggregate.aggregate_repetitions(runs)
    assert result["metrics"]["texts_per_second"]["32"]["mean"] == pytest.approx(100.0)
    assert result["metrics"]["texts_per_second"]["32"]["median"] == pytest.approx(100.0)
    assert result["metrics"]["texts_per_second"]["32"]["sample_standard_deviation"] == pytest.approx(1.0)
    assert result["metrics"]["request_latency_median_ms"]["64"]["median"] == pytest.approx(31.0)
    assert result["metrics"]["request_latency_p95_ms"]["64"]["median"] == pytest.approx(50.0)
    assert result["metrics"]["request_latency_p95_ms"]["64"]["p95"] == pytest.approx(59.0)
    assert result["metrics"]["best_stable_plateau"]["concurrency"] == 32


def test_aggregation_rejects_missing_or_incomplete_runs():
    incomplete = _raw((90.0, 100.0, 80.0))
    incomplete["complete"] = False
    with pytest.raises(harness_common.ExperimentError):
        aggregate.aggregate_repetitions([incomplete])

    failed = _raw((90.0, 100.0, 80.0))
    failed["concurrency_results"]["32"]["failed_requests"] = 1
    result = aggregate.aggregate_repetitions([failed])
    assert result["metrics"]["requests"]["failed"] == 1


def test_same_version_controls_use_absolute_stability():
    baseline = harness_common.normalized_repetition(
        environment=ENVIRONMENT,
        benchmark_parameters=PARAMETERS,
        concurrency_results={"16": _point(100.0)},
    )
    candidate = copy.deepcopy(baseline)
    candidate["metrics"]["texts_per_second"]["16"] = 115.0
    result = compare_results.compare_results(
        baseline, candidate, same_version=True
    )
    assert result["overall_status"] == "UNSTABLE"
    assert result["metrics"][0]["status"] == "UNSTABLE"

    candidate["metrics"]["texts_per_second"]["16"] = 101.0
    result = compare_results.compare_results(
        baseline, candidate, same_version=True
    )
    assert result["overall_status"] == "PASS"


def test_cross_version_throughput_drop_fails_and_improvement_passes():
    baseline = aggregate.aggregate_repetitions([_raw((100.0, 100.0, 100.0))])
    candidate = aggregate.aggregate_repetitions([_raw((85.0, 85.0, 85.0))])
    failed = compare_results.compare_results(baseline, candidate)
    assert failed["overall_status"] == "FAIL"
    assert failed["metrics"][1]["percentage_delta"] == pytest.approx(-15.0)

    improved = aggregate.aggregate_repetitions([_raw((115.0, 115.0, 115.0))])
    passed = compare_results.compare_results(baseline, improved)
    assert passed["overall_status"] == "PASS"


def test_missing_throughput_is_warn_and_failed_requests_are_visible():
    baseline = harness_common.normalized_repetition(
        environment=ENVIRONMENT,
        benchmark_parameters=PARAMETERS,
        concurrency_results={"32": _point(100.0)},
    )
    candidate = copy.deepcopy(baseline)
    candidate["metrics"]["texts_per_second"]["32"] = None
    candidate["metrics"]["requests"]["failed"] = 2
    result = compare_results.compare_results(baseline, candidate)
    assert result["metrics"][0]["status"] == "WARN"
    assert result["metrics"][1]["status"] == "FAIL"
    assert result["overall_status"] == "FAIL"


def test_normalized_result_is_json_serializable():
    payload = harness_common.normalized_repetition(
        environment=ENVIRONMENT,
        benchmark_parameters=PARAMETERS,
        concurrency_results={"32": _point(100.0)},
    )
    assert json.loads(json.dumps(payload)) == payload


def test_runner_uses_each_variant_environment_python():
    assert run_same_gpu.variant_python_executable({"environment_path": "/workspace/envs/v0190"}) == "/workspace/envs/v0190/bin/python"
    assert run_same_gpu.variant_python_executable({"environment_path": "/workspace/envs/v0210"}) == "/workspace/envs/v0210/bin/python"


def test_three_repetition_control_marks_rep3_instability(tmp_path):
    variant_dir = tmp_path / "baseline"
    for repetition, value in enumerate((100.0, 101.0, 120.0), start=1):
        _write_normalized(variant_dir / f"repetition-{repetition:03d}" / "normalized.json", value)
    status = run_same_gpu.run_stability_control(
        name="baseline-vs-baseline",
        variant_dir=variant_dir,
        repetitions=3,
        config={"comparison": {"stability_tolerance_percent": 5.0, "throughput_warn_percent": 5.0, "throughput_fail_percent": 10.0}},
        output_dir=tmp_path / "comparisons",
    )
    assert status == "UNSTABLE"


def test_three_repetition_control_passes_when_all_pairs_are_stable(tmp_path):
    variant_dir = tmp_path / "baseline"
    for repetition, value in enumerate((100.0, 101.0, 99.0), start=1):
        _write_normalized(variant_dir / f"repetition-{repetition:03d}" / "normalized.json", value)
    status = run_same_gpu.run_stability_control(
        name="baseline-vs-baseline",
        variant_dir=variant_dir,
        repetitions=3,
        config={"comparison": {"stability_tolerance_percent": 5.0, "throughput_warn_percent": 5.0, "throughput_fail_percent": 10.0}},
        output_dir=tmp_path / "comparisons",
    )
    assert status == "PASS"


def test_primary_concurrency_gates_historical_reproduction():
    cross = {
        "metrics": [
            {"metric": "texts_per_second[16]", "percentage_delta": -12.0},
            {"metric": "texts_per_second[32]", "percentage_delta": -4.0},
            {"metric": "texts_per_second[64]", "percentage_delta": -15.0},
        ]
    }
    assert not run_same_gpu.historical_regression_is_reproduced(
        baseline_control="PASS", candidate_control="PASS", cross_result=cross,
        primary_concurrency=32, fail_threshold_percent=10.0,
    )
    cross["metrics"][1]["percentage_delta"] = -10.1
    assert run_same_gpu.historical_regression_is_reproduced(
        baseline_control="PASS", candidate_control="PASS", cross_result=cross,
        primary_concurrency=32, fail_threshold_percent=10.0,
    )


def test_manifest_freezes_immutable_dataset_revision():
    manifest = json.loads((EXPERIMENT_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["revision"] == "7229066ed7a617ef50aad51c5a2d95957eaee537"
    assert manifest["resolved_commit_sha"] == manifest["revision"]
    assert "datasets_fingerprint" in manifest
    assert "selected_document_ids_sha256" in manifest
