import json
import sys
from pathlib import Path

import pytest


EXPERIMENT_DIR = (
    Path(__file__).parent.parent / "experiments" / "vllm-48035"
)
sys.path.insert(0, str(EXPERIMENT_DIR))

from aggregate import aggregate_raw_runs  # noqa: E402
import compare_results  # noqa: E402
from benchmark_client import (  # noqa: E402
    parse_args as parse_benchmark_args,
    resolve_variant_metadata,
)
from harness_common import (  # noqa: E402
    normalized_result,
    parse_gpu_metadata,
    request_metrics,
    require_matching_gpu_uuid,
)
from run_experiment import build_docker_command  # noqa: E402
from inference_doctor.benchmark_results import (  # noqa: E402
    parse_normalized_benchmark,
)


ENVIRONMENT = {
    "gpu": "NVIDIA GeForce RTX 4090",
    "vllm_version": "0.19.1",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "image": "vllm/vllm-openai:v0.19.1",
}
PARAMETERS = {
    "prompt_count": 10,
    "seeds": [17, 29, 43],
    "max_tokens": 1024,
    "concurrency": 1,
    "repetitions": 1,
}


def _record(
    decode_rate: float | None,
    ttft_ms: float | None,
    tpot_ms: float | None,
    success: bool = True,
) -> dict:
    return {
        "success": success,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "decode_tokens_per_second": decode_rate,
    }


def _raw_run(records: list[dict], duration: float = 1.0) -> dict:
    return {
        "complete": True,
        "environment": ENVIRONMENT,
        "benchmark_parameters": PARAMETERS,
        "measured_duration_seconds": duration,
        "requests": records,
    }


def test_request_metrics_excludes_first_token_from_decode():
    metrics = request_metrics(
        ttft_ms=100.0,
        total_latency_ms=10100.0,
        output_tokens=1001,
    )

    assert metrics["decode_duration_ms"] == 10000.0
    assert metrics["tpot_ms"] == 10.0
    assert metrics["decode_tokens_per_second"] == 100.0


def test_normalization_produces_compare_compatible_result():
    result = normalized_result(
        environment=ENVIRONMENT,
        benchmark_parameters=PARAMETERS,
        records=[
            _record(100.0, 20.0, 10.0),
            _record(120.0, 30.0, 8.0),
            _record(None, None, None, success=False),
        ],
        measured_duration_seconds=2.0,
    )

    parsed = parse_normalized_benchmark(result)
    assert parsed.metrics.output_token_throughput == 110.0
    assert parsed.metrics.request_throughput == 1.0
    assert parsed.metrics.requests.successful == 2.0
    assert parsed.metrics.requests.failed == 1.0
    assert parsed.metrics.ttft_ms.median == 25.0


def test_aggregation_reports_run_to_run_variation():
    result = aggregate_raw_runs(
        [
            _raw_run(
                [
                    _record(100.0, 10.0, 10.0),
                    _record(120.0, 20.0, 8.0),
                ]
            ),
            _raw_run(
                [
                    _record(80.0, 30.0, 12.0),
                    _record(100.0, 40.0, 10.0),
                ]
            ),
        ]
    )

    parsed = parse_normalized_benchmark(result)
    variation = result["aggregation"][
        "run_to_run_decode_throughput"
    ]

    assert parsed.benchmark_parameters["repetitions"] == 2
    assert parsed.metrics.output_token_throughput == 100.0
    assert parsed.metrics.request_throughput == 2.0
    assert parsed.metrics.ttft_ms.p95 == pytest.approx(38.5)
    assert result["aggregation"]["decode_tokens_per_second"] == {
        "mean": 100.0,
        "median": 100.0,
    }
    assert variation["repetition_means"] == [110.0, 90.0]
    assert variation["mean"] == 100.0
    assert variation["sample_standard_deviation"] == pytest.approx(
        14.1421356
    )
    assert variation["coefficient_of_variation"] == pytest.approx(
        0.141421356
    )


def test_aggregation_rejects_incomplete_repetition():
    run = _raw_run([_record(100.0, 10.0, 10.0)])
    run["complete"] = False

    with pytest.raises(ValueError, match="incomplete"):
        aggregate_raw_runs([run])


def test_committed_workload_shape_is_fixed():
    config = json.loads(
        (EXPERIMENT_DIR / "experiment.json").read_text(encoding="utf-8")
    )
    prompts = json.loads(
        (EXPERIMENT_DIR / "prompts.json").read_text(encoding="utf-8")
    )

    assert len(prompts) == 10
    assert len({prompt["id"] for prompt in prompts}) == 10
    assert config["seeds"] == [17, 29, 43]
    assert config["request"]["max_tokens"] == 1024
    assert config["request"]["concurrency"] == 1
    assert config["repetitions"] == 3
    assert config["comparison"]["throughput_fail_percent"] == 10.0


def test_docker_command_uses_version_compatible_model_flag():
    image = "vllm/vllm-openai:v0.19.1"
    model = "Qwen/Qwen3-4B-Instruct-2507"
    command = build_docker_command(
        image,
        model,
        ["--max-model-len", "8192"],
    )

    image_index = command.index(image)
    assert command[image_index + 1 : image_index + 3] == ["--model", model]
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert command[-2:] == ["--max-model-len", "8192"]


def test_external_server_cli_defaults_to_repetition_set(tmp_path):
    args = parse_benchmark_args(
        [
            "--server-url",
            "http://localhost:8000",
            "--version-label",
            "0.19.1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert args.server_url == "http://localhost:8000"
    assert args.version_label == "0.19.1"
    assert args.repetition is None
    assert args.repetitions is None


def test_version_label_resolves_manifest_variant_and_image():
    config = json.loads(
        (EXPERIMENT_DIR / "experiment.json").read_text(encoding="utf-8")
    )

    variant, image = resolve_variant_metadata(
        config,
        "0.24.0",
        None,
        None,
    )

    assert variant == "candidate"
    assert image == "vllm/vllm-openai:v0.24.0"


def test_gpu_metadata_requires_one_rtx_4090_and_captures_uuid():
    metadata = parse_gpu_metadata(
        "NVIDIA GeForce RTX 4090, GPU-abc123, 610.43.02, 24564\n",
        "NVIDIA GeForce RTX 4090",
    )

    assert metadata == {
        "gpu": "NVIDIA GeForce RTX 4090",
        "gpu_uuid": "GPU-abc123",
        "driver_version": "610.43.02",
        "gpu_memory_mib": "24564",
    }


def test_cross_version_validation_rejects_different_gpu_uuid():
    baseline = {"environment": {"gpu_uuid": "GPU-baseline"}}
    candidate = {"environment": {"gpu_uuid": "GPU-candidate"}}

    with pytest.raises(ValueError, match="GPU UUID mismatch"):
        require_matching_gpu_uuid(baseline, candidate)


def test_offline_driver_aborts_before_comparison_on_gpu_change(
    tmp_path,
    monkeypatch,
):
    parameters = dict(PARAMETERS)
    baseline_environment = {**ENVIRONMENT, "gpu_uuid": "GPU-baseline"}
    candidate_environment = {
        **ENVIRONMENT,
        "vllm_version": "0.24.0",
        "image": "vllm/vllm-openai:v0.24.0",
        "gpu_uuid": "GPU-candidate",
    }

    for variant, environment in (
        ("baseline", baseline_environment),
        ("candidate", candidate_environment),
    ):
        for repetition in (1, 2):
            path = (
                tmp_path
                / variant
                / f"repetition-{repetition:03d}"
                / "raw.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "environment": environment,
                        "benchmark_parameters": parameters,
                        "measured_duration_seconds": 1.0,
                        "requests": [_record(100.0, 10.0, 10.0)],
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        compare_results,
        "run_comparison",
        lambda **kwargs: pytest.fail(
            "comparison ran before GPU UUID validation"
        ),
    )

    with pytest.raises(ValueError, match="GPU UUID mismatch"):
        compare_results.run_offline_comparisons(
            baseline_dir=tmp_path / "baseline",
            candidate_dir=tmp_path / "candidate",
            output_dir=tmp_path / "comparisons",
            config={
                "comparison": {
                    "latency_warn_percent": 5.0,
                    "latency_fail_percent": 10.0,
                    "throughput_warn_percent": 5.0,
                    "throughput_fail_percent": 10.0,
                    "max_failed_request_increase": 0,
                }
            },
        )


def test_offline_driver_labels_matching_uuid_cross_result_valid(
    tmp_path,
    monkeypatch,
):
    environment = {**ENVIRONMENT, "gpu_uuid": "GPU-same"}
    candidate_environment = {
        **environment,
        "vllm_version": "0.24.0",
        "image": "vllm/vllm-openai:v0.24.0",
    }
    for variant, run_environment in (
        ("baseline", environment),
        ("candidate", candidate_environment),
    ):
        for repetition in (1, 2):
            path = (
                tmp_path
                / variant
                / f"repetition-{repetition:03d}"
                / "raw.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "environment": run_environment,
                        "benchmark_parameters": PARAMETERS,
                        "measured_duration_seconds": 1.0,
                        "requests": [_record(100.0, 10.0, 10.0)],
                    }
                ),
                encoding="utf-8",
            )

    statuses = iter(["PASS", "PASS", "FAIL"])
    monkeypatch.setattr(
        compare_results,
        "run_comparison",
        lambda **kwargs: next(statuses),
    )

    summary = compare_results.run_offline_comparisons(
        baseline_dir=tmp_path / "baseline",
        candidate_dir=tmp_path / "candidate",
        output_dir=tmp_path / "comparisons",
        config={
            "comparison": {
                "latency_warn_percent": 5.0,
                "latency_fail_percent": 10.0,
                "throughput_warn_percent": 5.0,
                "throughput_fail_percent": 10.0,
                "max_failed_request_increase": 0,
            }
        },
    )

    assert summary["gpu_uuid"] == "GPU-same"
    assert summary["cross_version_valid"] is True
    assert summary["historical_regression_reproduced"] is True
