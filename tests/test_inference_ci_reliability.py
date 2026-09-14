from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments" / "inference-ci-reliability-001"
LEGACY_DIR = Path(__file__).parent.parent / "experiments" / "vllm-52630"
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(LEGACY_DIR))
for module_name in (
    "common",
    "analysis",
    "benchmark_block",
    "run_experiment",
    "harness_common",
    "benchmark_pooling",
    "run_same_gpu",
):
    sys.modules.pop(module_name, None)

import analysis  # noqa: E402
import benchmark_block  # noqa: E402
import run_experiment  # noqa: E402

# The legacy runner uses generic module names (for example ``aggregate``).
# Remove those aliases after importing the functions under test so later
# experiment suites can import their own modules without collection-order
# dependent collisions.
for module_name in (
    "aggregate",
    "benchmark_pooling",
    "compare_results",
    "harness_common",
    "run_same_gpu",
    "run_experiment",
):
    sys.modules.pop(module_name, None)


def _summary(values, failed_requests=0):
    return analysis.summarize_measurements(
        values,
        failed_requests=failed_requests,
        successful_texts=len(values) * 32,
    )


def test_frozen_config_uses_five_minute_windows_and_exact_abba_schedule():
    config = json.loads((EXPERIMENT_DIR / "experiment.json").read_text(encoding="utf-8"))
    assert config["warmup_seconds"] == 300
    assert config["measurement_seconds"] == 300
    assert tuple(config["schedule"]) == run_experiment.SCHEDULE
    assert config["schedule"].count("baseline") == 6
    assert config["schedule"].count("candidate") == 6
    assert config["dataset"]["concurrency"] == 32


def test_short_fake_window_cycles_and_records_requests():
    calls = []

    async def fake_sender(batch_index, batch, cycle_index, phase):
        calls.append((batch_index, cycle_index, phase))
        await asyncio.sleep(0.001)
        return {
            "success": True,
            "texts_returned": len(batch),
            "latency_ms": 1.0,
            "error": None,
        }

    batches = [[{"id": str(index), "text": "x"} for index in range(32)] for _ in range(4)]
    window = asyncio.run(
        benchmark_block.run_window(
            client=None,
            server_url="http://unused",
            model="bge",
            batches=batches,
            headers={},
            duration_seconds=0.02,
            concurrency=2,
            phase="measurement",
            request_sender=fake_sender,
        )
    )
    assert window["configured_duration_seconds"] == pytest.approx(0.02)
    assert window["requests"]
    assert {record["phase"] for record in window["requests"]} == {"measurement"}


def test_cycle_indices_repeat_the_full_4096_document_workload():
    indices = benchmark_block.cycle_batch_indices(128, 260)
    assert indices[:4] == [0, 1, 2, 3]
    assert indices[127] == 127
    assert indices[128:132] == [0, 1, 2, 3]


def test_gpu_uuid_is_validated_on_every_block():
    same = {"gpu_uuid": "GPU-1"}
    run_experiment.validate_block_gpu_uuid("GPU-1", same, "GPU-1", same)
    with pytest.raises(ValueError, match="GPU UUID changed"):
        run_experiment.validate_block_gpu_uuid("GPU-1", same, "GPU-2", same)


def test_cv_at_or_below_five_percent_is_stable():
    result = analysis.stability_status(
        _summary([100, 101, 99, 100, 100, 100]),
        _summary([100, 101, 99, 100, 100, 100]),
    )
    assert result == "STABLE"


def test_cv_above_five_percent_is_inconclusive():
    result = analysis.stability_status(
        _summary([100, 115, 100, 100, 100, 100]),
        _summary([100, 100, 100, 100, 100, 100]),
    )
    assert result == "INCONCLUSIVE / UNSTABLE"


def test_stable_twelve_percent_regression_fails():
    result = analysis.regression_decision(_summary([100] * 6), _summary([88] * 6))
    assert result["status"] == "FAIL / REGRESSION"


def test_stable_seven_percent_regression_passes():
    result = analysis.regression_decision(_summary([100] * 6), _summary([93] * 6))
    assert result["status"] == "PASS"


def test_unstable_fifteen_percent_regression_is_inconclusive():
    result = analysis.regression_decision(
        _summary([100] * 6),
        _summary([85, 100, 100, 100, 100, 100]),
    )
    assert result["status"] == "INCONCLUSIVE"


def test_no_outlier_removal_from_six_measurements():
    result = _summary([100, 100, 100, 100, 100, 160])
    assert result["count"] == 6
    assert result["values"] == [100.0, 100.0, 100.0, 100.0, 100.0, 160.0]
    assert result["max"] == 160.0
