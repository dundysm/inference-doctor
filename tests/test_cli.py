"""The exit-code contract shared by every command, and --version."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inference_doctor import __version__
from inference_doctor.cli import EXIT_ERROR, app

runner = CliRunner()


def _write(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "environment": {"gpu": "g", "model": "m"},
        "benchmark_parameters": {"c": 1},
        "metrics": {"requests": {"successful": 100, "failed": 0}, **metrics},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_version_matches_package():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"inference-doctor {__version__}"


def test_changelog_has_a_section_for_the_package_version():
    changelog = (Path(__file__).parent.parent / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## {__version__}" in changelog


@pytest.mark.parametrize(
    "args",
    [
        ["nosuchcommand"],
        ["compare-runs", "--baseline", "."],
        ["compare-runs", "--baseline", ".", "--candidate", ".", "--no-such-option"],
        ["compare-runs", "--baseline", "does-not-exist", "--candidate", "."],
        ["compare", "--baseline", "does-not-exist.json", "--candidate", "x.json"],
    ],
)
def test_usage_errors_exit_3_not_inconclusive(args, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, args)
    assert result.exit_code == EXIT_ERROR


def test_internal_error_exits_3_without_traceback(tmp_path, monkeypatch):
    import inference_doctor.cli as cli

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(cli, "load_and_compare_directories", boom)
    monkeypatch.delenv("INFERENCE_DOCTOR_DEBUG", raising=False)
    for side in ("baseline", "candidate"):
        _write(tmp_path / side / "repetition-001.json", {"output_token_throughput": 1})
    result = runner.invoke(
        app,
        [
            "compare-runs",
            "--baseline",
            str(tmp_path / "baseline"),
            "--candidate",
            str(tmp_path / "candidate"),
        ],
    )
    assert result.exit_code == EXIT_ERROR
    assert "internal error: RuntimeError: synthetic failure" in result.output
    assert "Traceback" not in result.output
