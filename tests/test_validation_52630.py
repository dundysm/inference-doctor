"""The verdicts recorded in docs/validation-vllm-52630.md, on the real #52630 numbers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from validate_vllm_52630 import comparisons, decide  # noqa: E402

DOC = Path(__file__).parent.parent / "docs" / "validation-vllm-52630.md"
VERDICTS = {(c.group, c.name): decide(c) for c in comparisons()}


@pytest.mark.parametrize(
    ("group", "name", "verdict"),
    [
        ("Reporter, H100 PCIe", "v0.19.0 -> v0.27.1", "FAIL"),
        ("Reporter, H100 PCIe", "v0.19.0 -> v0.28.0", "FAIL"),
        ("Reporter, H100 PCIe", "v0.27.1 -> v0.28.0", "PASS"),
        ("Reporter, H100 PCIe", "nightly attempt 1 -> attempt 2 (same build)", "PASS"),
        ("Reporter, H100 PCIe", "nightly -> nightly with --renderer-num-workers 8", "INCONCLUSIVE"),
        ("Maintainer, H100 SXM", "torch 2.10 -> 2.11, same vLLM commit", "PASS"),
        ("Maintainer, H100 SXM", "torch 2.11 boundary -> main 29c9af52", "INCONCLUSIVE"),
        ("This repository, H100 SXM", "v0.19.0 -> v0.21.0, concurrency 64", "INCONCLUSIVE"),
    ],
)
def test_recorded_verdicts(group, name, verdict):
    assert VERDICTS[(group, name)].status == verdict


def test_every_row_in_the_document_matches_the_tool():
    text = DOC.read_text(encoding="utf-8")
    for (group, name), decision in VERDICTS.items():
        row = (
            f"| {group} | {name} | {-decision.regression_percent:+.1f}% "
            f"| {decision.regression_percent_low:+.1f}% to {decision.regression_percent_high:+.1f}% "
            f"| {decision.status} |"
        )
        assert row in text, row
