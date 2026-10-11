"""Apply the compare-runs decision to every repeated-run comparison in vLLM issue #52630.

    python tools/validate_vllm_52630.py

Two sources, both public:

* the repetition values posted in the issue thread
  (https://github.com/vllm-project/vllm/issues/52630), copied on 2026-10-11, by the
  reporter (Yambr, 2026-08-28, H100 PCIe) and a maintainer (taneem-ibrahim, 2026-08-19 and
  2026-08-24, H100 SXM);
* this repository's own #52630 run, whose repetition values are kept in
  docs/benchmark-reliability-52630.md (its raw files are not in the repository).

Only the interval decision is applied. The validity checks of compare-runs (failed
requests, matching environments) need fields the posted numbers do not include, so every
comparison here is between runs on the same machine as reported by their authors.
docs/validation-vllm-52630.md records the output and what it means.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from inference_doctor.quality import QualityThresholds, RegressionDecision, decide_regression

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "docs" / "benchmark-reliability-52630.md"
# Rows of the "Repetition-level throughput" table: version, concurrency, three values.
STUDY_ROW = re.compile(r"^\| (baseline|candidate) \| (\d+) \| ([\d.]+, [\d.]+, [\d.]+) \|")

# texts/s per run, as posted. The reporter's value per run is the plateau (best) of a
# 16/32/64 concurrency sweep; the maintainer's runs are at concurrency 32.
REPORTER_V0190 = [3067.0, 3178.0, 3131.0]
MAINTAINER_TORCH_210 = [3119.00, 3034.29, 2982.14]
MAINTAINER_TORCH_211 = [3315.23, 3320.01, 3084.15]
MAINTAINER_MAIN = [3026.92, 2969.38, 2936.26]
NIGHTLY_ATTEMPT_1 = [2320.0, 2288.0, 2217.0]
NIGHTLY_ATTEMPT_2 = [2345.0, 2342.0, 2314.0]


@dataclass(frozen=True)
class Comparison:
    group: str
    name: str
    baseline: list[float]
    candidate: list[float]


def study_values() -> dict[tuple[str, int], list[float]]:
    values: dict[tuple[str, int], list[float]] = {}
    for line in STUDY.read_text(encoding="utf-8").splitlines():
        match = STUDY_ROW.match(line)
        if match:
            side, concurrency, numbers = match.groups()
            values[(side, int(concurrency))] = [float(n) for n in numbers.split(",")]
    return values


def comparisons() -> list[Comparison]:
    reporter = "Reporter, H100 PCIe"
    maintainer = "Maintainer, H100 SXM"
    rows = [
        Comparison(reporter, "v0.19.0 -> v0.27.1", REPORTER_V0190, [2404.0, 2464.0, 2425.0]),
        Comparison(reporter, "v0.19.0 -> v0.28.0", REPORTER_V0190, [2402.0, 2433.0, 2424.0]),
        Comparison(
            reporter, "v0.19.0 -> nightly 6f7df92a8, attempt 1", REPORTER_V0190, NIGHTLY_ATTEMPT_1
        ),
        Comparison(
            reporter, "v0.19.0 -> nightly 6f7df92a8, attempt 2", REPORTER_V0190, NIGHTLY_ATTEMPT_2
        ),
        Comparison(
            reporter, "v0.19.0 -> nightly a9a17e70", REPORTER_V0190, [1704.0, 2122.0, 2256.0]
        ),
        Comparison(
            reporter, "v0.27.1 -> v0.28.0", [2404.0, 2464.0, 2425.0], [2402.0, 2433.0, 2424.0]
        ),
        Comparison(
            reporter,
            "nightly attempt 1 -> attempt 2 (same build)",
            NIGHTLY_ATTEMPT_1,
            NIGHTLY_ATTEMPT_2,
        ),
        Comparison(
            reporter,
            "nightly -> nightly with --renderer-num-workers 8",
            NIGHTLY_ATTEMPT_2,
            [1654.0, 2377.0, 2429.0],
        ),
        Comparison(
            maintainer,
            "torch 2.10 -> 2.11, same vLLM commit",
            MAINTAINER_TORCH_210,
            MAINTAINER_TORCH_211,
        ),
        Comparison(
            maintainer,
            "torch 2.10 boundary -> main 29c9af52",
            MAINTAINER_TORCH_210,
            MAINTAINER_MAIN,
        ),
        Comparison(
            maintainer,
            "torch 2.11 boundary -> main 29c9af52",
            MAINTAINER_TORCH_211,
            MAINTAINER_MAIN,
        ),
    ]
    study = study_values()
    for concurrency in sorted({key[1] for key in study}):
        rows.append(
            Comparison(
                "This repository, H100 SXM",
                f"v0.19.0 -> v0.21.0, concurrency {concurrency}",
                study[("baseline", concurrency)],
                study[("candidate", concurrency)],
            )
        )
    return rows


def decide(comparison: Comparison) -> RegressionDecision:
    return decide_regression(
        comparison.baseline,
        comparison.candidate,
        lower_is_better=False,
        thresholds=QualityThresholds(),
    )


def main() -> None:
    print(
        "| Source | Comparison (3 runs per side) | Mean change | Regression, 90% interval | Verdict |"
    )
    print("| --- | --- | ---: | ---: | --- |")
    for comparison in comparisons():
        decision = decide(comparison)
        print(
            f"| {comparison.group} | {comparison.name} | {-decision.regression_percent:+.1f}% "
            f"| {decision.regression_percent_low:+.1f}% to {decision.regression_percent_high:+.1f}% "
            f"| {decision.status} |"
        )


if __name__ == "__main__":
    main()
