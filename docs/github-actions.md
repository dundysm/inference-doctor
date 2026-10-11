# GitHub Actions integration

Inference Doctor is designed to sit after your benchmark step and act as the merge gate.

It does not provision GPUs or decide how to benchmark your model. Your workflow produces repeated normalized benchmark artifacts; Inference Doctor evaluates measurement quality and regression risk.

## Minimal usage

```yaml
- name: Check inference performance
  uses: dundysm/inference-doctor@main
  with:
    baseline: artifacts/baseline
    candidate: artifacts/candidate
    metric: output_token_throughput, ttft_p99_ms
    regression-percent: "10"
    min-repetitions: "5"
```

`metric` takes one metric or several separated by commas. With several, the step fails if any metric fails, and the job summary has one row per metric.

The Action returns:

- `PASS` when the whole confidence interval for the regression is below the threshold
- `FAIL` when the whole interval is at or past the threshold
- `INCONCLUSIVE` when the interval contains the threshold, or the comparison is not valid (too few repetitions, failed requests, missing metrics, incomparable environments)
- `ERROR` for invalid input or execution errors

`FAIL`, `INCONCLUSIVE`, and `ERROR` fail the GitHub Action step. This is intentional: an untrustworthy or invalid benchmark should not silently approve a change.

## Typical self-hosted GPU workflow

```yaml
name: Inference performance

on:
  pull_request:

jobs:
  benchmark:
    runs-on: [self-hosted, linux, gpu]
    steps:
      - uses: actions/checkout@v4

      # Your own benchmark harness produces repeated normalized JSON files.
      - name: Benchmark baseline and candidate
        run: ./scripts/run-inference-benchmarks.sh

      - name: Gate performance regression
        uses: dundysm/inference-doctor@main
        with:
          baseline: artifacts/baseline
          candidate: artifacts/candidate
          metric: output_token_throughput
```

## Outputs

The Action exposes:

- `result`: the combined verdict
- `report`: the path to the full JSON report
- `baseline-mean`
- `candidate-mean`
- `delta-percent`
- `regression-percent-low` and `regression-percent-high`: the confidence interval for the regression, positive is worse
- `baseline-cv`
- `candidate-cv`

The per-metric outputs describe the first metric when several are given; read the JSON report for the others.

It also writes a Markdown summary to the GitHub Actions job summary. Invalid input is surfaced as a readable `ERROR` summary instead of a JSON parsing failure.

## Input contract

Each baseline/candidate directory contains repeated normalized JSON benchmark results. At least `min-repetitions` (default 3) are required on each side; 5 or more is recommended, see [gate calibration](gate-calibration.md).

Other inputs: `confidence` (default `0.90`) and `stability-cv-percent` (an optional CV ceiling, empty by default).

See [benchmark-format.md](benchmark-format.md) for the exact JSON contract and `examples/quickstart/` for the smallest copyable example.

Environment and workload parameters must be comparable across runs. See [measurement-quality.md](measurement-quality.md) for the full decision semantics.

## Why INCONCLUSIVE exists

A simple threshold can produce false confidence when benchmark variance is high. Inference Doctor refuses to make a merge recommendation when the confidence interval cannot separate the candidate from the threshold, requests fail, required metrics are unavailable, there are too few repetitions, or benchmark environments are not comparable.
