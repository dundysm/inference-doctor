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
    metric: output_token_throughput
    stability-cv-percent: "5"
    regression-percent: "10"
```

The Action returns:

- `PASS` when both sides are stable and the regression threshold is not crossed
- `FAIL` when both sides are stable and the candidate crosses the regression threshold
- `INCONCLUSIVE` when the comparison is not trustworthy enough to make a merge decision
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

- `result`
- `baseline-mean`
- `candidate-mean`
- `delta-percent`
- `baseline-cv`
- `candidate-cv`

It also writes a Markdown summary to the GitHub Actions job summary. Invalid input is surfaced as a readable `ERROR` summary instead of a JSON parsing failure.

## Input contract

Each baseline/candidate directory contains repeated normalized JSON benchmark results. At least two repetitions per side are required for a stability decision.

See [benchmark-format.md](benchmark-format.md) for the exact JSON contract and `examples/quickstart/` for the smallest copyable example.

Environment and workload parameters must be comparable across runs. See [measurement-quality.md](measurement-quality.md) for the full decision semantics.

## Why INCONCLUSIVE exists

A simple threshold can produce false confidence when benchmark variance is high. Inference Doctor refuses to make a merge recommendation when the baseline or candidate exceeds the configured stability threshold, requests fail, required metrics are unavailable, or benchmark environments are not comparable.
