# Measurement Quality MVP

`compare-runs` compares repeated, already-produced normalized benchmark
results. It does not provision GPUs, launch servers, generate traffic, or
change Inference Doctor diagnostic rules.

## Decision semantics

The defaults are a 5% coefficient-of-variation stability threshold and a 10%
regression threshold:

```text
invalid input or execution error -> exit 3
benchmark/environment validity failure -> INCONCLUSIVE, exit 2
baseline CV > 5% -> INCONCLUSIVE, exit 2
candidate CV > 5% -> INCONCLUSIVE, exit 2
directional regression >= 10% -> FAIL, exit 1
otherwise -> PASS, exit 0
```

Throughput regressions are decreases. Latency regressions are increases. Every
repetition remains in the official mean, median, sample standard deviation,
CV, minimum, and maximum. Simple MAD/IQR flags are descriptive only; no
observation is automatically removed.

## Input and output

Each input directory may contain `normalized.json`, `*.normalized.json`, or
`repetition-*.json` files recursively. The normalized schema supports the
existing throughput fields plus optional `metrics.texts_per_second` for
pooling/embedding benchmarks.

```bash
inference-doctor compare-runs \
  --baseline path/to/baseline \
  --candidate path/to/candidate \
  --metric output_token_throughput \
  --stability-cv-percent 5 \
  --regression-percent 10
```

Supported metrics include output-token throughput, request throughput, texts
per second, TTFT/TPOT mean/median/p95, and request counts. Add `--json` for a
machine-readable report. Exit codes are `0 PASS`, `1 FAIL`, `2 INCONCLUSIVE`,
and `3 invalid input or execution error`.

The comparison validates GPU model, model name, and benchmark parameters. A
baseline/candidate vLLM version difference is expected and is retained as
metadata; it is not itself a diagnostic signal.

## Example

The sanitized `examples/inference-ci-reliability-001` fixture preserves all
six baseline and six candidate values from the completed reliability run,
including the anomalous baseline block. Its official result is
`INCONCLUSIVE`: the observed throughput delta was about -4.19%, but baseline
CV was 8.93%, above the 5% stability threshold.

The optional [GitHub Actions workflow](../.github/workflows/inference-ci.yml)
assumes a self-hosted runner already has benchmark artifacts in
`artifacts/baseline` and `artifacts/candidate`. It writes the result, means,
delta, and CVs to the job summary and preserves the CLI exit code.
