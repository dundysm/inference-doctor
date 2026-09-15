# Normalized benchmark format

Inference Doctor does not force a benchmark harness on you. Your existing load test or benchmark only needs to emit one small normalized JSON file per repetition.

`compare-runs` discovers files named `normalized.json`, `*.normalized.json`, or `repetition-*.json` recursively under the baseline and candidate directories.

## Minimal example

```json
{
  "schema_version": "1",
  "environment": {
    "gpu": "NVIDIA H100 80GB HBM3",
    "vllm_version": "0.19.0",
    "model": "my-model",
    "image": "my-serving-image"
  },
  "benchmark_parameters": {
    "concurrency": 32,
    "input_tokens": 512,
    "output_tokens": 128
  },
  "metrics": {
    "output_token_throughput": 12450.7,
    "request_throughput": 18.4,
    "requests": {
      "successful": 1000,
      "failed": 0
    },
    "ttft_ms": {
      "mean": 42.1,
      "median": 39.8,
      "p95": 61.5
    },
    "tpot_ms": {
      "mean": 8.2,
      "median": 8.0,
      "p95": 9.1
    }
  }
}
```

Only the metric you choose for `compare-runs` needs a value, but every repetition should record `metrics.requests.failed` so measurement validity can be checked. Use `0` when there were no failed requests.

For pooling or embedding workloads, use `texts_per_second`:

```json
"metrics": {
  "texts_per_second": 3378.56,
  "requests": {
    "successful": 1000,
    "failed": 0
  },
  "ttft_ms": {},
  "tpot_ms": {}
}
```

## Required metadata

Each result needs:

- `schema_version`: currently `"1"`
- `environment.gpu`: non-empty GPU model/name
- `environment.model`: non-empty model name
- `benchmark_parameters`: a non-empty JSON object describing the workload
- `metrics.requests.failed`: use `0` for a clean run
- the metric selected on the CLI

`environment.vllm_version` and `environment.image` may be `null`.

## Comparability rules

Inference Doctor requires baseline and candidate runs to agree on:

- GPU
- model
- `benchmark_parameters`

A different `vllm_version` is allowed because comparing versions is a common use case. Keep every workload-affecting setting in `benchmark_parameters` so an accidental apples-to-oranges comparison becomes `INCONCLUSIVE` instead of a false performance decision.

Example parameters worth recording include concurrency, request count or duration, input/output token lengths, tensor parallelism, dtype/quantization, batch size, dataset revision, and server flags that materially change the workload.

## Directory layout

At least two repetitions per side are required for a stability decision.

```text
artifacts/
  baseline/
    repetition-001.json
    repetition-002.json
    repetition-003.json
  candidate/
    repetition-001.json
    repetition-002.json
    repetition-003.json
```

Then run:

```bash
inference-doctor compare-runs \
  --baseline artifacts/baseline \
  --candidate artifacts/candidate \
  --metric output_token_throughput
```

See `examples/quickstart/` for copyable files and [measurement-quality.md](measurement-quality.md) for the exact decision semantics.
