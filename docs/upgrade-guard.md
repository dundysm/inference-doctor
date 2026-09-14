# Experimental inference upgrade guard

`inference-doctor compare` compares two previously produced, normalized benchmark result files. It does not launch vLLM, generate traffic, provision hardware, or alter the existing diagnostic rules.

```bash
inference-doctor compare \
  --baseline baseline.json \
  --candidate candidate.json
```

Use `--json` for machine-readable output. Exit code `0` means the comparison passed, `1` means a regression failed the guard or a same-version control is unstable, and `2` means the inputs or thresholds were invalid or the runs were incomparable. WARN rows do not fail the overall guard.

Same-version controls use a separate stability mode:

```bash
inference-doctor compare \
  --baseline baseline-repetition-001.json \
  --candidate baseline-repetition-002.json \
  --same-version \
  --stability-tolerance-percent 5
```

In this mode, percentage-based metrics are STABLE/PASS only when the absolute percentage variation is within the tolerance. Positive and negative variation beyond the tolerance produce `UNSTABLE` and exit code `1`. Cross-version comparisons keep the directional regression semantics shown below.

## Normalized input schema

Latency values are milliseconds; throughput values are rates per second.

```json
{
  "schema_version": "1",
  "environment": {
    "gpu": "NVIDIA L4",
    "vllm_version": "0.29.0",
    "model": "Qwen/Qwen3-0.6B",
    "image": "vllm/vllm-openai:v0.29.0-cu129"
  },
  "benchmark_parameters": {
    "dataset": "random",
    "input_tokens": 512,
    "output_tokens": 128,
    "num_prompts": 100,
    "max_concurrency": 32,
    "request_rate": "inf",
    "seed": 7
  },
  "metrics": {
    "ttft_ms": {"mean": 40.0, "median": 38.0, "p95": 60.0},
    "tpot_ms": {"mean": 10.0, "median": 9.5, "p95": 12.0},
    "output_token_throughput": 1000.0,
    "request_throughput": 10.0,
    "requests": {"successful": 100, "failed": 0}
  }
}
```

`gpu`, `model`, and the complete `benchmark_parameters` object must match between runs. vLLM version and image may differ because they are expected upgrade dimensions. Environment metadata is reported but does not affect metric classification.

Missing or non-finite metrics are rendered as unavailable and classified WARN. Negative values and non-numeric metric values are rejected.

## Thresholds

Lower TTFT and TPOT are better; higher throughput and successful-request counts are better. Failed-request increases are evaluated as an absolute count.

| Option | Default |
| --- | ---: |
| `--latency-warn-percent` | 5% |
| `--latency-fail-percent` | 10% |
| `--throughput-warn-percent` | 5% |
| `--throughput-fail-percent` | 10% |
| `--max-failed-request-increase` | 0 |
| `--stability-tolerance-percent` | 5% |

For percentage-based metrics, regressions below the warn threshold PASS, regressions at or above the warn threshold WARN, and regressions at or above the fail threshold FAIL. Improvements always PASS. Any increase above the configured failed-request allowance FAILs; a positive increase within a nonzero allowance WARNs.

The vLLM 48035 harness invokes `--same-version` for A/A and B/B controls. Its default 5% tolerance means a control with either a +15% or -15% change is `UNSTABLE`; an approximately +/-1% change is stable. The A/B comparison remains directional.

The example fixtures are [baseline.json](../tests/fixtures/compare/baseline.json) and [candidate_28pct_regression.json](../tests/fixtures/compare/candidate_28pct_regression.json).
