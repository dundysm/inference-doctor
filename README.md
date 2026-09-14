# Inference Doctor

Inference Doctor is a small, read-only CLI that turns vLLM Prometheus metrics into focused hypotheses about inference bottlenecks. Point it at an existing Prometheus server and it reports whether the current evidence is more consistent with queue/capacity pressure, decode pressure, or meaningful KV-cache pressure.

It is deliberately conservative: high GPU or KV-cache utilization by itself is not an incident. A finding needs corroborating latency or scheduler evidence, so a busy but healthy server remains a healthy server.

```
vLLM /metrics  --->  Prometheus  --->  inference-doctor diagnose  --->  terminal report
    histograms,          PromQL              snapshot + rules          evidence + next test
    scheduler state
```

## Quick start

Use a supported Python version (3.11 or newer), clone this repository, and install the CLI into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -e .
```

With vLLM already being scraped by Prometheus, run:

```bash
inference-doctor diagnose \
  --prometheus http://localhost:9090 \
  --ttft-slo-ms 400 \
  --tpot-slo-ms 50
```

The command reads Prometheus only. It does not alter vLLM, Prometheus, or traffic. Omit an SLO only when you intentionally do not want latency-based findings for that dimension.

To compare two already-produced normalized benchmark results as a CI upgrade guard:

```bash
inference-doctor compare \
  --baseline baseline.json \
  --candidate candidate.json
```

See [docs/upgrade-guard.md](docs/upgrade-guard.md) for the experimental schema, thresholds, output, and exit codes. This command compares benchmark results only; it does not run workloads or change diagnostic behavior.

For repeated measurements, use the measurement-quality MVP:

```bash
inference-doctor compare-runs \
  --baseline artifacts/baseline \
  --candidate artifacts/candidate \
  --metric output_token_throughput
```

It reports `PASS`, `FAIL`, or `INCONCLUSIVE`. The default rule requires both
repetition sets to have CV at or below 5% and then treats a 10% directional
regression as `FAIL`. An unstable measurement is a first-class
`INCONCLUSIVE` result, not a failed command or an implicit merge approval.
See [docs/measurement-quality.md](docs/measurement-quality.md) and the
[sanitized reliability example](examples/inference-ci-reliability-001/README.md).

For a local Docker Desktop / WSL2 setup and an optional disposable RunPod reproduction, see [docs/local-integration.md](docs/local-integration.md). The checked-in compose stack pins the vLLM image used for integration validation rather than relying on `latest`.

## What a report looks like

The CLI prints a compact current snapshot followed by any triggered findings and suggested discriminating experiments. [docs/example-reports.md](docs/example-reports.md) contains four annotated, representative reports:

- healthy low-load service, including high utilization without a false positive
- queue/capacity pressure
- decode pressure
- KV-cache pressure

Queue p95 is included as observational context, not as evidence that increases queue-pressure confidence. Prometheus histogram bucket granularity can make a queue p95 look surprising next to TTFT; see [Limitations](#limitations).

## v0.2 signals

Inference Doctor expects the following vLLM Prometheus metric families. A missing metric is rendered as unavailable rather than zero; non-finite Prometheus scalar values (`NaN`, `+Inf`, and `-Inf`) are also rendered as unavailable.

| Area | Metric family / calculation | Reported as |
| --- | --- | --- |
| Time to first token | `vllm:time_to_first_token_seconds_bucket` histogram quantile | TTFT p95 |
| Time per output token | `vllm:inter_token_latency_seconds_bucket` histogram quantile | TPOT p95 |
| Queue delay | `vllm:request_queue_time_seconds_bucket` histogram quantile | Queue p95 (observational) |
| Queue delay | `rate(vllm:request_queue_time_seconds_sum[window]) / rate(vllm:request_queue_time_seconds_count[window])` | Queue mean |
| Prefill duration | `vllm:request_prefill_time_seconds_bucket` histogram quantile | Prefill p95 |
| Prompt size | `vllm:request_prompt_tokens_bucket` histogram quantile | Prompt tokens p95 |
| Scheduler activity | `vllm:num_requests_running`, `vllm:num_requests_waiting` | Running / waiting requests |
| KV occupancy | `vllm:kv_cache_usage_perc` | KV cache usage |
| Reclamation pressure | `rate(vllm:num_preemptions_total[window]) * 60` | Preemptions per minute |

The default query window is five minutes and can be changed with `--window`.

## Current diagnostic rules

v0.2 intentionally contains only three rule families:

| Finding | Required condition | Corroborating evidence used for confidence |
| --- | --- | --- |
| `QUEUE_PRESSURE` | TTFT p95 exceeds a supplied TTFT SLO | Waiting requests, KV usage at or above 85%, and TPOT remaining within its supplied SLO |
| `DECODE_PRESSURE` | TPOT p95 exceeds a supplied TPOT SLO | TTFT remaining within its supplied SLO increases confidence |
| `KV_CACHE_PRESSURE` | KV usage is at least 90% | Waiting requests, preemptions, or TTFT exceeding its supplied SLO |

`QUEUE_PRESSURE` does **not** score queue p95. The metric remains visible because it is useful for human investigation, but it is too sensitive to histogram buckets to be treated as direct per-request evidence. Similarly, KV occupancy alone cannot trigger `KV_CACHE_PRESSURE`.

## Limitations

- Prometheus histogram quantiles are estimates bounded by bucket resolution. In particular, a coarse queue-time histogram can report a p95 that is larger than an observed TTFT p95 without meaning that a single request spent more time queued than its TTFT.
- Latency findings depend on the SLOs supplied to the command. Without a TTFT or TPOT SLO, the corresponding latency rule cannot fire.
- Rate-based values need enough recent traffic and a suitable query window. An unavailable value is not proof of a healthy or idle system.
- These rules identify plausible bottleneck classes from vLLM telemetry; they do not establish root cause or replace profiling, traces, model-aware capacity tests, or GPU telemetry.
- The integration setup was validated with `Qwen/Qwen3-0.6B` and `vllm/vllm-openai:v0.29.0-cu129`. Metric names and semantics may change across vLLM releases; validate against your deployed version before operational use.

## Development

Install development dependencies and run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for issue and pull-request guidance, [CHANGELOG.md](CHANGELOG.md) for release notes, and [LICENSE](LICENSE) for the Apache-2.0 license.

The first historical regression harness is the read-only reproduction of [vLLM issue #48035](experiments/vllm-48035/README.md).
