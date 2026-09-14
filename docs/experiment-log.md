# v0.2 Integration Evidence

This note records the useful, reproducible outcomes from the v0.2 integration validation. It intentionally excludes account details, pod identifiers, regions, local paths, tokens, and transient container logs.

## Environment

- GPU class: NVIDIA L4 on a disposable cloud pod
- vLLM image: `vllm/vllm-openai:v0.29.0-cu129`
- model: `Qwen/Qwen3-0.6B`
- Prometheus: scraping vLLM `/metrics` every 5 seconds
- traffic: repeatable asynchronous OpenAI-compatible completion requests
- collector window: two to five minutes, depending on the experiment

The public reproduction steps are in [local-integration.md](local-integration.md). Values below are representative observations from that validation, not performance claims for every L4 or vLLM deployment.

## Selected outcomes

| Experiment | Workload | Observed telemetry | Expected result | Actual result |
| --- | --- | --- | --- | --- |
| E001 healthy baseline | 40 requests, concurrency 4, short prompts, 16 output tokens | TTFT p95 38.9 ms; TPOT p95 9.6 ms; no waiting requests; no preemptions | No finding | No finding |
| E003 queue/capacity | 240 requests, concurrency 64, medium prompts, 64 output tokens; TTFT SLO 100 ms | TTFT p95 194.2 ms; TPOT p95 9.6 ms | `QUEUE_PRESSURE` | `QUEUE_PRESSURE` |
| E006 decode | 160 requests, concurrency 32, short prompts, 512 output tokens; TPOT SLO 10 ms | TTFT p95 59.0 ms; TPOT p95 23.4 ms | `DECODE_PRESSURE` | `DECODE_PRESSURE` |
| E011 prefill observability | 24 requests, concurrency 4, large prompts, 24 output tokens | Prefill p95 486.7 ms; prompt tokens p95 19,500 | No new prefill finding | No finding |
| E012 high KV, healthy | One long sequence with a constrained KV cache | KV usage 90.1%; zero waiting requests; zero preemptions | No finding | No finding |
| E013 KV pressure | Four long concurrent sequences using the same constrained cache | KV usage 92.2%; three waiting requests; zero preemptions | `KV_CACHE_PRESSURE` | `KV_CACHE_PRESSURE` |

## Queue histogram interpretation

The healthy baseline reported queue p95 of about 285 ms while TTFT p95 was about 39 ms. Raw queue samples fell into the first available 0.3-second queue histogram bucket, while TTFT used much finer buckets around 0.02 to 0.04 seconds. Prometheus interpolates each histogram quantile independently, so these p95 values are not paired timings from the same request.

The apparently larger queue p95 therefore reflects coarse queue histogram buckets, not evidence that each request waited longer than its TTFT. In v0.2, queue p95 remains displayed as observational data and does not increase queue-pressure confidence.

## Non-finite values

When a short query window lacked usable rate samples, Prometheus returned non-finite scalar results for some histogram-derived queries. v0.2 converts `NaN`, `+Inf`, and `-Inf` to unavailable (`null` in JSON and `N/A` in the terminal report), rather than exposing Python `nan` values.
