# Cost doctor (experimental)

`inference-doctor cost` turns the numbers a serving engine already exposes into dollars:
$/1M tokens, an estimate of the money spent on idle capacity, and a ranked list of fixes.
`inference-doctor cost-compare` gates a change on its $/1M output tokens with the Welch
log-ratio interval used by `compare-runs` (PASS, FAIL or INCONCLUSIVE; exit 0/1/2). In the
Action, set `cost-gpu` (or `cost-price`) and optionally `cost-gpus` and
`cost-max-increase-percent`.

```
inference-doctor cost --demo                                   # bundled sample, no GPU
inference-doctor cost --metrics t0.prom --metrics t1.prom --gpu H100-SXM --gpus 8 --replicas 4 --tp 2
inference-doctor cost --url http://localhost:8000/metrics --seconds 60 --gpu L40S --gpus 2
inference-doctor cost --runs results/ --gpu A100-80G --gpus 1 --api-model gpt-oss-120b
inference-doctor cost-compare --baseline base/ --candidate cand/ --gpu H100-SXM --max-increase-pct 5
```

Inputs: vLLM (`vllm:*`, V0 and V1 names) and SGLang (`sglang:*`) Prometheus text, plus
`DCGM_FI_DEV_GPU_UTIL` when the scrape includes it; or any benchmark directory the gate reads.

## Measured vs heuristic

| Output | Basis |
| --- | --- |
| prompt/output tokens per second | measured: counter delta over the window between two scrapes |
| queue, running requests, KV usage, prefix hit rate, GPU util | measured gauges, averaged over the two scrapes |
| $/1M tokens, monthly cost | measured throughput x your `--price`, or the cited list price |
| occupancy | max(running / (replicas x max-num-seqs), KV usage) |
| $ wasted / month | **heuristic**: monthly cost x (1 - occupancy / 80%), zero while requests queue |
| right-size | **heuristic**: GPUs needed to reach 80% occupancy, rounded to `--tp` |
| raise max-num-seqs | **heuristic**: queueing with KV < 60% |
| prefix caching | **heuristic**: 20% of the prompt-token share of cost, when caching is off |
| FP8 | **heuristic**: assumes a 1.3x throughput gain on FP8-capable GPUs |
| switch GPU | list prices x an **assumed** relative throughput (`rel_throughput` in the table) |
| hosted API | measured traffic x cited list price per 1M tokens |

A short window misses peaks: validate any right-size with p95 TTFT before acting.
Savings overlap and are not additive.

## Prices

`src/inference_doctor/data/gpu_prices.json` and `api_prices.json`, copied on 2026-10-31 from
https://www.runpod.io/pricing (Secure Cloud on-demand, page dated 2026-10-08) and
https://www.together.ai/pricing (GPU Clusters on-demand; serverless per-1M-token prices).
Prices move; pass `--price` with what you actually pay.

## Validated on a real server

`tests/fixtures/real_vllm_0_32/` holds two real scrapes of vLLM 0.32.0 (Qwen2.5-0.5B-Instruct,
one RTX A5000, 2026-10-31). Every metric the cost view reads was present under the expected
name; plain `/metrics` has no scrape time, so with two `--metrics` files and no `--seconds`
the window falls back to the files' modification times. An unlisted GPU works with `--price`
(FP8 and switch-GPU advice are skipped since its capabilities are unknown).

```
$ inference-doctor cost --metrics t0.prom --metrics t1.prom --seconds 60 --gpu "RTX A5000" --price 0.27
  throughput        13,719 prompt + 4,240 output tok/s   [measured]
  cost / 1M tokens  $0.004 all, $0.018 output
```
SGLang names are still checked only against synthetic fixtures.
