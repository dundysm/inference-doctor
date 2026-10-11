# vLLM `bench serve` example

Synthetic results in the exact format `vllm bench serve --save-result` writes, five
runs per side. No conversion step is needed: `compare-runs` detects the format.

The runs were "recorded" with `--metadata gpu="NVIDIA L4" vllm_version=...`, so the GPU
is compared between the two sides. The candidate is about 1.8% lower on output-token
throughput and about 18% higher on mean time to first token.

```bash
inference-doctor compare-runs   --baseline examples/vllm-bench-serve/baseline   --candidate examples/vllm-bench-serve/candidate
```

Throughput (the default metric): `PASS`, the whole interval is below 10%.

```bash
inference-doctor compare-runs   --baseline examples/vllm-bench-serve/baseline   --candidate examples/vllm-bench-serve/candidate   --metric ttft_mean_ms
```

Mean TTFT: `FAIL`, the whole interval is past 10%.

This data exists only to demonstrate the input format.
