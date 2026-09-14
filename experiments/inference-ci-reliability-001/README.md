# inference-ci-reliability-001

This is a separate, preregistered benchmark-reliability experiment for one
question:

> Does longer-duration, interleaved measurement make a 10% inference-throughput
> regression decision trustworthy at concurrency 32?

The completed vLLM #52630 reproduction remains officially INVALID. This
experiment does not reinterpret that result and does not modify Inference
Doctor diagnostic behavior.

## Frozen workload

- One physical NVIDIA H100 80GB HBM3, with the same GPU UUID required for every block.
- `BAAI/bge-m3` pooling server.
- RuBQ corpus at commit `7229066ed7a617ef50aad51c5a2d95957eaee537`.
- First 4096 documents in dataset order.
- 32 documents per `/v1/embeddings` request, 128 batches per workload cycle.
- Concurrency 32 only.
- Baseline: vLLM `0.19.0`, Torch `2.10.0`, `/workspace/envs/v0190`.
- Candidate: vLLM `0.21.0`, Torch `2.11.0`, `/workspace/envs/v0210`.
- Same server arguments as #52630.

## Frozen schedule and windows

The exact schedule is:

```text
A B B A
A B B A
A B B A
```

where A is baseline and B is candidate. It produces six measured blocks per
version.

For every block, the runner:

1. records the GPU UUID;
2. starts the selected vLLM server with its version-specific compile caches;
3. waits for `/health`;
4. performs exactly 300 seconds of warmup traffic, excluded from measurement;
5. continuously cycles the 4096-document workload for exactly 300 seconds;
6. records every measured request and block summary;
7. stops the server and verifies vLLM process and GPU-memory cleanup;
8. records the GPU UUID again and invalidates the run if it changed.

The measurement window admits repeated workload cycles until the five-minute
deadline. In-flight requests are drained and preserved, while throughput is
denominated by the configured five-minute window. Warmup requests are stored
separately and never enter the measured summaries.

## Preregistered decisions

For each version, the six `texts_per_second` block values are summarized with
mean, median, sample standard deviation, CV, minimum, maximum, successful
texts, and failed requests. No outlier is deleted.

The controls are stable only when:

```text
baseline CV <= 5%
candidate CV <= 5%
no failed requests
same GPU UUID and frozen configuration throughout
```

The candidate regression is:

```text
(candidate_mean - baseline_mean) / baseline_mean
```

Decision:

- stable controls and delta `<= -10%`: `FAIL / REGRESSION`;
- stable controls and delta `> -10%`: `PASS`;
- either control above 5% CV, failed requests, or GPU/configuration mismatch:
  `INCONCLUSIVE`.

The report also includes a deterministic 10,000-sample empirical bootstrap
across the six measured blocks. It is supporting evidence only and does not
replace the CV plus 10% rule.

## Execution

Run only on an already prepared GPU host with the two isolated environments,
the existing immutable dataset manifest, and `nvidia-smi` available. This
command provisions nothing:

```bash
/workspace/envs/v0190/bin/python \
  /workspace/inference-doctor/experiments/inference-ci-reliability-001/run_experiment.py \
  --output-dir /workspace/inference-ci-reliability-001-results
```

To add best-effort host telemetry without changing the workload or decision
semantics, pass `--telemetry-dir /workspace/inference-ci-reliability-001-results/telemetry`.
The runner records one JSON file per block at approximately five-second
intervals. Missing `nvidia-smi`, `/proc`, or optional `psutil` data is stored
as unavailable and does not fail the benchmark.

The output contains `run.json`, `summary.json`, and one directory under
`blocks/` for every scheduled block. Each block preserves raw warmup and
measurement requests, server launch metadata, server logs, GPU UUIDs, and
post-stop memory evidence.
