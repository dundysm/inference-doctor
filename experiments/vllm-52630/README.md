# vLLM #52630 Pooling Regression Harness

This harness reproduces the published pooling/embeddings throughput comparison
from [vLLM issue #52630](https://github.com/vllm-project/vllm/issues/52630).
It compares vLLM `0.19.0` with Torch `2.10.0` against vLLM `0.21.0` with
Torch `2.11.0`, using `BAAI/bge-m3` on one continuously leased H100.

The RunPod target is an H100 SXM 80 GB because H100 PCIe is currently
unavailable. That is recorded as an explicit hardware deviation from the
issue's H100 PCIe 80 GB host.

This is an experiment harness only. It does not change Inference Doctor
diagnostic rules, the product comparison command, or the #48035 artifacts.

## Layout

```text
vllm-52630/
  aggregate.py
  benchmark_pooling.py
  compare_results.py
  dataset_manifest.json
  experiment.json
  harness_common.py
  README.md
  run_same_gpu.py
  RUNPOD.md
```

## Environments

The two isolated environments are:

```text
/workspace/envs/v0190  vLLM 0.19.0, Torch 2.10.0
/workspace/envs/v0210  vLLM 0.21.0, Torch 2.11.0
```

The issue identifies the Torch 2.10 to 2.11 boundary as the likely transition.
The official release images are the reference for the bundled versions. The
RunPod procedure uses version-specific Python environments inside one generic,
continuously running CUDA container so the physical GPU is never released.

The exact wheel preflight commands are documented in `RUNPOD.md`. The old
release wheel indexes must be checked on the target before a paid run; the
harness does not silently substitute a different Torch or vLLM build.

## Workload

- Dataset: `mteb/RuBQRetrieval`, `corpus` split.
- Selection: rows `0..4095`, dataset order, no shuffle.
- The committed manifest freezes dataset commit
  `7229066ed7a617ef50aad51c5a2d95957eaee537` and contains the selected-index
  hash. Each run records the datasets fingerprint and selected document ID hash.
- Exactly 32 documents are sent in each `POST /v1/embeddings` request.
- Four warmup batches run before every measured concurrency point.
- Measured concurrency is `16`, `32`, and `64`.
- Each point runs three repetitions by default.
- Each request records timing, status, payload size, document IDs, text length,
  request hash, returned embedding count, and errors.

The primary metric is texts/sec, not token throughput. Each point reports
successful/failed requests, total texts, wall-clock duration, request
throughput, texts/sec, median request latency, p95 request latency, and server
errors.

## Local commands after raw data exists

Aggregate one version independently:

```powershell
python aggregate.py `
  --input-dir results/baseline `
  --output results/baseline/aggregate.json
```

Run a same-version control:

```powershell
python compare_results.py `
  --baseline results/baseline/repetition-001/normalized.json `
  --candidate results/baseline/repetition-002/normalized.json `
  --same-version
```

Run the cross-version comparison:

```powershell
python compare_results.py `
  --baseline results/baseline/aggregate.json `
  --candidate results/candidate/aggregate.json `
  --json `
  --output results/comparisons/baseline-vs-candidate.json
```

## Comparison policy

Same-version controls use absolute percentage variation with a default 5%
stability tolerance. A positive or negative movement beyond that tolerance is
`UNSTABLE`. The control must be `PASS` before an A/B result is valid. The
top-level runner evaluates all three repetition pairs (1/2, 1/3, and 2/3) for
each version; every pair must pass.

The preregistered historical-regression point is concurrency 32. Concurrency
16 and 64, plus the best stable plateau, remain supporting evidence but cannot
alone mark issue #52630 reproduced.

Cross-version texts/sec keeps directional regression semantics. A candidate
throughput drop at or above the configured 10% failure threshold is `FAIL`; an
improvement is `PASS`. Missing values are `WARN` and never fabricated.

## Expected result shape

The issue reports approximately 3,092 texts/sec for vLLM 0.19.0 and 2,537
texts/sec for vLLM 0.21.0 at concurrency 32, or about an 18% drop. Those values
are expectations only and are not hard-coded into pass/fail logic. A/A and B/B
controls must be stable before the A/B result is interpreted.

## Methodological differences

- Hardware is H100 SXM 80 GB here instead of the issue's H100 PCIe 80 GB.
- The issue used official vLLM Docker tags and a shared host with unrelated
  services using about 28 GB of GPU memory. This harness uses isolated Python
  environments inside one generic outer Pod and requires the same GPU UUID for
  both versions.
- The issue's dataset is frozen to commit
  `7229066ed7a617ef50aad51c5a2d95957eaee537`; the datasets fingerprint and
  selected document ID hash are recorded when the pinned dataset is loaded.
- The issue reports one warmup pass before each measurement; this harness makes
  that explicit as four warmup batches before each concurrency point.

See `RUNPOD.md` for the disposable-key/Paramiko execution and teardown plan.
