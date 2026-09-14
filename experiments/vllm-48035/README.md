# Reproduction harness for vLLM issue #48035

This experiment tests the FP16 version-to-version decode-throughput regression reported in [vLLM issue #48035](https://github.com/vllm-project/vllm/issues/48035): approximately 136–137 decode tokens/s on vLLM 0.19.1 versus 98 tokens/s on vLLM 0.24.0 for a single RTX 4090 at sequential batch size 1.

The harness does not diagnose, tune, or alter either server. It runs a fixed workload, preserves raw request records, normalizes each repetition for `inference-doctor compare`, aggregates repetitions, and runs two same-version controls plus the cross-version comparison.

## Fixed experiment

- Baseline: `vllm/vllm-openai:v0.19.1`
- Candidate: `vllm/vllm-openai:v0.24.0`
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Hardware: exactly one visible NVIDIA RTX 4090, tensor parallelism 1
- Workload: 10 committed prompts × seeds `17`, `29`, and `43`
- Request shape: sequential chat completions, concurrency 1, `max_tokens=1024`, temperature 0.7, top-p 0.95
- Repetitions: 3 by default; one full-shape warmup request is excluded from each repetition
- Throughput fail threshold: 10%

All serving arguments are recorded in [experiment.json](experiment.json). The Docker driver follows the [vLLM 0.19.1 Docker documentation](https://docs.vllm.ai/en/v0.19.1/deployment/docker/) and passes `--model Qwen/Qwen3-4B-Instruct-2507` plus the issue's engine arguments to the official image entrypoint.

## GPU-host requirements

- Linux NVIDIA host with one visible RTX 4090 and a driver compatible with both images
- Docker Engine plus NVIDIA Container Toolkit
- Python 3.12 and this repository installed with `python -m pip install -e .`
- Network and disk capacity to pull both images and the model
- Port 8000 free on loopback

The driver refuses a non-4090 or more than one visible GPU, refuses to reuse its fixed container name, binds vLLM only to `127.0.0.1:8000`, and always stops the experiment container in a `finally` block. It retains a named Hugging Face cache volume to avoid downloading the model twice.

## Run the complete experiment

From the repository root on the GPU host:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python experiments/vllm-48035/run_experiment.py
```

To select a durable output location or override the repetition count:

```bash
python experiments/vllm-48035/run_experiment.py \
  --output-dir /data/vllm-48035/run-001 \
  --repetitions 3
```

The driver runs baseline first, then candidate. Each same-version control compares repetition 1 with repetition 2 using absolute variation in either direction and the configured `stability_tolerance_percent` (5% by default). All completed repetitions contribute to the cross-version aggregates. It creates:

```text
run-001/
  run.json
  baseline/
    server.json
    server.log
    repetition-001/{raw.json,normalized.json}
    repetition-002/{raw.json,normalized.json}
    repetition-003/{raw.json,normalized.json}
    aggregate.normalized.json
  candidate/
    server.json
    server.log
    repetition-001/{raw.json,normalized.json}
    repetition-002/{raw.json,normalized.json}
    repetition-003/{raw.json,normalized.json}
    aggregate.normalized.json
  comparisons/
    baseline-vs-baseline.{json,txt}
    candidate-vs-candidate.{json,txt}
    baseline-vs-candidate.{json,txt}
  summary.json
```

Raw JSON is checkpointed after every measured request. An interrupted repetition remains marked `"complete": false` and is rejected by aggregation.

## Run individual stages

With the baseline vLLM server already listening on port 8000, collect the configured three-repetition set directly. This mode does not invoke Docker. It verifies `/health`, requires exactly one visible RTX 4090, and records its UUID, driver, memory, Python/runtime metadata, and selected RunPod environment fields in every raw and normalized result:

```bash
python experiments/vllm-48035/benchmark_client.py \
  --server-url http://localhost:8000 \
  --version-label 0.19.1 \
  --output-dir /workspace/vllm-48035-results/baseline
```

`--version-label` identifies baseline or candidate from `experiment.json`, including the corresponding image label. The older `--base-url`, `--vllm-version`, `--variant`, `--image`, and `--repetition` arguments remain available for collecting a specific repetition.

After both version directories contain raw repetitions, aggregate and run all comparisons independently:

```bash
python experiments/vllm-48035/compare_results.py \
  --baseline-dir /workspace/vllm-48035-results/baseline \
  --candidate-dir /workspace/vllm-48035-results/candidate \
  --output-dir /workspace/vllm-48035-results/comparisons
```

This regenerates each repetition's normalized file from its raw record, writes each aggregate, and executes both controls plus the cross-version comparison using the thresholds in `experiment.json`. A/A and B/B use same-version stability semantics; A/B keeps directional regression semantics. It refuses to execute the cross-version comparison when either aggregate lacks `environment.gpu_uuid` or the UUIDs differ.

Aggregation alone remains independently runnable:

```bash
python experiments/vllm-48035/aggregate.py \
  --input-dir /workspace/vllm-48035-results/baseline \
  --output /workspace/vllm-48035-results/baseline/aggregate.normalized.json
```

For the complete no-nested-Docker procedure, see [RUNPOD.md](RUNPOD.md).

The expected validation shape is:

```text
baseline vs baseline   PASS when STABLE
candidate vs candidate PASS when STABLE
baseline vs candidate  FAIL if the historical regression reproduces
```

A same-version control reported as `UNSTABLE` invalidates the cross-version conclusion. A cross-version PASS is recorded as “not reproduced”; the driver does not modify prompts, seeds, sampling, thresholds, or serving settings in response.

## Measurement definitions

- TTFT is wall time from sending the streaming HTTP request until the first non-empty content delta.
- Total latency is wall time until the stream closes.
- Decode duration is `total latency - TTFT`.
- TPOT is `decode duration / (output tokens - 1)` when usage reports at least two output tokens.
- Decode tokens/s is `(output tokens - 1) / decode duration`.
- Per-repetition output-token throughput is the mean of per-request decode tokens/s.
- Request throughput is successful measured requests divided by measured repetition wall time.
- Aggregate standard deviation is the sample standard deviation of repetition-level mean decode throughput. Coefficient of variation is that standard deviation divided by the run-to-run mean.

The client requests streaming usage from vLLM. If a version does not return `completion_tokens`, the request remains successful but its TPOT and decode-throughput values are unavailable; the comparison reports those missing metrics as WARN rather than inventing a token count.

## Differences from the public issue

The public issue specifies the versions, model, GPU, serving flags, 10-prompt × 3-seed shape, sequential execution, 1024-token limit, excluded warmup, and approximate results. It does not publish its prompts, seeds, sampling parameters, source code, or exact timing formulas. This harness therefore commits a new fixed prompt set, seeds, sampling settings, and explicit timing definitions.

Other known differences:

- The issue host used driver `610.43.02` and a GPU shared with a quiet embedding/RAG stack. This harness records the actual driver and requires one visible GPU; it does not recreate that co-tenancy.
- The issue added `--enable-auto-tool-choice --tool-call-parser hermes` to v0.24.0 for tool-bearing requests. These fixed prompts contain no tools, so those candidate-only flags are intentionally omitted.
- The issue also measured an official FP8 checkpoint. This experiment tests only the stated FP16 v0.19.1-to-v0.24.0 regression.
- The issue says results were reproduced across repeated runs but does not state its repetition count or whether the server restarted between runs. This harness defaults to three repetitions per version on one server process, with a warmup before every repetition.
- Image tags are pulled at run time. `server.json` records the resolved local image ID and repository digests so an archived run does not rely on the tag alone.

These differences mean a non-reproduction is informative but not proof that the reported regression never existed. Do not change the committed workload after seeing results; create a separately named follow-up experiment for any revised hypothesis.
