# Competitive gap: Inference Doctor vs. vLLM Doctor

Comparison date: 2026-09-13. This is a product and implementation comparison between this repository at v0.2 and the public [`vllm-doctor/vllm-doctor`](https://github.com/vllm-doctor/vllm-doctor) project. Claims about our behavior are based on the current source, tests, and [validated experiments](experiment-log.md); claims about the public project are based on its repository and documentation.

## Overlap

Both projects occupy the same basic category: a read-only CLI that converts live vLLM telemetry into an evidence-backed diagnosis rather than another dashboard.

| Capability | Inference Doctor v0.2 | Public vLLM Doctor |
| --- | --- | --- |
| Primary input | Prometheus | Prometheus or direct vLLM `/metrics` scrape |
| Shared telemetry | Running/waiting requests, KV usage, TTFT p95, queue p95, preemptions, and decode-latency p95 | The same core families, plus broader throughput, outcome, and cache signals |
| Shared diagnoses | Queue/capacity pressure, decode/TPOT pressure, and KV-cache pressure | Queue pressure, high TPOT, and KV-cache pressure among ten rule families |
| Output | Rich terminal report and JSON | Compact/verbose terminal reports and versioned JSON |
| Explanation | Confidence, supporting evidence, explanation, and next experiment | Severity/confidence, evidence, likely cause, recommended configuration or operational checks, and a top-level assessment |
| Operating posture | One-shot, read-only diagnosis | One-shot or watch mode; diagnosis remains observational |

The overlap is substantial enough that competing on “diagnose vLLM metrics from the command line” alone would be weak positioning. The public project already presents a more complete version of that promise, including a single likely-bottleneck assessment and recommendations above the individual findings ([README](https://github.com/vllm-doctor/vllm-doctor), [assessment documentation](https://docs.vllm.doctor/assessment/)).

There is also a philosophical overlap: both correlate multiple signals instead of treating every metric independently. The implementations differ in an important way, however. Public vLLM Doctor reports high KV usage alone as a medium-confidence pressure finding, raising confidence when requests wait ([KV-cache rule](https://docs.vllm.doctor/rules/kv-cache-pressure/)). Our v0.2 rule deliberately requires waiting, preemption, or a TTFT SLO violation before high KV occupancy can produce a finding.

## Gaps

The public project is materially ahead as a general-purpose diagnostic product.

| Capability they have that we lack | Competitive significance |
| --- | --- |
| Ten documented rule families | They cover queue latency, preemption pressure, low throughput, error/abort rates, prefix-cache efficiency, replica imbalance, high TTFT, and high TPOT in addition to the three areas we cover ([rule catalog](https://docs.vllm.doctor/)). |
| Direct `/metrics` scrape | They can provide gauge/counter diagnosis without requiring Prometheus, while clearly disabling percentile rules that need historical buckets ([metrics documentation](https://docs.vllm.doctor/metrics/)). We require Prometheus. |
| Root-cause assessment | They combine findings into one ranked category such as queue saturation, long prefill, decode, replica imbalance, or idle. We emit an ordered list without a top-level synthesis. |
| Per-model and per-replica analysis | They filter by model and retain pod-level values for replica-imbalance analysis. Our collector sums or takes a maximum across every matching series and has no deployment filter. |
| More operational signals | They collect prefill/decode throughput, request completion reasons, error and abort rates, generation-token length, and prefix-cache hit/query counters. We collect prefill-duration and prompt-token histograms but have no corresponding rule. |
| Configurable rule thresholds | They support TOML configuration for rule thresholds and target metadata ([configuration](https://docs.vllm.doctor/configuration/)). We expose only TTFT/TPOT SLOs and the query window; other thresholds are fixed. |
| History and watch operation | They persist runs in local SQLite, list/show history, and save only state changes during watch mode ([history documentation](https://docs.vllm.doctor/commands/history/)). We are one-shot and stateless. |
| Automation contract | Their JSON has a documented schema version, metadata, notices, assessment, and stable rule IDs ([JSON documentation](https://docs.vllm.doctor/json-output/)). Our JSON is a direct serialization of current internal models. |
| Distribution and release maturity | They publish a Rust crate, install script, and prebuilt container. We currently support editable Python installation and provide an integration compose stack, not a packaged runtime image. |

The four requested optimization areas are open in both projects, with one important adjacent competitor: vLLM itself already ships an auto-tuning benchmark workflow.

| Area | Inference Doctor v0.2 | Public vLLM Doctor | Gap assessment |
| --- | --- | --- | --- |
| Automated experimentation | Findings suggest a next experiment, but nothing provisions, restarts, loads, measures, or compares configurations. Our validation harness was operated manually. | Its README explicitly says it is not a benchmark runner and recommends pairing it with GuideLLM for workload generation. | Neither closes the loop between diagnosis and a reproducible experiment. |
| Cost optimization | No price, energy, utilization-cost, or cost-per-token model. | No documented cost objective or provider-price integration. | Open space, but our single-GPU L4 validation is not enough evidence to claim a cost optimizer yet. |
| Configuration search | No candidate generation, parameter-space definition, trial scheduler, or objective function. | TOML changes diagnostic thresholds; recommendations name serving flags, but the project does not search vLLM configurations. | Open between these two projects, but official vLLM already searches `max-num-seqs`, `max-num-batched-tokens`, GPU-memory utilization, and sustainable request rate ([official auto-tune workflow](https://github.com/vllm-project/vllm/tree/main/benchmarks/auto_tune)). |
| Closed-loop tuning | No actuator, trial lifecycle, acceptance gate, rollback, or convergence logic. | Watch mode observes and stores state changes; it does not apply recommendations or verify their effects. | A genuine gap, but it carries production-safety and causal-attribution risk. |

There are two narrower technical gaps worth preserving rather than copying blindly:

- Public vLLM Doctor has a queue-latency rule that can fire from queue p95 alone at low confidence ([queue-latency rule](https://docs.vllm.doctor/rules/queue-latency/)). Our experiments showed vLLM's coarse first queue bucket producing an apparent 285 ms p95 while TTFT was roughly 39 ms. We therefore retain queue p95 for observation but exclude it from confidence scoring.
- Public vLLM Doctor already distinguishes long prefill in its top-level assessment using optional prompt/generation length evidence. We collect prefill-duration p95 and prompt-token p95, and E011 proved those observations respond to a prefill-heavy workload, but we intentionally have no prefill diagnostic rule yet.

## Possible differentiation

The strongest differentiated direction is an **experiment-backed optimizer**, not another larger static rule catalog:

```text
observe -> form a falsifiable hypothesis -> run a bounded trial
        -> measure SLO, throughput, stability, and cost -> accept or roll back
```

### 1. Automated, diagnosis-directed experiments

Turn each recommendation into a reproducible experiment contract: baseline configuration, one controlled change, fixed workload, warm-up policy, sample count, measurement window, expected telemetry movement, pass/fail criteria, and an immutable result artifact. The current E011-E013 sequence is the useful seed:

- E011 demonstrated that large prompts move prompt-token and prefill p95 observations without inventing a diagnosis.
- E012 drove KV occupancy above 90% with no waiting or preemption and correctly remained healthy.
- E013 kept the constrained cache and increased sequence pressure until waiting appeared, correctly producing `KV_CACHE_PRESSURE`.

That pairwise “change one pressure variable and verify the predicted signal” approach is more defensible than adding rules from intuition. It also fills the workflow gap left when public vLLM Doctor hands workload generation to GuideLLM.

### 2. SLO-constrained cost optimization

Compare configurations and hardware using cost per successful million input/output tokens, with TTFT/TPOT SLO compliance and error rate as hard constraints. Include GPU-hour price, achieved goodput, failed requests, startup/amortization time, and idle capacity. The result should be a Pareto frontier rather than one universal “cheapest” answer.

This is plausible differentiation, not yet a validated capability. It requires repeated measurements across multiple GPU classes, models, prompt/output distributions, and prices. The existing L4/Qwen experiment proves the integration path, not the cost model.

### 3. Diagnosis-guided configuration search

Use observed bottleneck class to narrow a guarded parameter space before benchmarking. For example, KV pressure could prioritize `max-num-seqs`, `max-num-batched-tokens`, `gpu-memory-utilization`, `kv-cache-memory-bytes`, and `max-model-len`; decode pressure could prioritize concurrency, quantization, and parallelism choices. Search should optimize measured goodput under SLO and stability constraints, record every rejected trial, and treat OOM/startup failure as a result rather than silently retrying.

This should complement, not duplicate, vLLM's official grid-style auto-tuner. A differentiated search would use diagnosis to choose the next experiment, compare baseline and candidate telemetry, and stop when evidence no longer supports the hypothesis.

### 4. Guarded closed-loop tuning

A credible closed loop needs stronger controls than “apply the recommendation”:

- operate on a disposable replica or canary, never mutate the only production server
- allowlist mutable flags and enforce time, trial-count, spend, and failure budgets
- health-check startup, warm the model, run a fixed workload, and wait for valid Prometheus samples
- reject regressions in correctness, errors, TTFT, TPOT, or goodput even when one target metric improves
- preserve baseline configuration and automatically roll back failed candidates
- require explicit approval before promoting a winning configuration

Public vLLM Doctor's watch mode is an observation loop, not this control loop. Building the guarded experiment lifecycle first would create the evidence base needed for later automation without prematurely granting a diagnostic rule authority to change production.

The practical product wedge is therefore: **reproducible experiments that prove or falsify a vLLM configuration hypothesis, then optimize cost under explicit SLOs**. Rule breadth, history, direct scrape, and general incident triage are already better served by the public project today.
