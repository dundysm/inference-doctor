# Product direction

Inference Doctor is a merge gate for LLM inference performance.

The user question is simple:

> Did this change make inference performance worse, and is the benchmark trustworthy enough to decide?

The product returns exactly three decision states: `PASS`, `FAIL`, or `INCONCLUSIVE`.

Current focus:

- repeated baseline vs candidate comparisons, decided by a confidence interval
- validity checks (environment, failed requests, missing metrics, repetitions) before any decision
- reading the result files benchmark tools already write (vLLM, AIPerf, GuideLLM, SGLang),
  plus a small normalized format for anything else
- CLI + GitHub Action integration

Intentionally out of scope for now:

- running benchmarks or provisioning GPUs
- dashboards/SaaS or a history of past runs
- automatic tuning
- Kubernetes operators
- automated root-cause diagnosis (`diagnose` reports patterns, not causes)
