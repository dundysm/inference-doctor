# Product direction

Inference Doctor is a merge gate for LLM inference performance.

The user question is simple:

> Did this change make inference performance worse, and is the benchmark trustworthy enough to decide?

The product returns exactly three decision states: `PASS`, `FAIL`, or `INCONCLUSIVE`.

Current focus:

- repeated baseline vs candidate comparisons
- measurement-quality checks before regression decisions
- CLI + GitHub Action integration
- simple, auditable normalized benchmark inputs

Intentionally out of scope for now:

- hosted GPU provisioning
- dashboards/SaaS
- automatic tuning
- Kubernetes operators
- automated root-cause diagnosis
- broad multi-runtime support
