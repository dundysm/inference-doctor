# Inference Doctor

**Stop shipping inference regressions.**

Inference Doctor is a CLI + GitHub Action that compares repeated LLM inference benchmarks and tells you whether a change is safe to ship.

```text
PASS          candidate is within your regression threshold
FAIL          candidate has a measurable performance regression
INCONCLUSIVE  the benchmark is too noisy or invalid to trust
```

The key difference from a simple benchmark threshold: **Inference Doctor checks whether the measurement itself is trustworthy before making a merge decision.**

## 5-minute quickstart

Python 3.11+ is required.

```bash
git clone https://github.com/dundysm/inference-doctor.git
cd inference-doctor
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

Run the checked-in example. **No GPU is required** because the benchmark results are already included.

```bash
inference-doctor compare-runs \
  --baseline examples/quickstart/baseline \
  --candidate examples/quickstart/candidate \
  --metric texts_per_second
```

You should get a `PASS` result. That is the core workflow: feed Inference Doctor repeated baseline and candidate results, and it returns a CI-safe decision.

## Why this exists

A PR can pass every functional test and still make inference slower after a vLLM, PyTorch, CUDA, model, batching, KV-cache, quantization, or serving-config change.

A naive gate compares two numbers. That is dangerous because GPU benchmarks can be noisy.

Inference Doctor first checks repeatability, then evaluates the regression:

- baseline CV must be at or below 5% by default
- candidate CV must be at or below 5% by default
- failed requests or incomparable benchmark environments make the result `INCONCLUSIVE`
- a stable 10% directional regression is `FAIL` by default
- apparent outliers are reported but remain in the official calculation

Example outcomes:

```text
PASS
throughput: -2.1%
measurement quality: stable
```

```text
FAIL
throughput: -13.4%
measurement quality: stable
regression threshold: 10%
```

```text
INCONCLUSIVE
throughput: -7.3%
baseline CV: 8.9%
reason: baseline measurements are too noisy to trust
```

## GitHub Actions

After your benchmark job produces repeated normalized results, use Inference Doctor as the merge gate:

```yaml
- name: Check inference performance
  uses: dundysm/inference-doctor@main
  with:
    baseline: artifacts/baseline
    candidate: artifacts/candidate
    metric: output_token_throughput
    stability-cv-percent: "5"
    regression-percent: "10"
```

The Action writes a readable GitHub step summary and exposes the result plus baseline/candidate means, delta, and CVs as outputs.

Inference Doctor intentionally **does not provision GPUs**. Run your preferred benchmark on your existing GPU/self-hosted runner, save normalized results, then let Inference Doctor make the measurement-quality + regression decision.

See [docs/github-actions.md](docs/github-actions.md).

## CLI

### Compare repeated runs — recommended

```bash
inference-doctor compare-runs \
  --baseline artifacts/baseline \
  --candidate artifacts/candidate \
  --metric output_token_throughput
```

Supported metrics include output-token throughput, request throughput, texts/sec, TTFT, TPOT, successful requests, and failed requests.

| Result | Meaning | Exit code |
| --- | --- | ---: |
| `PASS` | Stable measurements; threshold not crossed | 0 |
| `FAIL` | Stable measurements; regression threshold crossed | 1 |
| `INCONCLUSIVE` | Measurement quality/comparability is not good enough to decide | 2 |
| input error | Invalid schema/files/arguments | 3 |

Use `--json` for machine-readable output.

See [docs/measurement-quality.md](docs/measurement-quality.md). The [sanitized H100 reliability example](examples/inference-ci-reliability-001/README.md) shows a real `INCONCLUSIVE` result caused by unstable baseline measurements.

### Compare two normalized files

For simple one-to-one comparisons:

```bash
inference-doctor compare \
  --baseline baseline.json \
  --candidate candidate.json
```

See [docs/upgrade-guard.md](docs/upgrade-guard.md).

## Secondary feature: diagnose a live vLLM service

Inference Doctor also keeps its original read-only Prometheus diagnostic mode for queue/capacity pressure, decode pressure, and meaningful KV-cache pressure.

```bash
inference-doctor diagnose \
  --prometheus http://localhost:9090 \
  --ttft-slo-ms 400 \
  --tpot-slo-ms 50
```

It reads Prometheus only and does not modify vLLM, Prometheus, or traffic. High GPU/KV utilization alone is not treated as an incident; findings require corroborating evidence.

See [docs/example-reports.md](docs/example-reports.md) and [docs/local-integration.md](docs/local-integration.md).

## Scope

The product is deliberately narrow right now:

> **Unit tests protect correctness. Inference Doctor protects inference performance.**

Not included yet: hosted GPU provisioning, dashboard/SaaS, optimizer, Kubernetes operator, SGLang support, or automatic root-cause analysis.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md), and [LICENSE](LICENSE).

Historical experiments that informed the measurement-quality design live under `experiments/`.
