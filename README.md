# Inference Doctor

[![CI](https://github.com/dundysm/inference-doctor/actions/workflows/inference-ci.yml/badge.svg)](https://github.com/dundysm/inference-doctor/actions/workflows/inference-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Stop shipping inference regressions.**

Inference Doctor is a CLI + GitHub Action that compares repeated LLM inference benchmarks and turns them into a merge-gate decision:

```text
PASS          candidate is within your regression threshold
FAIL          candidate has a measurable performance regression
INCONCLUSIVE  the benchmark is too noisy or invalid to trust
```

The key difference from a simple benchmark threshold: **Inference Doctor checks whether the measurement itself is trustworthy before making a decision.**

## Who this is for

Use Inference Doctor if you maintain self-hosted inference and regularly change things like vLLM/PyTorch/CUDA versions, model revisions, batching, KV-cache settings, quantization, tensor parallelism, or serving configuration.

Your benchmark harness stays yours. Inference Doctor sits after it and answers: **did this change regress performance, and is the evidence trustworthy enough to decide?**

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

You should get `PASS`. That is the core workflow: repeated baseline results + repeated candidate results -> `PASS`, `FAIL`, or `INCONCLUSIVE`.

Want to use your own benchmark output? Start with the [normalized benchmark format](docs/benchmark-format.md).

## Why this exists

A PR can pass every functional test and still make inference slower. A naive performance gate compares two numbers, but GPU benchmarks can be noisy enough to create false regressions or false confidence.

Inference Doctor checks repeatability first, then evaluates the regression. By default:

- baseline CV must be at or below 5%
- candidate CV must be at or below 5%
- failed requests or incomparable benchmark environments make the result `INCONCLUSIVE`
- a stable 10% directional regression is `FAIL`
- apparent outliers are reported but remain in the official calculation

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

The Action writes a readable GitHub step summary and exposes the result, baseline/candidate means, delta, and CVs as outputs. `FAIL` and `INCONCLUSIVE` both fail the Action step so noisy evidence cannot silently approve a change.

Inference Doctor intentionally **does not provision GPUs**. Run your preferred benchmark on your existing GPU/self-hosted runner, save normalized results, then let Inference Doctor make the measurement-quality + regression decision.

See [GitHub Actions integration](docs/github-actions.md).

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

Use `--json` for machine-readable output. See [measurement quality](docs/measurement-quality.md) for the decision semantics and [benchmark format](docs/benchmark-format.md) for the input contract.

The [sanitized H100 reliability example](examples/inference-ci-reliability-001/README.md) preserves a real experiment where the correct result was `INCONCLUSIVE` because baseline measurements were unstable.

### Compare two normalized files

For a simple one-to-one comparison:

```bash
inference-doctor compare \
  --baseline baseline.json \
  --candidate candidate.json
```

See [upgrade guard](docs/upgrade-guard.md).

## Secondary feature: diagnose a live vLLM service

Inference Doctor also keeps its original read-only Prometheus diagnostic mode for queue/capacity pressure, decode pressure, and meaningful KV-cache pressure.

```bash
inference-doctor diagnose \
  --prometheus http://localhost:9090 \
  --ttft-slo-ms 400 \
  --tpot-slo-ms 50
```

It reads Prometheus only and does not modify vLLM, Prometheus, or traffic. High GPU/KV utilization alone is not treated as an incident; findings require corroborating evidence.

See [example reports](docs/example-reports.md) and [local integration](docs/local-integration.md).

## Try it on a real change

The project is at the stage where real-world feedback matters more than more features. If you run Inference Doctor on an actual inference change, open a **Benchmark feedback** issue and tell us what worked, what was confusing, and whether you would put the gate in CI.

Please redact prompts, credentials, private model names, customer data, and cloud access details.

## Scope

> **Unit tests protect correctness. Inference Doctor protects inference performance.**

Deliberately not included yet: hosted GPU provisioning, dashboard/SaaS, optimizer, Kubernetes operator, SGLang support, or automatic root-cause analysis.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md), and [LICENSE](LICENSE).

Historical experiments that informed the measurement-quality design live under `experiments/`.
