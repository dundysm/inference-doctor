<p align="center">
  <img src="https://raw.githubusercontent.com/dundysm/inference-doctor/main/docs/assets/banner.svg" alt="inference-doctor: did this change make LLM inference slower, and can your benchmark actually tell?" width="900">
</p>

<p align="center">
  <a href="https://pypi.org/project/inference-doctor/"><img alt="PyPI" src="https://img.shields.io/pypi/v/inference-doctor?color=3776ab"></a>
  <a href="https://github.com/dundysm/inference-doctor/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/dundysm/inference-doctor/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776ab">
  <img alt="Reads vLLM, AIPerf, GuideLLM and SGLang" src="https://img.shields.io/badge/reads-vLLM%2C%20AIPerf%2C%20GuideLLM%2C%20SGLang-1f6feb">
  <a href="https://github.com/dundysm/inference-doctor/blob/main/LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-2ea043"></a>
  <a href="https://github.com/dundysm/inference-doctor/blob/main/STATUS.md"><img alt="Status: early" src="https://img.shields.io/badge/status-early-d29922"></a>
  <a href="https://github.com/dundysm/inference-doctor/blob/main/CONTRIBUTING.md"><img alt="PRs welcome" src="https://img.shields.io/badge/PRs-welcome-2ea043"></a>
</p>

A pull request can pass every test and still make inference slower. Putting a benchmark
in CI is the obvious fix, and it goes wrong in three ways:

* **false alarms**: GPU benchmarks are noisy, so a gate on two numbers fails changes that
  did nothing, and the team turns it off,
* **missed regressions**: with two or three runs, a real 12% slowdown can land inside the
  noise and pass,
* **apples to oranges**: the two sides ran on a different GPU, model or load, and the
  difference is the setup, not the change.

inference-doctor compares repeated baseline and candidate runs and returns `PASS` or
`FAIL` only when the data can support it. Otherwise it returns `INCONCLUSIVE` and says
why. It reads the files `vllm bench serve`, AIPerf, GuideLLM and SGLang already write.

<p align="center">
  <img src="https://raw.githubusercontent.com/dundysm/inference-doctor/main/docs/assets/terminal.svg" alt="Terminal output of inference-doctor compare-runs on a real H100 run: six runs per side, throughput 4.19% lower, 90% confidence interval for the regression -3.33% to +11.17%, which contains the 10% threshold, so the result is INCONCLUSIVE." width="860">
  <br>
  <sub>A real H100 run (<code>examples/inference-ci-reliability-001</code>), rendered from the CLI by
  <code>tools/render_readme_media.py</code>. Throughput is 4.19% lower, but one noisy baseline block leaves an interval
  from -3.33% to +11.17%. Six runs cannot rule out a 10% regression, so there is no merge recommendation.</sub>
</p>

## How it decides

<p align="center">
  <img src="https://raw.githubusercontent.com/dundysm/inference-doctor/main/docs/assets/decision.svg" alt="Three 90% confidence intervals for the regression against a 10% threshold: the quickstart, +1.08% to +2.92%, PASS; a real H100 run, -3.33% to +11.17%, INCONCLUSIVE; a regressed candidate, +13.13% to +14.87%, FAIL." width="900">
</p>

1. **Is the comparison valid?** Same GPU, model and benchmark parameters on every run, no
   failed requests, the metric present everywhere, and at least 3 runs per side. If not,
   the result is `INCONCLUSIVE` and the reason names the check that failed.
2. **How big is the change, and how sure are we?** A 90% confidence interval for the
   regression (positive is worse), from a Welch t interval on the log ratio of means.
3. **Verdict.** `FAIL` when the whole interval is at or past the threshold (default 10%),
   `PASS` when the whole interval is below it, `INCONCLUSIVE` when it contains the
   threshold. More runs narrow the interval.

Outliers are flagged but never removed: a bad run widens the interval instead of being
hidden. Details and references: [measurement quality](docs/measurement-quality.md).

## Quick start

Python 3.11 or newer. No GPU: the example results are checked in, so the clone is only for
them.

```bash
pip install inference-doctor
git clone https://github.com/dundysm/inference-doctor.git && cd inference-doctor

inference-doctor compare-runs \
  --baseline examples/quickstart/baseline \
  --candidate examples/quickstart/candidate \
  --metric texts_per_second
```

`PASS`: the candidate is 2% slower, and the interval, +1.08% to +2.92%, is well below 10%.
Point `--candidate` at `examples/quickstart/regressed-candidate` to see a `FAIL`.

## Use your benchmark's own output

No converter needed. Put each side's runs in a directory and point `compare-runs` at it:

```bash
inference-doctor compare-runs \
  --baseline results/main \
  --candidate results/my-branch \
  --metric output_token_throughput --metric ttft_p99_ms
```

Repeat `--metric` to gate on several at once: the result is `FAIL` if any metric fails,
`INCONCLUSIVE` if none fails but one cannot be decided, and `PASS` only when all pass.

| Tool | What it reads |
| --- | --- |
| `vllm bench serve --save-result` | each result file, or each line with `--append-result` |
| NVIDIA AIPerf / GenAI-Perf | each `profile_export_aiperf.json` |
| GuideLLM | `benchmarks.json`, one load point of the sweep (`--guidellm-benchmark`) |
| SGLang `bench_serving` | each line of its `.jsonl` output |
| anything else | the small [normalized format](docs/benchmark-format.md) |

Metrics: output, total and request throughput, and TTFT, TPOT, ITL and end-to-end latency
at mean, median, p95 and p99. Most tools do not record the GPU; pass `--gpu` to label it,
or record it in the results (for vLLM, `--metadata gpu=...`). The
[input formats](docs/input-formats.md) page lists every field read from each tool.

[`examples/vllm-bench-serve`](examples/vllm-bench-serve/README.md) is a small example in
vLLM's format where throughput passes and TTFT fails, so gating on both catches what a
throughput-only gate would let through.

## In GitHub Actions

Run your benchmark on your own GPU runner, then gate on the results:

```yaml
- name: Gate inference performance
  uses: dundysm/inference-doctor@v1
  with:
    baseline: results/main
    candidate: results/my-branch
    metric: output_token_throughput, ttft_p99_ms
    min-repetitions: "5"
```

The Action writes a job summary with the verdict, the interval and the reason, and exposes
them as outputs. `FAIL`, `INCONCLUSIVE` and errors fail the step, so noisy evidence cannot
approve a change. inference-doctor does not provision GPUs or run the benchmark. See
[GitHub Actions](docs/github-actions.md).

## What it costs (experimental)

`inference-doctor cost` turns the same numbers into dollars: $/1M tokens, an estimate of
money spent on idle capacity, and ranked fixes. Try it with no GPU:

```
pip install inference-doctor
inference-doctor cost --demo
```

```
  hardware          8 x H100-SXM @ $3.99/GPU-h (https://www.runpod.io/pricing)
  throughput        5,200 prompt + 1,300 output tok/s   [measured]
  cost / 1M tokens  $1.364 all, $6.821 output
  occupancy         22%  (batch slots / KV cache, max of the two)
  est. $ wasted/mo  $16,894  [heuristic: idle capacity below 80% occupancy]
Ranked fixes (estimated monthly savings):
  1. Use a hosted API: gpt-oss-120B (Together)  ~$19,202/mo  [measured traffic x list price]
  2. Right-size: 8 -> 4 GPUs                    ~$11,651/mo  [heuristic]
  ...
```

Point it at two Prometheus scrapes of vLLM or SGLang (`--metrics a.prom --metrics b.prom`),
a live `--url`, or a benchmark directory (`--runs`), with `--gpu`/`--gpus` or the `--price`
you actually pay. Throughput and cache gauges are measured; waste and savings are
heuristics, labelled as such. `cost-compare` gates $/1M output tokens with the same
Welch interval as `compare-runs`, and the Action runs it when you set `cost-gpu` or
`cost-price`. Prices are list prices copied with sources and a date; see [cost](docs/cost.md).

## How much to trust it

Simulated with known truth, at default settings, from
[gate calibration](docs/gate-calibration.md) (4,000 trials per row):

| Runs per side, noise (CV) | No change | True 5% drop | True 15% drop | True 20% drop |
| --- | --- | --- | --- | --- |
| 3 runs, 4% | 78% PASS, 0% FAIL | 34% PASS, 0.1% FAIL | 0% PASS, 37% FAIL | 0% PASS, 85% FAIL |
| 5 runs, 4% | 98% PASS, 0% FAIL | 60% PASS, 0% FAIL | 0% PASS, 65% FAIL | 0% PASS, 99% FAIL |
| 10 runs, 4% | 100% PASS, 0% FAIL | 90% PASS, 0% FAIL | 0% PASS, 92% FAIL | 0% PASS, 100% FAIL |

The rest of each row is `INCONCLUSIVE`. Right at the 10% threshold, each wrong verdict
happens about 5% of the time; further from it, at most 0.5%. Noise costs decisions, not
correctness. Use 5 or more runs per side. These rates assume independent, roughly normal
runs; [interleave baseline and candidate runs](docs/measurement-quality.md#assumptions)
to stay close to that.

## Commands

| Command | What it does |
| --- | --- |
| `compare-runs` | The gate: repeated baseline and candidate runs, one or more metrics, `PASS`, `FAIL` or `INCONCLUSIVE`. |
| `compare` | Deprecated. One normalized file against another, every metric. A single run per side cannot measure noise; use `compare-runs`. See [upgrade guard](docs/upgrade-guard.md). |
| `cost` | Experimental. $/1M tokens, estimated idle-capacity $ and ranked fixes from Prometheus scrapes or benchmark results. `--demo` runs on bundled data. See [cost](docs/cost.md). |
| `cost-compare` | Experimental. The gate on $/1M output tokens at a fixed $/GPU-hour: `PASS`, `FAIL` or `INCONCLUSIVE`. |
| `diagnose` | Experimental. Reads a live vLLM server's Prometheus metrics and reports queue, prefill, decode and KV-cache pressure. Read-only. Exits 1 on a critical finding (`--fail-on` changes that). See [example reports](docs/example-reports.md). |

`compare-runs` options: `--metric` (repeatable), `--regression-percent` (10), `--confidence` (0.90),
`--min-repetitions` (3), `--stability-cv-percent` (an optional CV ceiling, off),
`--format` (auto), `--gpu`, `--model`, `--guidellm-benchmark` and `--json`.

**Exit codes**, the same for every command: `0` PASS, `1` FAIL (for `diagnose`, a
finding at or above `--fail-on`), `2` INCONCLUSIVE (or UNSTABLE for `compare`), `3` error
with no decision (bad arguments or input, an unreachable Prometheus, or an internal
error). `inference-doctor --version` prints the
version.

## What has been measured

* **The decision rule**: the calibration table above, regenerated by
  `tools/gate_simulation.py` and checked by `tests/test_gate.py` on every CI run.
* **A real upstream regression**: applied to every three-run comparison in
  [vLLM #52630](https://github.com/vllm-project/vllm/issues/52630), the gate `FAIL`s each
  upgrade on the reporter's H100 PCIe (at least 18% slower at 95% confidence), `PASS`es
  the same-build controls and the maintainers' torch-only A/B, and returns `INCONCLUSIVE`
  where the thread itself disagrees. See [validation on #52630](docs/validation-vllm-52630.md).
* **One real GPU run of our own**: `examples/inference-ci-reliability-001`, six runs per
  side of BAAI/bge-m3 on one H100 across two vLLM versions. The honest answer was
  `INCONCLUSIVE`, and it still is.
* **The format readers**: tested against records built from each tool's source code. They
  have not yet been run on a large set of real files from each tool, so please
  [report](https://github.com/dundysm/inference-doctor/issues/new/choose) any file they
  misread.
* **Not yet**: a gate running in another project's CI. See [STATUS.md](STATUS.md).

## Help test it

The most useful thing right now is a real change: run `compare-runs` on results from your
own benchmark and open a **Benchmark feedback** issue saying what it decided, whether that
was right, and what was confusing. Redact prompts, credentials, private model names and
customer data.

Want to contribute code or docs? Start with an issue labelled
[good first issue](https://github.com/dundysm/inference-doctor/labels/good%20first%20issue) or [help wanted](https://github.com/dundysm/inference-doctor/labels/help%20wanted); the ones labelled
[needs GPU](https://github.com/dundysm/inference-doctor/labels/needs%20GPU) are for anyone with access to an NVIDIA GPU.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), and [SUPPORT.md](SUPPORT.md) for where to ask
what. Security reports go through [SECURITY.md](SECURITY.md). The changelog is
[CHANGELOG.md](CHANGELOG.md), and [CITATION.cff](CITATION.cff) says how to cite the project.
The experiments that informed the design live under `experiments/`.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check . && ruff format --check . && mypy
```

## License

Apache-2.0. See [LICENSE](LICENSE).
