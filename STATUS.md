# Status

October 11, 2026. Unreleased work on `main` after 0.3.0; not on PyPI yet. Install from
a clone (`python -m pip install .`). See the [README](README.md) for what the tool does.

## What works

- **The gate.** `compare-runs` validates the comparison (environment, failed requests,
  missing metrics, at least 3 runs per side), then decides from a 90% confidence
  interval for the regression: `PASS`, `FAIL` or `INCONCLUSIVE`, each with a reason.
- **Tool formats.** It reads `vllm bench serve` results (including `--append-result`),
  AIPerf and GenAI-Perf exports, GuideLLM `benchmarks.json` and SGLang JSONL directly,
  plus the normalized schema for other harnesses.
- **CI.** A composite GitHub Action with a job summary, outputs, and an exit code that
  fails the step on `FAIL`, `INCONCLUSIVE` or an error. One exit-code scheme across all
  commands; `--version`.
- **Checks.** 129 tests, ruff, and mypy run in CI on Python 3.11 and 3.14 on Linux and
  3.14 on macOS and Windows, plus the Action itself on the checked-in examples.

## Measured

- **Decision rule (simulation).** PASS, FAIL and INCONCLUSIVE rates for 3, 5 and 10 runs
  per side at 2%, 4% and 8% CV and true drops of 0 to 20%, in
  [docs/gate-calibration.md](docs/gate-calibration.md). At the threshold each wrong
  verdict occurs about 5% of the time, elsewhere at most 0.5%. The simulation assumes
  independent, normal runs; real benchmarks with drift or warm-up effects are not
  modelled.
- **One real run.** `examples/inference-ci-reliability-001`: BAAI/bge-m3 on one H100,
  six runs per side, two vLLM versions. `INCONCLUSIVE`, with an interval of -3.33% to
  +11.17% caused by one anomalous baseline block. This is a single experiment, not a
  rate.
- **Format readers.** Tested against records built from each tool's source on its main
  branch in October 2026. Not yet checked against a corpus of real files from each tool
  or across tool versions.

## Limitations

- No project other than this one runs the gate in CI yet.
- One metric per run of `compare-runs`; gating on throughput and TTFT p99 together takes
  two steps.
- `--gpu` and `--model` label both sides at once, so they cannot catch a run on the wrong
  GPU. Only GPU names recorded in the results (vLLM `--metadata`, AIPerf telemetry) are
  compared.
- `diagnose` has known bugs: its queue mean is wrong with more than one vLLM replica,
  queue pressure can be reported with no queue evidence, and it always exits 0. They are
  next in line.
- `compare` judges one run against one run and cannot measure noise; it has two known
  bugs (a crash on a zero baseline, and `PASS` on empty metrics).

## What is left

Next, in order:

1. Fix or retire `diagnose` and `compare`.
2. Publish to PyPI with trusted publishing, tag the Action as `v1`, and install the
   Action into its own environment instead of the caller's Python.
3. Community files: security policy, code of conduct, support, citation, issue forms.
4. Run the gate on the upstream vLLM regressions reproduced under `experiments/`
   (vllm-project/vllm#48035 and #52630) and share the results there.
