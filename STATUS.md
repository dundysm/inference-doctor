# Status

October 11, 2026. inference-doctor 0.4.0 (`pip install inference-doctor`; the Action as
`dundysm/inference-doctor@v1`). See the [README](README.md) for what the tool does.

## What works

- **The gate.** `compare-runs`, on one metric or several at once, validates the
  comparison (environment, failed requests, missing metrics, at least 3 runs per side),
  then decides from a 90% confidence interval for the regression: `PASS`, `FAIL` or
  `INCONCLUSIVE`, each with a reason.
- **Tool formats.** It reads `vllm bench serve` results (including `--append-result`),
  AIPerf and GenAI-Perf exports, GuideLLM `benchmarks.json` and SGLang JSONL directly,
  plus the normalized schema for other harnesses.
- **CI.** A composite GitHub Action with a job summary, outputs, and an exit code that
  fails the step on `FAIL`, `INCONCLUSIVE` or an error. One exit-code scheme across all
  commands; `--version`.
- **Checks.** 157 tests, ruff, and mypy run in CI on Python 3.11 and 3.14 on Linux and
  3.14 on macOS and Windows, plus the Action itself on the checked-in examples.

## Measured

- **Decision rule (simulation).** PASS, FAIL and INCONCLUSIVE rates for 3, 5 and 10 runs
  per side at 2%, 4% and 8% CV and true drops of 0 to 20%, in
  [docs/gate-calibration.md](docs/gate-calibration.md). At the threshold each wrong
  verdict occurs about 5% of the time, elsewhere at most 0.5%. The simulation assumes
  independent, normal runs; real benchmarks with drift or warm-up effects are not
  modelled.
- **A real upstream regression (decision rule only).** Applied to the 14 three-run
  comparisons in vLLM issue #52630, from the reporter, a maintainer and this repository:
  every upgrade on the reporter's H100 PCIe `FAIL`s (at least 18% slower at 95% confidence),
  the two same-machine controls and the torch-only A/B `PASS`, and the comparisons the
  thread disagrees about are `INCONCLUSIVE`. The validity checks were not applied, because
  the posted numbers lack failed-request counts and environment fields. See
  [docs/validation-vllm-52630.md](docs/validation-vllm-52630.md).
- **One real run.** `examples/inference-ci-reliability-001`: BAAI/bge-m3 on one H100,
  six runs per side, two vLLM versions. `INCONCLUSIVE`, with an interval of -3.33% to
  +11.17% caused by one anomalous baseline block. This is a single experiment, not a
  rate.
- **Format readers.** Tested against records built from each tool's source on its main
  branch in October 2026. Not yet checked against a corpus of real files from each tool
  or across tool versions.

## Limitations

- No project other than this one runs the gate in CI yet.
- `--gpu` and `--model` label both sides at once, so they cannot catch a run on the wrong
  GPU. Only GPU names recorded in the results (vLLM `--metadata`, AIPerf telemetry) are
  compared.
- `diagnose` is experimental. Its rules are checked against synthetic snapshots. The
  queue rule changed in 0.4.0 to require queue evidence, and the one real integration run
  of it (E003 in `docs/experiment-log.md`) was recorded under the old rule without the
  waiting-request count, so it has not been re-verified.
- `compare` is deprecated: it judges one run against one run and cannot measure noise.

## What is left

Next, in order:

1. Share the #52630 validation in that issue's thread (drafted, not yet posted).
2. Get the gate running in a project's real CI, with the validity checks on.

The #48035 harness under `experiments/` tests a claim its reporter has since retracted (a
v0.19.1 to v0.24.0 FP16 slowdown caused by configuration drift), so there is nothing to
post there.
