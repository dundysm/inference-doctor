# Validation on vLLM issue #52630

[vLLM issue #52630](https://github.com/vllm-project/vllm/issues/52630) reports a pooling
throughput regression for `BAAI/bge-m3` after vLLM 0.19. By October 2026 its thread held
several comparisons with three runs per side, from the reporter (H100 PCIe), from a
maintainer (H100 SXM), and from this repository's own run (H100 SXM). This page applies the
`compare-runs` decision, with its defaults, to each of them.

Regenerate the table with `python tools/validate_vllm_52630.py`; `tests/test_validation_52630.py`
checks the verdicts in CI.

## Method and limits

- **Data.** Numbers posted in the issue thread, copied on 2026-10-11, and the repetition
  values of this repository's run from [benchmark-reliability-52630.md](benchmark-reliability-52630.md).
  No raw result files were available for the thread's numbers.
- **Decision.** The interval decision of `compare-runs` with its defaults: a 90% Welch
  confidence interval on the log ratio of mean texts/s, a 10% regression threshold.
- **Not re-checked.** The validity checks (failed requests, matching environments) need
  fields the posted numbers do not include. Every comparison is between runs on the same
  machine according to their authors; comparisons across machines are left out on purpose,
  because the reporter's and the maintainer's hardware differ.
- **What a run is.** Each of the reporter's values is the best throughput of a 16/32/64
  concurrency sweep; the maintainer's runs are at concurrency 32; this repository's are
  per concurrency. The interval assumes the three runs of a side are independent, which
  three runs cannot confirm.

## Result

| Source | Comparison (3 runs per side) | Mean change | Regression, 90% interval | Verdict |
| --- | --- | ---: | ---: | --- |
| Reporter, H100 PCIe | v0.19.0 -> v0.27.1 | -22.2% | +20.0% to +24.3% | FAIL |
| Reporter, H100 PCIe | v0.19.0 -> v0.28.0 | -22.6% | +20.4% to +24.7% | FAIL |
| Reporter, H100 PCIe | v0.19.0 -> nightly 6f7df92a8, attempt 1 | -27.2% | +24.5% to +29.8% | FAIL |
| Reporter, H100 PCIe | v0.19.0 -> nightly 6f7df92a8, attempt 2 | -25.3% | +23.2% to +27.4% | FAIL |
| Reporter, H100 PCIe | v0.19.0 -> nightly a9a17e70 | -35.1% | +17.8% to +48.8% | FAIL |
| Reporter, H100 PCIe | v0.27.1 -> v0.28.0 | -0.5% | -1.5% to +2.4% | PASS |
| Reporter, H100 PCIe | nightly attempt 1 -> attempt 2 (same build) | +2.6% | -6.4% to +1.1% | PASS |
| Reporter, H100 PCIe | nightly -> nightly with --renderer-num-workers 8 | -7.7% | -29.5% to +34.2% | INCONCLUSIVE |
| Maintainer, H100 SXM | torch 2.10 -> 2.11, same vLLM commit | +6.4% | -13.4% to +0.2% | PASS |
| Maintainer, H100 SXM | torch 2.10 boundary -> main 29c9af52 | -2.2% | -1.3% to +5.6% | PASS |
| Maintainer, H100 SXM | torch 2.11 boundary -> main 29c9af52 | -8.1% | +1.9% to +13.9% | INCONCLUSIVE |
| This repository, H100 SXM | v0.19.0 -> v0.21.0, concurrency 16 | -4.8% | -11.8% to +19.0% | INCONCLUSIVE |
| This repository, H100 SXM | v0.19.0 -> v0.21.0, concurrency 32 | -8.2% | -7.3% to +21.4% | INCONCLUSIVE |
| This repository, H100 SXM | v0.19.0 -> v0.21.0, concurrency 64 | -9.8% | +4.7% to +14.6% | INCONCLUSIVE |

## What it shows

- **The regression on the reporter's H100 PCIe is established with three runs.** Every
  build after v0.19.0 is at least 18% slower at 95% confidence, and the two controls on
  that machine (v0.27.1 against v0.28.0, and two attempts of the same nightly) pass. A gate
  set to 10% would have failed each of those upgrades.
- **Torch 2.11 is not the cause.** On the maintainer's same-commit A/B, torch 2.11 is at
  most 0.2% slower and probably faster. This agrees with the maintainer's own reading.
- **"Resolved on main" depends on the reference run.** Against the maintainer's torch 2.10
  run, main passes (at most 5.6% slower). Against the torch 2.11 run on the same machine,
  main is at least 1.9% and at most 13.9% slower: not enough runs to decide at 10%. This is
  the disagreement the reporter raised in the thread, and fresh v0.19.0 and main runs on one
  machine would settle it.
- **Raising `--renderer-num-workers` to 8 is neither shown to help nor to hurt.** Its first
  run (1654 texts/s) widens the interval to about ±30%. Discarding first runs, as the
  reporter suggests nightlies need, or more runs would decide it.
- **This repository's H100 SXM run could not decide anything at 10%,** the same answer it
  gave when it was run under the older rule. At concurrency 64 it does show the candidate
  slower by at least 4.7%, and by at most 14.6%.

## Evidence level

This is the decision rule applied to real, independently produced measurements, with
verdicts that agree with what the thread established by other means (the torch-only A/B,
the controls). It is not the tool running in a CI pipeline, and the validity checks were
not applied.
