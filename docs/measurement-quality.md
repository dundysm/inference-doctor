# Measurement quality and the decision rule

`compare-runs` compares repeated, already-produced benchmark results. It does not
provision GPUs, launch servers or generate traffic.

## Decision rule

First, validity. Any of these makes the result `INCONCLUSIVE` (exit 2), whatever the
numbers say:

- the GPU model, model name or benchmark parameters differ between repetitions or
  between baseline and candidate
- a repetition recorded failed requests, or its failed-request count is missing
- the selected metric is missing or non-finite in a repetition
- a side has fewer repetitions than `--min-repetitions` (default 3)
- a side's coefficient of variation is above `--stability-cv-percent`, if you set one
  (no ceiling by default)

Then, the regression. Inference Doctor computes a confidence interval for the change in
the mean, as a percentage, signed so that positive is worse (a throughput drop or a
latency rise):

```text
whole interval at or past --regression-percent  -> FAIL, exit 1
whole interval below --regression-percent       -> PASS, exit 0
interval contains the threshold                 -> INCONCLUSIVE, exit 2
invalid arguments or input, or an internal error -> no decision, exit 3
```

The defaults are a 10% regression threshold and a 90% two-sided confidence level, so
each end of the interval is a one-sided 95% bound. In practice:

- a FAIL means the regression is at least the threshold with 95% confidence;
- a PASS means the regression is below the threshold with 95% confidence;
- INCONCLUSIVE means the data cannot tell. The reason states the interval, and more
  repetitions narrow it.

### The interval

The interval is a Welch t interval on the log of the ratio of means. The standard error
of each side's log mean is estimated with the delta method, sample variance divided by
n times the squared mean, and the degrees of freedom with the Welch-Satterthwaite
formula. The two sides may differ in spread and in number of repetitions. The interval
is transformed back to a percentage change. A log scale treats "10% slower" and "10%
faster" symmetrically and keeps the bounds above -100%.

This is the approach recommended for benchmark comparisons in Georges, Buytaert and
Eeckhout, "Statistically Rigorous Java Performance Evaluation" (OOPSLA 2007) and
Kalibera and Jones, "Rigorous Benchmarking in Reasonable Time" (ISMM 2013). The
three-way outcome is the logic of an equivalence test: a verdict in either direction is
given only when the interval clears the threshold.

### Why the CV threshold is no longer the gate

Versions up to 0.3.0 required each side's CV to be at most 5% and then compared the
means against the threshold with no interval. Simulation showed two problems:

- a healthy benchmark (true CV 4%, no change, 5 repetitions) was INCONCLUSIVE about a
  third of the time;
- with 2 repetitions at 2% CV, a true 12% regression passed about 13% of the time.

The interval accounts for noise directly: noisy data widens it, which leads to
INCONCLUSIVE rather than a confident wrong answer. `--stability-cv-percent` remains as an
optional extra ceiling for teams that want one.

[Gate calibration](gate-calibration.md) gives the measured PASS, FAIL and INCONCLUSIVE
rates for 3, 5 and 10 repetitions at several noise levels and true changes. Use 5 or
more repetitions per side.

### Several metrics

Repeat `--metric` to gate on several metrics in one run. Each metric gets the full
decision above, and the combined result is `FAIL` if any metric fails, `INCONCLUSIVE` if
none fails but one is inconclusive, and `PASS` only if every metric passes.

Each additional metric is one more chance of a false alarm. With no real change, the
chance that at least one of k metrics FAILs is at most k times the per-metric rate
(about 5% each at the threshold, far less below it). Gate on the few metrics you would
act on, such as throughput and TTFT p99, rather than all of them. The `--json` report for
several metrics has the combined `result` and one full comparison per metric under
`comparisons`.

### Assumptions

The interval assumes the repetitions of each side are independent and roughly normal.
Drift during a session, warm-up effects or a bimodal benchmark break that assumption.
To stay close to it:

- discard warm-up runs;
- interleave baseline and candidate repetitions (A, B, A, B, ...) on the same machine
  instead of running all of one side first;
- read the outlier flags. Inference Doctor flags values outside 1.5 IQR or 3 MAD of the
  median but never removes them, so one bad repetition widens the interval instead of
  being hidden.

## Input and output

Each input directory holds one normalized JSON file per repetition, named
`normalized.json`, `*.normalized.json` or `repetition-*.json`, found recursively. See
[benchmark-format.md](benchmark-format.md).

```bash
inference-doctor compare-runs \
  --baseline path/to/baseline \
  --candidate path/to/candidate \
  --metric output_token_throughput \
  --regression-percent 10 \
  --confidence 0.90 \
  --min-repetitions 5
```

Supported metrics are output-token throughput, request throughput, texts per second,
TTFT and TPOT mean, median and p95, and successful and failed request counts. Add
`--json` for the machine-readable report (`schema_version: inference-ci-quality-2`).
It includes the interval as `confidence_interval` with the regression bounds, the
ratio bounds and the degrees of freedom.

A difference in vLLM version between baseline and candidate is expected and kept as
metadata.

## Example

The sanitized `examples/inference-ci-reliability-001` fixture preserves all six
baseline and six candidate values from a real H100 run, including one anomalous
baseline block. The observed throughput change is -4.19%, but the 90% interval for the
regression is -3.33% to +11.17%. It contains the 10% threshold, so the result is
`INCONCLUSIVE`: six repetitions with one bad block cannot rule out a 10% regression.

The [GitHub Action](github-actions.md) runs the same decision in CI on benchmark results
your own runner produced.
