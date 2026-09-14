# vLLM #52630 Benchmark Reliability Study

## Scope and official result

This is an offline measurement-reliability analysis of the completed #52630
raw benchmark artifacts. It does not change the preregistered experiment
interpretation or any product behavior.

The official experiment result remains:

- INVALID
- A/A UNSTABLE
- B/B UNSTABLE
- A/B directionally PASS
- historical regression NOT reproduced

Source data was limited to the six saved raw JSON files under the completed
experiment export:

the sanitized `vllm-52630-results-v3` experiment export

The run used three repetitions for each version at concurrency 16, 32, and
64. Values below are `texts_per_second` from the raw files. Standard deviation
is sample standard deviation. Pairwise percentages use:

`100 * (second repetition - first repetition) / first repetition`

No observation was deleted or excluded.

## Analysis

### Repetition-level throughput

| Version | Concurrency | Repetition values, texts/s | Mean | Median | SD | CV | Min - max |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 16 | 3250.59, 2790.57, 2942.19 | 2994.45 | 2942.19 | 234.42 | 7.83% | 2790.57 - 3250.59 |
| baseline | 32 | 3435.63, 3330.51, 3369.53 | 3378.56 | 3369.53 | 53.14 | 1.57% | 3330.51 - 3435.63 |
| baseline | 64 | 3378.04, 3490.60, 3411.32 | 3426.65 | 3411.32 | 57.82 | 1.69% | 3378.04 - 3490.60 |
| candidate | 16 | 3147.51, 2569.22, 2831.85 | 2849.53 | 2831.85 | 289.55 | 10.16% | 2569.22 - 3147.51 |
| candidate | 32 | 3440.74, 2924.94, 2942.48 | 3102.72 | 2942.48 | 292.87 | 9.44% | 2924.94 - 3440.74 |
| candidate | 64 | 3202.74, 3092.01, 2981.92 | 3092.22 | 3092.01 | 110.41 | 3.57% | 2981.92 - 3202.74 |

### Pairwise repetition differences

| Version | Concurrency | Rep 1 -> 2 | Rep 1 -> 3 | Rep 2 -> 3 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 16 | -14.15% | -9.49% | +5.43% |
| baseline | 32 | -3.06% | -1.92% | +1.17% |
| baseline | 64 | +3.33% | +0.99% | -2.27% |
| candidate | 16 | -18.37% | -10.03% | +10.22% |
| candidate | 32 | -14.99% | -14.48% | +0.60% |
| candidate | 64 | -3.46% | -6.89% | -3.56% |

The pairwise table explains the control result. With the existing absolute
5% stability tolerance, baseline fails at concurrency 16, while candidate
fails at 16 and 32 and also has a 6.89% 1-to-3 difference at 64.

### Baseline versus candidate

The mean and median comparisons are deliberately separated. The median-based
comparison is **POST-HOC EXPLORATORY only** and does not replace the official
mean-based preregistered comparison.

| Concurrency | Mean baseline | Mean candidate | Mean delta | Median baseline | Median candidate | Median delta, exploratory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2994.45 | 2849.53 | -4.84% | 2942.19 | 2831.85 | -3.75% |
| 32 | 3378.56 | 3102.72 | -8.16% | 3369.53 | 2942.48 | -12.67% |
| 64 | 3426.65 | 3092.22 | -9.76% | 3411.32 | 3092.01 | -9.36% |

The mean comparison remains below the frozen 10% failure threshold at all
three points. The median comparison crosses 10% at concurrency 32, but that is
post-hoc exploratory evidence and cannot repair the unstable controls.

## Noise analysis

### Within-version noise

| Concurrency | Baseline CV | Candidate CV | Baseline range / mean | Candidate range / mean |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 7.83% | 10.16% | 15.36% | 20.29% |
| 32 | 1.57% | 9.44% | 3.11% | 16.62% |
| 64 | 1.69% | 3.57% | 3.28% | 7.14% |

Baseline is comparatively repeatable at 32 and 64. Candidate noise is much
higher at 16 and 32. The observed mean A/B effects of -4.84%, -8.16%, and
-9.76% are therefore small relative to candidate repetition noise at 16 and
32, and close to the candidate 64 range at 64.

### Apparent outliers

The transparent checks were a 3-MAD rule and a 1.5-IQR rule applied separately
to each three-value group. These are descriptive flags only. No value was
removed.

| Version | Concurrency | MAD | IQR | Apparent flag |
| --- | ---: | ---: | ---: | --- |
| baseline | 16 | 151.61 | 230.01 | none |
| baseline | 32 | 39.02 | 52.56 | none |
| baseline | 64 | 33.28 | 56.28 | none |
| candidate | 16 | 262.63 | 289.15 | none |
| candidate | 32 | 17.54 | 257.90 | repetition 1 by 3-MAD only |
| candidate | 64 | 110.09 | 110.41 | none |

Candidate repetition 1 at concurrency 32, 3440.74 texts/s, is flagged by the
3-MAD rule because repetitions 2 and 3 cluster near 2930 texts/s. It is not an
IQR outlier. With only three observations, this is not enough evidence to call
it an actual outlier, and it remains in every calculation.

### A/A, B/B, and A/B comparison

- A/A is unstable at concurrency 16 and stable at 32 and 64 under the current
  5% pairwise rule.
- B/B is unstable at 16 and 32 and unstable at 64 because its 1-to-3 pair is
  -6.89%.
- A/B is directionally lower for the candidate at every concurrency, but the
  mean effect is below 10% everywhere.
- Because both same-version controls are unstable overall, the A/B result is
  not a valid historical-regression claim.

The sample size is three repetitions per cell. It supports describing the
observed behavior, but not a strong statistical conclusion about the true
version effect or long-run variance.

## Decision instability simulation

This is an empirical resampling exercise, not a confidence interval with a
valid independent-sample guarantee. For each cell, the three observed
repetitions were resampled with replacement. For `n=1`, `n=2`, and `n=3`, all
possible ordered resamples were enumerated. Same-version false-fail risk is the
fraction of two independent resampled summaries whose absolute percentage
difference exceeds the threshold. Cross-version miss risk is the fraction of
independent baseline/candidate resampled summaries whose candidate degradation
does not cross the one-sided threshold.

Cells are written as `n=1 / n=2 / n=3` percentages.

### Estimated same-version false-fail risk

| Version | Concurrency | 5% threshold | 10% threshold | 15% threshold |
| --- | ---: | ---: | ---: | ---: |
| baseline | 16 | 66.7 / 49.4 / 38.3 | 33.3 / 11.1 / 4.9 | 11.1 / 1.2 / 0.1 |
| baseline | 32 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| baseline | 64 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| candidate | 16 | 66.7 / 59.3 / 46.1 | 55.6 / 28.4 / 15.9 | 22.2 / 7.4 / 2.3 |
| candidate | 32 | 44.4 / 59.3 / 63.9 | 44.4 / 9.9 / 16.6 | 22.2 / 4.9 / 1.1 |
| candidate | 64 | 22.2 / 12.3 / 1.9 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

The apparent non-monotonic values are a consequence of resampling only three
observations, not evidence that more runs inherently increase failure risk.
They show that the candidate control decision is highly sensitive to which
observations are sampled, especially at 16 and 32.

### Estimated cross-version miss risk

| Concurrency | Observed mean delta | Threshold | Miss risk n=1 / n=2 / n=3 |
| ---: | ---: | ---: | ---: |
| 16 | -4.84% | 5% | 55.6 / 48.1 / 52.5% |
| 16 | -4.84% | 10% | 66.7 / 74.1 / 83.1% |
| 16 | -4.84% | 15% | 88.9 / 93.8 / 96.2% |
| 32 | -8.16% | 5% | 33.3 / 25.9 / 25.9% |
| 32 | -8.16% | 10% | 33.3 / 55.6 / 70.4% |
| 32 | -8.16% | 15% | 100.0 / 100.0 / 100.0% |
| 64 | -9.76% | 5% | 0.0 / 0.0 / 0.0% |
| 64 | -9.76% | 10% | 55.6 / 55.6 / 55.8% |
| 64 | -9.76% | 15% | 100.0 / 100.0 / 100.0% |

At concurrency 32, the observed mean effect does not reach the 10% threshold,
so a high 10% miss risk is expected rather than surprising. At concurrency
64, every one-run resample crossed the 5% threshold, but only about 44% of the
three-run resamples crossed 10%. The empirical cross-version decision is
therefore threshold-sensitive and repetition-sensitive.

The resampled cross-version delta ranges were:

- concurrency 16: -20.96% to +12.79%
- concurrency 32: -14.86% to +3.31%
- concurrency 64: -14.57% to -5.19%

Those ranges show why a single favorable or unfavorable repetition can change
the decision. They should not be read as a population confidence interval.

## Convergence

The following are cumulative means in recorded repetition order. They are
descriptive convergence traces, not independent estimates.

| Version | Concurrency | n=1 mean | n=2 mean | n=3 mean |
| --- | ---: | ---: | ---: | ---: |
| baseline | 16 | 3250.59 | 3020.58 | 2994.45 |
| baseline | 32 | 3435.63 | 3383.07 | 3378.56 |
| baseline | 64 | 3378.04 | 3434.32 | 3426.65 |
| candidate | 16 | 3147.51 | 2858.37 | 2849.53 |
| candidate | 32 | 3440.74 | 3182.84 | 3102.72 |
| candidate | 64 | 3202.74 | 3147.38 | 3092.22 |

Baseline appears close to convergence by three repetitions at 32 and 64. The
candidate does not appear converged at 32: adding the third repetition moves
the cumulative mean another 2.5% downward from the two-run mean. Candidate 64
also continues downward, while candidate 16 is dominated by the large first to
second change. Three repetitions are not enough to establish stable estimates
at the noisier candidate points.

## Next experiment design

The next experiment should be preregistered to answer:

> How much measurement duration and how many repetitions are required to
> reliably detect a 5%, 10%, and 15% inference throughput regression while
> keeping false failures low?

### Proposed fixed design

1. Use one physical H100 with the UUID recorded before and after every version
   block. Abort validity on UUID, driver, model, or serving-argument changes.
2. Keep the current model, immutable dataset commit, first-4096 selection,
   32-document request shape, server arguments, and concurrency points 16, 32,
   and 64.
3. Use a five-minute fixed warmup after every server start and version swap.
   Exclude warmup requests from measurements. Keep the request order and
   payloads fixed.
4. Run matched ABBA cycles at the primary concurrency 32: baseline, candidate,
   candidate, baseline. Restart and warm each server between blocks. This
   reduces time-order bias while preserving isolated version environments.
5. Collect 12 measured repetitions per version at concurrency 32, with a
   five-minute measurement window per repetition. Collect six repetitions per
   version at 16 and 64 with three-minute windows for supporting evidence.
6. Treat the first six repetitions per version as the minimum pilot and the
   full twelve at concurrency 32 as the preregistered primary result. Do not
   stop early because a candidate result is favorable or unfavorable.
7. Before looking at the cross-version result, evaluate repeated A/A and B/B
   controls. Require complete data, matched hardware, no request failures, and
   all preregistered control comparisons within 5%, or label the result
   INCONCLUSIVE / UNSTABLE.
8. For a regression claim, require stable controls and a one-sided uncertainty
   bound for the candidate delta to cross the relevant threshold. Report the
   raw mean, median, CV, pairwise values, and all repetitions regardless of the
   decision.

### Why this scale

The current candidate CV is 9.44% at the primary concurrency. Three short
repetitions cannot distinguish an 8% observed effect from ordinary run noise.
Longer fixed-duration windows should reduce request-count and startup effects;
12 interleaved primary repetitions provide enough observations to estimate
control variance instead of treating three values as stable by default.

### Estimated runtime and cost

For the proposed design, the measured windows total approximately:

- primary concurrency 32: 12 repetitions x 2 versions x 5 minutes = 120 minutes
- supporting 16 and 64 points: 6 repetitions x 2 versions x 2 points x 3 minutes = 72 minutes
- warmups, model loads, server restarts, metadata, and export: approximately
  60 to 90 minutes

Estimated total: approximately 4.2 to 4.7 H100 hours. At the observed
`$3.49/hour` Secure H100 rate, that is approximately `$14.66` to `$16.40`.
This is an estimate only. No GPU or RunPod resource was provisioned for this
study.

## Product implications

These are recommendations for future Inference Performance CI semantics only.
They are not implemented here.

- **PASS:** complete data, matched hardware and configuration, stable same-version
  controls, and no preregistered candidate regression threshold crossed.
- **FAIL:** complete and stable controls, with the candidate degradation crossing
  the preregistered threshold under the predeclared uncertainty rule.
- **INCONCLUSIVE / UNSTABLE:** any failed control, GPU/configuration mismatch,
  missing repetitions, material request failures, or uncertainty wide enough
  that the threshold decision changes under plausible resampling.
- Fixed percentage thresholds alone are not sufficient. A threshold must be
  combined with control stability, enough repeated evidence, and a declared
  uncertainty rule.
- A merge-blocking minimum should require at least six repetitions per version
  for a pilot, with twelve primary repetitions when observed control CV is near
  the 10% seen here. It should also require stable A/A and B/B controls on the
  same physical GPU and preserve all raw repetitions for audit.

## Findings

1. The completed run is still officially INVALID: A/A and B/B are UNSTABLE,
   even though the directional A/B comparison is PASS.
2. The candidate degradation looks directionally real: candidate mean
   throughput is lower at all three concurrency points, with the largest mean
   effect at 64 (-9.76%).
3. The size and reliability of the effect remain statistically unresolved.
   Candidate noise is high at 16 and 32, the primary 32 point has a 9.44% CV,
   and the resampling decision changes materially with repetition count and
   threshold.
4. The proposed next experiment is a same-GPU, ABBA-interleaved, longer-window
   design with 12 primary repetitions per version and six supporting repetitions
   at the other concurrency points.
5. Estimated runtime is 4.2 to 4.7 H100 hours, or approximately `$14.66` to
   `$16.40` at `$3.49/hour`. No cloud resource was provisioned.
6. Exact file written: `docs/benchmark-reliability-52630.md`
