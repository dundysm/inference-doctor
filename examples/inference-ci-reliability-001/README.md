# Sanitized reliability example

This fixture preserves the six measured `texts_per_second` values from the
completed `inference-ci-reliability-001` run. The anomalous baseline value
`2649.7066666666665` is retained in `baseline/repetition-003.json`; no value
was removed or rewritten.

The result is **INCONCLUSIVE**. Baseline mean is approximately 2251.22
texts/s with 8.93% CV. Candidate mean is approximately 2156.78 texts/s with
3.86% CV. The observed delta is approximately -4.19%.

The experiment was preregistered with the rule used up to 0.3.0, a 5% CV
ceiling, and the baseline CV exceeded it (`summary.json` records that
decision). The current rule reaches the same verdict for a more direct
reason: the 90% confidence interval for the regression, -3.33% to +11.17%,
contains the 10% threshold, so six repetitions with one bad block cannot rule
out a 10% regression.

Run the local MVP comparison with:

```bash
inference-doctor compare-runs \
  --baseline examples/inference-ci-reliability-001/baseline \
  --candidate examples/inference-ci-reliability-001/candidate \
  --metric texts_per_second
```

The source experiment used one H100, concurrency 32, BAAI/bge-m3, six blocks
per version, and the frozen immutable workload. This sanitized fixture keeps
only the normalized measurement values and public experiment metadata.
