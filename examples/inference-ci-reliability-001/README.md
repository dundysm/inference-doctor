# Sanitized reliability example

This fixture preserves the six measured `texts_per_second` values from the
completed `inference-ci-reliability-001` run. The anomalous baseline value
`2649.7066666666665` is retained in `baseline/repetition-003.json`; no value
was removed or rewritten.

The official result is **INCONCLUSIVE**. Baseline mean is approximately
2251.22 texts/s with 8.93% CV. Candidate mean is approximately 2156.78
texts/s with 3.86% CV. The observed delta is approximately -4.19%, but the
baseline CV exceeded the preregistered 5% stability threshold, so no
regression decision was made.

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
