# Quickstart example

This example is intentionally tiny and requires no GPU. It contains two repeated baseline measurements and two candidate measurements using the normalized benchmark schema.

Run from the repository root:

```bash
inference-doctor compare-runs \
  --baseline examples/quickstart/baseline \
  --candidate examples/quickstart/candidate \
  --metric texts_per_second
```

The baseline values are `1000` and `1010` texts/sec. The candidate values are `980` and `990` texts/sec. Both sets are stable under the default 5% CV threshold and the candidate regression is below the default 10% failure threshold, so the expected result is `PASS`.

This data is synthetic and exists only to demonstrate the CLI and GitHub Action workflow.
