# Quickstart example

Synthetic results that need no GPU. Each directory holds five repetitions in the
[normalized benchmark format](../../docs/benchmark-format.md).

| Directory | texts/sec per repetition | Mean |
| --- | --- | ---: |
| `baseline` | 1000, 1010, 995, 1005, 990 | 1000 |
| `candidate` | 980, 990, 975, 985, 970 | 980 |
| `regressed-candidate` | 860, 870, 855, 865, 850 | 860 |

Run from the repository root:

```bash
inference-doctor compare-runs   --baseline examples/quickstart/baseline   --candidate examples/quickstart/candidate   --metric texts_per_second
```

The candidate is 2% slower. The 90% confidence interval for the regression, about +1.1% to
+2.9%, is entirely below the default 10% threshold, so the result is `PASS` (exit 0).

Swap in the regressed candidate:

```bash
inference-doctor compare-runs   --baseline examples/quickstart/baseline   --candidate examples/quickstart/regressed-candidate   --metric texts_per_second
```

It is 14% slower and the whole interval, about +13% to +15%, is past the threshold, so the
result is `FAIL` (exit 1).

This data exists only to demonstrate the CLI and the GitHub Action.
