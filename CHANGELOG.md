# Changelog

## Unreleased

## 0.3.0 - 2026-09-14

### Added

- Reusable GitHub Action for repeated-run inference regression gating.
- No-GPU quickstart fixtures for trying `compare-runs` locally in minutes.
- GitHub Actions documentation and PR smoke coverage.
- Measurement-quality gate with first-class `PASS`, `FAIL`, and `INCONCLUSIVE` outcomes.
- Repeated-run statistics including mean, median, sample standard deviation, CV, min/max, and descriptive outlier flags.
- Optional environment telemetry helpers for benchmark investigations.
- Experimental `inference-doctor compare` upgrade guard for normalized benchmark result files, with configurable regression thresholds, terminal and JSON output, and CI exit codes.
- Auditable historical regression and benchmark-reliability experiments that informed the measurement-quality design.

### Changed

- Repositioned the project around trustworthy inference performance CI: measurement quality is checked before a merge decision is made.
- `compare-runs` is now the primary documented workflow; Prometheus diagnosis remains supported as a secondary feature.
- Apparent outliers remain included in official decisions rather than being silently removed.

## 0.2.0 - 2026-09-13

### Added

- Queue mean, prefill p95, and prompt-token p95 collection and reporting.
- Regression coverage for non-finite Prometheus scalar values.
- Public integration reproduction notes and sanitized validation evidence.

### Changed

- Non-finite Prometheus scalar values now render as unavailable.
- Queue p95 remains visible but no longer raises queue-pressure confidence.

## 0.1.0

- Initial vertical slice with Prometheus collection, terminal and JSON reports, and queue, decode, and KV-cache diagnostic rules.
