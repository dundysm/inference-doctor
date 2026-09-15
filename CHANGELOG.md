# Changelog

## Unreleased

### Added

- Reusable GitHub Action for repeated-run inference regression gating.
- No-GPU quickstart fixtures for trying `compare-runs` locally in minutes.
- GitHub Actions documentation and PR smoke coverage.
- Experimental `inference-doctor compare` upgrade guard for normalized benchmark result files, with configurable regression thresholds, terminal and JSON output, and CI exit codes.
- Auditable reproduction harness for the FP16 vLLM 0.19.1-to-0.24.0 regression reported in public issue #48035.
- External-server and same-Pod RunPod execution paths for the #48035 harness, including GPU UUID continuity validation.

### Changed

- Repositioned the project around trustworthy inference performance CI: `PASS`, `FAIL`, or `INCONCLUSIVE` after measurement-quality checks.
- `compare-runs` is now the primary documented workflow; Prometheus diagnosis remains supported as a secondary feature.

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
