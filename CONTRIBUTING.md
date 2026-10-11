# Contributing

Thanks for helping improve Inference Doctor.

The primary product goal is narrow: **trustworthy performance regression testing for LLM inference**. Changes should make it easier to compare baseline vs candidate performance without giving false confidence when benchmark data is noisy.

The original read-only Prometheus diagnostic mode remains supported as a secondary feature.

## Issues

For inference-CI issues, include:

- selected metric
- baseline and candidate repetition counts
- relevant normalized benchmark metadata
- configured stability/regression thresholds
- redacted terminal or JSON output

For Prometheus diagnostic issues, include the vLLM version, relevant metric names, query window, supplied SLOs, and a redacted report.

Never include API tokens, credentials, customer prompts, private model data, or cloud access details.

## Pull requests

- Keep each change focused on one observable behavior or usability improvement.
- Preserve `PASS`, `FAIL`, and `INCONCLUSIVE` semantics unless the proposal includes evidence and tests.
- Do not silently remove statistical outliers from official decisions.
- Add or update regression tests for behavior changes.
- Update docs when schemas, thresholds, supported metrics, or exit codes change.
- Keep GPU/cloud provisioning outside the core comparison engine.
- Do not commit virtual environments, model caches, credentials, `.env` files, or cloud-provider artifacts.

Before opening a pull request:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check . && ruff format --check . && mypy
```

CI runs the same checks, with tests on Python 3.11 and 3.14 on Linux and on 3.14 on macOS and Windows.

## Releasing

1. Move the `## Unreleased` entries in `CHANGELOG.md` under `## X.Y.Z - date` and set
   `__version__` in `src/inference_doctor/__init__.py` to `X.Y.Z`. Commit to `main`.
2. Once CI is green, tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`. The
   release workflow checks the tag, builds, and publishes to PyPI through trusted
   publishing.
3. Move the Action's major tag to the release when its inputs and outputs are compatible:
   `git tag -f v1 vX.Y.Z && git push -f origin v1`.

Use clear commit messages and describe the validation performed in the pull request.
