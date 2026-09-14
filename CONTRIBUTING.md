# Contributing

Thanks for helping improve Inference Doctor. The project is intentionally a narrow, read-only diagnostic CLI, so changes should stay small and evidence-driven.

## Issues

Please include the vLLM version, relevant metric names from `/metrics`, the Prometheus query window, supplied SLOs, and a redacted CLI report or JSON output. Do not include API tokens, pod identifiers, public proxy URLs, or customer prompts.

## Pull requests

- Keep a change focused on one observable behavior, compatibility fix, test, or documentation improvement.
- Preserve the current diagnostic rules unless the proposal explicitly includes validation evidence and tests.
- Add or update regression tests for collector and rule behavior.
- Update documentation when metric queries, supported signals, or report semantics change.
- Do not commit virtual environments, Prometheus data, model caches, `.env` files, credentials, or cloud-provider artifacts.

Before opening a pull request, run:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Use clear commit messages and describe the validation performed in the pull request.
