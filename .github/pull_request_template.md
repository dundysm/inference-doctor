## What this changes

<!-- One change per pull request. Link the issue it closes, if any: "Closes #12". -->

## How it was checked

<!-- Tests added or updated, commands run, outputs compared. -->

- [ ] `python -m pytest -q` passes
- [ ] `ruff check .`, `ruff format --check .` and `mypy` pass
- [ ] `CHANGELOG.md` has a line under `## Unreleased` (if a user would notice the change)
- [ ] If the decision rule changed: `docs/gate-calibration.md` is regenerated with `python tools/gate_simulation.py`
- [ ] No private benchmark results, prompts, credentials or model files are committed
