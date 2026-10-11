# Contributing to Inference Doctor

Thanks for looking. Inference Doctor decides whether a change made LLM inference slower,
and refuses to decide when the benchmark cannot tell. It gets better mostly from contact
with real benchmark results it has not seen. You do not need to write code to help.

## Ways to help

* **Run it on a real change.** Point `inference-doctor compare-runs` at your own baseline and
  candidate results and open a
  [benchmark feedback](https://github.com/dundysm/inference-doctor/issues/new?template=benchmark-feedback.yml)
  issue: what it decided, whether that was right, what was confusing. That is the most
  useful report there is.
* **Report a misread file.** The format readers were built from each tool's source, not from
  a large set of real files. If a vLLM, AIPerf, GuideLLM or SGLang result is read wrongly, a
  [bug report](https://github.com/dundysm/inference-doctor/issues/new?template=bug_report.yml)
  with one redacted file and the tool's version fixes it for everyone.
* **Ask for a format.** If your benchmark tool's output is not read yet, open a
  [format request](https://github.com/dundysm/inference-doctor/issues/new?template=format_request.yml)
  with an example file.
* **Questions and ideas** go in [Discussions](https://github.com/dundysm/inference-doctor/discussions).

Never include API tokens, credentials, customer prompts, private model names or cloud
access details in an issue.

## Development setup

```bash
git clone https://github.com/dundysm/inference-doctor && cd inference-doctor
python -m venv .venv && source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q                                  # ~5 s, no GPU
ruff check . && ruff format --check . && mypy
```

Python 3.11 or newer. CI runs the same checks, with tests on Python 3.11 and 3.14 on Linux
and 3.14 on macOS and Windows, and runs the GitHub Action on the checked-in examples.

## Where things are

| Path | What it holds |
|---|---|
| `src/inference_doctor/quality.py` | `compare-runs`: validity checks, the interval decision, several metrics |
| `src/inference_doctor/stats.py` | the Student t distribution and the Welch interval on the log ratio of means |
| `src/inference_doctor/adapters.py` | reading vLLM, AIPerf, GuideLLM, SGLang and normalized results |
| `src/inference_doctor/benchmark_results.py` | the normalized schema and the comparability check |
| `src/inference_doctor/cli.py` | every command and the exit codes |
| `src/inference_doctor/diagnostics.py`, `collectors/` | `diagnose` (experimental): Prometheus queries and rules |
| `src/inference_doctor/comparison.py` | `compare` (deprecated) |
| `action.yml` | the composite GitHub Action |
| `tools/gate_simulation.py` | the simulation behind `docs/gate-calibration.md` |
| `tools/render_readme_media.py` | renders `docs/assets/terminal.svg` from the CLI |
| `examples/` | checked-in results used by the README, the docs, the tests and CI |
| `experiments/` | the GPU experiments that informed the design |

## Adding a format

A reader turns one tool's result record into a `BenchmarkResult`. `_from_vllm` in
`adapters.py` is the shortest template, `_from_aiperf` the most complete.

1. Find the code in the tool that writes the file and use its field names exactly; link it
   in a comment and in `docs/input-formats.md`.
2. A value the file does not hold stays `None` (or `unspecified` for GPU and model), never a
   guess. Convert units explicitly.
3. Record only workload settings in `benchmark_parameters`: they must be equal on both sides
   for a valid comparison, so leave out labels, timestamps and output paths.
4. Add detection to `detect_format` and a record builder plus tests in
   `tests/test_adapters.py` that check every metric, the failed-request count and the
   parameters.

## Pull requests

* Keep a pull request to one change, and say in the description what it changes and how
  you checked it.
* Add or update tests for behavior changes. `pytest`, `ruff` and `mypy` must pass.
* Keep the `PASS`, `FAIL` and `INCONCLUSIVE` semantics unless the proposal includes evidence.
  If the decision rule changes, regenerate `docs/gate-calibration.md` with
  `python tools/gate_simulation.py > docs/gate-calibration.md`.
* Never remove outliers from the official calculation.
* Add a line under `## Unreleased` in `CHANGELOG.md` for anything a user would notice.
* By submitting a pull request you agree that your contribution is licensed under the
  project's [Apache-2.0 license](LICENSE).

## Writing style

Reports, docs and messages say what was measured and how, plainly. A claim about accuracy
names what backs it: the simulation in `docs/gate-calibration.md`, a checked-in example,
or a real run. A synthetic example is never presented as a measurement.

## Releasing

1. Move the `## Unreleased` entries in `CHANGELOG.md` under `## X.Y.Z - date`, and set the
   version in `src/inference_doctor/__init__.py` and `CITATION.cff` (with
   `date-released`). Commit to `main`.
2. Once CI is green, tag and push: `git tag -a vX.Y.Z -m "inference-doctor X.Y.Z" && git
   push origin vX.Y.Z`. The release workflow checks the tag, builds, and publishes to PyPI
   through trusted publishing.
3. Move the Action's major tag when its inputs and outputs are compatible:
   `git tag -fa v1 -m "Action v1" vX.Y.Z && git push -f origin v1`.
4. Create the GitHub release from the changelog section.

## Conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
