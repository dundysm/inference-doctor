# Security policy

## Supported versions

Fixes go into the latest release on PyPI and the Action's `v1` tag. Please check that a
problem is present there before reporting it.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/dundysm/inference-doctor/security/advisories/new),
or email dundysm@gmail.com. You will get an answer within a week.

Things that are in scope:

* Code execution, or reads or writes outside the given directories, when Inference Doctor
  parses a result file it was pointed at. It only parses JSON; it never imports, evaluates
  or runs anything from the results.
* The GitHub Action running something other than the CLI with the inputs it was given, for
  example shell injection through an input value, or installing packages outside its own
  virtual environment.
* `diagnose` sending anything other than read-only PromQL queries to the Prometheus URL it
  was given.

Benchmark results are not secrets to Inference Doctor: the JSON report and the Action's job
summary repeat the model, GPU and parameters found in them. Do not feed it results that
contain prompts or credentials you would not want in a CI log.
