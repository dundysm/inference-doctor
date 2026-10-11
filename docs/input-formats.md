# Input formats

`compare-runs` reads the files your benchmark tool already writes. Put the baseline runs
in one directory and the candidate runs in another. Each file, or each line of a JSON
Lines file, counts as one repetition. Directories are searched recursively.

| Format | Tool and file | One repetition is |
| --- | --- | --- |
| `vllm` | `vllm bench serve --save-result` (any file name) | a file, or a line with `--append-result` |
| `aiperf` | NVIDIA AIPerf `profile_export_aiperf.json`, or GenAI-Perf `profile_export_genai_perf.json` | a file |
| `guidellm` | GuideLLM `benchmarks.json` | one benchmark of the sweep in a file (`--guidellm-benchmark`) |
| `sglang` | SGLang `bench_serving` output (`.jsonl`) | a line |
| `normalized` | Inference Doctor's own schema, see [benchmark-format.md](benchmark-format.md) | a file |

`--format auto`, the default, detects the format of each file. If any file is named
like a normalized result (`normalized.json`, `*.normalized.json`, `repetition-*.json`),
only those files are read, so a raw tool file kept next to its normalized copy is not
counted twice. Files whose format is not recognized are skipped and listed under
`input.ignored_files` in the `--json` report. A directory that mixes formats is an
error. These files are always skipped: vLLM's `*.pytorch.json` sidecar, AIPerf's
`*_aggregate.json` and `*_collated.json`, and `summary.json`, `run.json` or any name
containing `aggregate`.

Field names were checked against each tool's source on its main branch in October 2026.
If a newer release renames a field, the metric shows as missing and the result is
`INCONCLUSIVE` with a reason; it never silently becomes zero.

## What is read

Latency metrics are in milliseconds and can be compared at `mean`, `median`, `p95` and
`p99`, for example `--metric ttft_p99_ms` or `--metric e2el_p95_ms`.

| Metric | vLLM | AIPerf | GuideLLM (successful requests) | SGLang |
| --- | --- | --- | --- | --- |
| `ttft_*_ms` | `mean_ttft_ms`, `median_ttft_ms`, `p95_ttft_ms`, `p99_ttft_ms` | `time_to_first_token` | `time_to_first_token_ms` | `*_ttft_ms` |
| `tpot_*_ms` | `*_tpot_ms` | `inter_token_latency` (the per-request average, which is TPOT) | `time_per_output_token_ms` | `*_tpot_ms` |
| `itl_*_ms` | `*_itl_ms` | not available | `inter_token_latency_ms` | `*_itl_ms` |
| `e2el_*_ms` | `*_e2el_ms` | `request_latency` | `request_latency` (seconds, converted) | `*_e2e_latency_ms` |
| `output_token_throughput` | `output_throughput` | `output_token_throughput` | `output_tokens_per_second` | `output_throughput` |
| `total_token_throughput` | `total_token_throughput` | `total_token_throughput` | `tokens_per_second` | `total_throughput` |
| `request_throughput` | `request_throughput` | `request_throughput` | `requests_per_second` | `request_throughput` |
| successful requests | `completed` | `request_count` | `request_totals.successful` | `completed` |
| failed requests | `failed` | `error_request_count`, or completed minus successful | `request_totals.errored` | `errors` with `--output-details`, otherwise not recorded |
| model | `model_id` | `input_config.models.items[0].name` | `config.backend.model` | `server_info.model_path` |
| GPU | `--metadata gpu=...` | GPU telemetry `gpu_name`, if collected | not recorded | not recorded |

Notes by tool:

- **vLLM** writes only p99 by default. For p95 or end-to-end latency, run with
  `--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 95,99`. Values passed
  with `--metadata gpu=... vllm_version=... image=...` fill the environment. The
  workload parameters compared between runs are `backend`, `endpoint_type`,
  `tokenizer_id`, `num_prompts`, `request_rate`, `burstiness`, `max_concurrency` and the
  ramp-up settings. `label` is not compared, so you can label the two sides differently.
- **AIPerf** values are converted to milliseconds from the `unit` each metric records.
  The compared parameters are the endpoint type, streaming, and each phase's type,
  concurrency, rate and request count.
- **GuideLLM** puts every load point of a sweep in one file. Pass
  `--guidellm-benchmark INDEX` to choose one; the error message lists the indices with
  their strategies when the file holds more than one. Requests cut off by the end of a
  time-bounded run (`incomplete`) are not counted as failures.
- **SGLang** appends one line per run to its output file and writes no failed-request
  count. Without `--output-details` the report notes that failures could not be checked
  instead of returning `INCONCLUSIVE`. With it, a non-empty entry in `errors` counts as
  a failure.

## When the files do not say

Most tools do not record the GPU, and some do not record the model. Missing values are
stored as `unspecified`, the report notes that they could not be compared, and
`--gpu NAME` or `--model NAME` (Action inputs `gpu` and `model`) fill them in. These
flags apply to both sides, so they label the comparison; they cannot detect a run on the
wrong GPU. For that, record the GPU in the results: vLLM's `--metadata gpu=...`, or
AIPerf's GPU telemetry.
