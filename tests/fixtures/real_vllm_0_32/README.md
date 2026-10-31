Real `/metrics` scrapes from vLLM 0.32.0 (`vllm/vllm-openai:latest`) serving
Qwen/Qwen2.5-0.5B-Instruct on one RunPod RTX A5000 (Secure Cloud, $0.27/h), 2026-10-31.
Load: 32 concurrent chat requests (shared ~200-token system prompt, max_tokens 256) through
the RunPod HTTP proxy. `t1.prom` was taken 60 s after `t0.prom` (unix 1793423374 -> 1793423434).
Prefix caching on (vLLM default); no DCGM exporter on the pod.
