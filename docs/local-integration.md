# Reproducing the vLLM integration

This optional harness runs Inference Doctor against Prometheus telemetry scraped from an official vLLM OpenAI-compatible container. It is intentionally small: vLLM, Prometheus, and the existing CLI only. It does not add Grafana, DCGM, a dashboard, or product code.

The checked-in compose file pins the validated image and model combination:

- vLLM image: `vllm/vllm-openai:v0.29.0-cu129`
- model: `Qwen/Qwen3-0.6B`
- Prometheus scrape interval: 5 seconds

## Docker Desktop / WSL2

Requirements:

- Docker Desktop with the WSL2 backend and NVIDIA GPU support.
- Python 3.12 on Windows or in the WSL2 distribution.
- An optional `HF_TOKEN` in the shell environment for a gated model. Never commit a token or an `.env` file.

From the repository root, start the services:

```powershell
$env:VLLM_MODEL = "Qwen/Qwen3-0.6B"
docker compose up -d
docker compose logs -f vllm
```

The image entrypoint already runs `vllm serve`, so the compose command supplies only model and server arguments. Wait until the OpenAI-compatible server is ready, then verify both endpoints:

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/metrics
curl.exe "http://localhost:9090/api/v1/query?query=up%7Bjob%3D%22vllm%22%7D"
```

Generate enough traffic for the selected Prometheus window. This single request is only a smoke test:

```powershell
curl.exe http://localhost:8000/v1/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"$env:VLLM_MODEL\",\"prompt\":\"Say hello from vLLM metrics.\",\"max_tokens\":16}"
```

Install and run the CLI with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
inference-doctor diagnose `
  --prometheus http://localhost:9090 `
  --window 5m `
  --ttft-slo-ms 400 `
  --tpot-slo-ms 50
```

Stop the harness with `docker compose down`. Use `docker compose down -v` only when intentionally removing the cached model data as well.

## Optional disposable RunPod reproduction

This path was validated on a Secure Cloud NVIDIA L4, but it does not require a particular region, account, or pod identifier. Use a disposable GPU pod with sufficient disk for the image, model cache, repository, and Prometheus data.

1. Start `vllm/vllm-openai:v0.29.0-cu129` and expose the vLLM port only as required by your access controls.
2. Because the image entrypoint is `vllm serve`, pass only `Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000` as container arguments.
3. Verify `nvidia-smi`, then `curl http://127.0.0.1:8000/health` and `curl http://127.0.0.1:8000/metrics` from the pod.
4. Run Prometheus in the same private network namespace or container host, using a scrape target of `127.0.0.1:8000`; the supplied [prometheus.yml](../prometheus/prometheus.yml) has the equivalent Compose target.
5. Send controlled requests, wait for at least two scrapes, and run the local CLI against the Prometheus endpoint or an authenticated proxy.
6. Tear down the disposable pod when finished to stop charges.

Do not publish unauthenticated vLLM or Prometheus endpoints. No account credentials, pod IDs, local paths, or model tokens are needed in this repository.

## Metric compatibility

The v0.2 collector queries vLLM metric series for TTFT, inter-token latency, queue time, prefill time, prompt tokens, running/waiting requests, KV-cache usage, and preemptions. The preemption counter is queried as `vllm:num_preemptions_total`, which is the Prometheus counter series for the documented preemption metric family. Check the [README signal table](../README.md#v02-signals) against the `/metrics` endpoint of the vLLM version you deploy.
