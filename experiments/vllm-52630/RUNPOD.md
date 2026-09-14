# RunPod Procedure: vLLM #52630

This document is an execution plan only. It does not provision a Pod.

## Preflight

1. Query Secure H100 availability. The intended target is one H100 SXM 80 GB;
   record that it differs from the issue's H100 PCIe 80 GB.
2. Generate a disposable Ed25519 key outside the repository. Use it only with
   Paramiko and delete it after export.
3. Create one continuously running generic CUDA Pod with persistent
   `/workspace`. Do not use a vLLM image as PID 1 and do not reset or replace
   the Pod between versions.
4. Record Pod ID, GPU model, UUID, driver, CUDA runtime, OS, and image digest.
5. Connect with Paramiko, upload the repository with SFTP to
   `/workspace/inference-doctor`, and verify the uploaded file hash.

The issue environment was Ubuntu 24.04.3, CUDA 13.2, driver 595.71.05, and
one H100 PCIe. Record the actual RunPod values instead of treating those values
as assumptions.

## Environment installation preflight

Use Python 3.12 and separate caches. These are the pinned targets; run the
wheel availability checks before installing and abort rather than resolving to
different versions.

```bash
uv venv --python 3.12 /workspace/envs/v0190
uv venv --python 3.12 /workspace/envs/v0210

uv pip install --python /workspace/envs/v0190/bin/python \
  'torch==2.10.0' 'torchaudio==2.10.0' 'torchvision==0.25.0' \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /workspace/envs/v0190/bin/python \
  'vllm==0.19.0' \
  --extra-index-url https://wheels.vllm.ai/0.19.0/cu130 \
  --index-strategy unsafe-best-match
uv pip install --python /workspace/envs/v0190/bin/python \
  'flashinfer-python==0.6.6' 'flashinfer-cubin==0.6.6'

uv pip install --python /workspace/envs/v0210/bin/python \
  'torch==2.11.0' 'torchaudio==2.11.0' 'torchvision==0.26.0' \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /workspace/envs/v0210/bin/python \
  'vllm==0.21.0' \
  --extra-index-url https://wheels.vllm.ai/0.21.0/cu130 \
  --index-strategy unsafe-best-match
uv pip install --python /workspace/envs/v0210/bin/python \
  'flashinfer-python==0.6.8.post1' 'flashinfer-cubin==0.6.8.post1'

for env in /workspace/envs/v0190 /workspace/envs/v0210; do
  uv pip install --python "$env/bin/python" httpx datasets
  "$env/bin/python" -c \
    'import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
done
```

The exact x86_64 wheel files were verified as retrievable before this runbook
was prepared:

```text
vLLM 0.19.0:
https://wheels.vllm.ai/2a69949bdadf0e8942b7a1619b229cb475beef20/vllm-0.19.0%2Bcu130-cp38-abi3-manylinux_2_35_x86_64.whl

vLLM 0.21.0 (the cu130 index serves the release's un-suffixed ABI3 wheel):
https://wheels.vllm.ai/ad7125a431e176d4161099480a66f0169609a690/vllm-0.21.0-cp38-abi3-manylinux_2_24_x86_64.whl
```

Torch 2.10.0+cu130 and 2.11.0+cu130 x86_64 CPython 3.12 wheels are listed in
the official PyTorch cu130 index. The commands above keep vLLM resolution on
the corresponding official vLLM cu130 index and Torch on that exact PyTorch
cu130 index. No source build or alternate version is permitted.

These Torch, torchvision, torchaudio, and FlashInfer pins come from the
corresponding release `requirements/cuda.txt` files. The vLLM 0.19.0 CUDA
wheel index is published by vLLM. The 0.21.0 cu130 index and exact x86_64
wheel are also verified above. Do not replace an unavailable wheel with a
source build or a different Torch version without marking the experiment
invalid.

## Frozen server commands

The only version-specific change is the environment executable:

```bash
/workspace/envs/v0190/bin/vllm serve BAAI/bge-m3 \
  --served-model-name bge \
  --runner pooling \
  --port 8043 \
  --gpu-memory-utilization 0.10 \
  --api-server-count 8 \
  --api-key "$VLLM_API_KEY"

/workspace/envs/v0210/bin/vllm serve BAAI/bge-m3 \
  --served-model-name bge \
  --runner pooling \
  --port 8043 \
  --gpu-memory-utilization 0.10 \
  --api-server-count 8 \
  --api-key "$VLLM_API_KEY"
```

Use `HF_HOME=/workspace/hf-cache`,
`VLLM_CACHE_ROOT=/workspace/vllm-cache-v0190` or `v0210`, and the matching
`TORCHINDUCTOR_CACHE_DIR`. Keep model and dataset caches shared, but keep
compiled artifacts separate.

## Execute on the same GPU

After both environments pass preflight:

```bash
cd /workspace/inference-doctor/experiments/vllm-52630
/workspace/envs/v0190/bin/python run_same_gpu.py --output-dir /workspace/vllm-52630-results
```

The runner records the baseline UUID, starts v0.19.0, waits for `/health`, and
runs exactly three repetitions at each concurrency. It stops the server,
checks that no vLLM process remains, records cleanup evidence, re-reads the GPU
UUID, and aborts if it changed. It then repeats the process for v0.21.0 and
writes A/A, B/B, and A/B comparisons.

## Export and teardown

1. Stop both server processes and verify no vLLM process remains.
2. Save server logs, launch commands, package freezes, GPU metadata, raw JSON,
   normalized JSON, aggregates, comparisons, and `summary.json` under
   `/workspace/vllm-52630-results`.
3. Create and hash the archive:

   ```bash
   tar -czf /workspace/vllm-52630-results.tgz -C /workspace vllm-52630-results
   sha256sum /workspace/vllm-52630-results.tgz > /workspace/vllm-52630-results.tgz.sha256
   ```

4. Download both files with SFTP, verify the archive opens and the SHA256
   matches, then delete the Pod immediately.
5. Delete the disposable local keypair.

No H100 should remain running after export verification.
