# RunPod same-GPU vLLM 0.19.1 vs 0.24.0 runbook

This runbook executes the vLLM issue #48035 experiment on one continuously
running Secure Cloud RTX 4090. Both vLLM versions run sequentially in the
same Pod, in separate Python environments. The Pod image is never changed,
reset, or replaced.

The benchmark workload, prompts, seeds, sampling parameters, server
arguments, repetitions, and comparison thresholds remain defined by
`experiment.json`. This runbook changes only the execution path.

## Compatibility basis

Use the generic NVIDIA CUDA image `nvidia/cuda:12.9.1-devel-ubuntu22.04` as
the outer Pod image, with a long-lived `sleep infinity` command. It is not a
vLLM image and does not start vLLM as PID 1.

The versioned vLLM installation guides document Python 3.12 and the `uv`
installer with `--torch-backend`; the release requirement files pin the
matching PyTorch versions:

- vLLM 0.19.1: PyTorch 2.10.0, torchaudio 2.10.0, torchvision 0.25.0.
- vLLM 0.24.0: PyTorch 2.11.0, torchaudio 2.11.0, torchvision 0.26.0.

Both PyTorch trios are installed from the official CUDA 12.9 index by the
commands below. vLLM 0.24.0 is installed from its official CUDA 12.9 release
wheel explicitly; the unqualified PyPI 0.24.0 package is the default CUDA
13.0 build and must not be used for this run.

References:

- [vLLM 0.19.1 GPU installation](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/)
- [vLLM 0.24.0 GPU installation](https://docs.vllm.ai/en/v0.24.0/getting_started/installation/gpu/)
- [vLLM 0.19.1 CUDA requirements](https://raw.githubusercontent.com/vllm-project/vllm/releases/v0.19.1/requirements/cuda.txt)
- [vLLM 0.24.0 CUDA requirements](https://raw.githubusercontent.com/vllm-project/vllm/releases/v0.24.0/requirements/cuda.txt)
- [vLLM 0.19.1 Dockerfile](https://raw.githubusercontent.com/vllm-project/vllm/releases/v0.19.1/docker/Dockerfile)
- [PyTorch CUDA 12.9 wheels](https://download.pytorch.org/whl/cu129/torch/)
- [vLLM 0.19.1 release](https://github.com/vllm-project/vllm/releases/tag/v0.19.1)
- [vLLM 0.24.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.24.0)

## 0. Local preflight

Do not provision until the GPU price and availability have been checked again.
Use the authenticated RunPod MCP connection to create the Pod; use SSH or
`runpodctl` only for file transfer and remote shell access after creation.

The requested Pod shape is exactly one Secure Cloud NVIDIA GeForce RTX 4090,
with a persistent volume mounted at `/workspace`. Use at least 30 GB of
container disk and 60 GB of persistent volume for the repository, two Python
environments, model cache, vLLM compile cache, and raw results. The Pod must
expose SSH and no vLLM image ports are required.

The Pod creation parameters are:

```text
image: nvidia/cuda:12.9.1-devel-ubuntu22.04
command: bash -lc 'exec sleep infinity'
gpu: NVIDIA GeForce RTX 4090
gpu_count: 1
cloud: Secure Cloud
mount: /workspace
```

Record the returned Pod ID. Do not call any image update, reset, or replace
operation after creation.

## 1. Copy the repository and create the result layout

From WSL, copy the Windows checkout into the persistent volume:

```bash
runpodctl send \
  /mnt/c/Users/dundy/Downloads/projects/inference-doctor \
  "$POD_ID:/workspace/inference-doctor"
```

Inside the Pod:

```bash
set -euo pipefail

export ROOT=/workspace
export REPO="$ROOT/inference-doctor"
export RESULTS="$ROOT/vllm-48035-results-same-gpu"
export HF_HOME="$ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export VLLM_CACHE_ROOT="$ROOT/vllm-cache"
export UV_CACHE_DIR="$ROOT/uv-cache"
export PATH="$HOME/.local/bin:$PATH"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates curl git jq procps psmisc
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

mkdir -p "$RESULTS" "$ROOT/envs" "$HF_HOME" "$VLLM_CACHE_ROOT"
cd "$REPO"
uv venv --python 3.12 --seed --managed-python "$ROOT/inference-doctor/.venv"
"$ROOT/inference-doctor/.venv/bin/python" -m pip install -e .
```

Keep the previous local export separate. It is historical evidence only and
must not be copied into either the new `baseline` or `candidate` directory.

## 2. Record the GPU identity before installing vLLM

```bash
mkdir -p "$RESULTS/metadata" "$RESULTS/baseline" "$RESULTS/candidate"
nvidia-smi | tee "$RESULTS/metadata/nvidia-smi.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap \
  --format=csv,noheader,nounits | tee "$RESULTS/metadata/gpu-before.csv"
uname -a | tee "$RESULTS/metadata/uname.txt"
cat /etc/os-release | tee "$RESULTS/metadata/os-release.txt"
nvcc --version | tee "$RESULTS/metadata/nvcc.txt" || true
```

Save the UUID from `gpu-before.csv` as the immutable identity for this run.
If `nvidia-smi` fails, stop and export the diagnostics; do not install or
benchmark.

## 3. Create the two isolated vLLM environments

The following commands are the exact version-specific install paths. They
must be run once, before either server starts:

```bash
set -euo pipefail

uv venv --python 3.12 --seed --managed-python /workspace/envs/v0191
uv pip install \
  --python /workspace/envs/v0191/bin/python \
  --torch-backend=cu129 \
  "vllm==0.19.1"

uv venv --python 3.12 --seed --managed-python /workspace/envs/v0240
uv pip install \
  --python /workspace/envs/v0240/bin/python \
  --torch-backend=cu129 \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
```

Verify both environments before serving:

```bash
/workspace/envs/v0191/bin/python -c \
  'import sys, torch, vllm; print(sys.version); print(vllm.__version__); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'

/workspace/envs/v0240/bin/python -c \
  'import sys, torch, vllm; print(sys.version); print(vllm.__version__); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Expected version checks are vLLM `0.19.1` with PyTorch `2.10.0+cu129`, and
vLLM `0.24.0` with PyTorch `2.11.0+cu129`. If either resolver selects a
different CUDA backend or either CUDA sanity check fails, preserve the install
logs and stop. Do not substitute a wheel, image, or server argument silently.

## 4. Freeze and record the server arguments

Use the same argument list for both versions, matching `experiment.json`:

```bash
cat > "$RESULTS/server-args.txt" <<'EOF'
Qwen/Qwen3-4B-Instruct-2507
--host 0.0.0.0
--port 8000
--tensor-parallel-size 1
--max-model-len 8192
--gpu-memory-utilization 0.7
--max-num-seqs 8
--max-num-batched-tokens 2048
--dtype float16
--enable-prefix-caching
--enable-chunked-prefill
--default-chat-template-kwargs {"enable_thinking":false}
EOF
```

Do not add candidate-only flags or change the workload after observing a
result. A version that rejects these frozen arguments is a compatibility
blocker and invalidates the experiment rather than justifying a methodology
change.

## 5. Run the baseline in its isolated environment

Start the server in a dedicated process group so it can be fully stopped
without affecting the next environment:

```bash
mkdir -p "$RESULTS/baseline"
setsid env HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" \
  VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  /workspace/envs/v0191/bin/vllm serve \
  Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --max-model-len 8192 --gpu-memory-utilization 0.7 \
  --max-num-seqs 8 --max-num-batched-tokens 2048 --dtype float16 \
  --enable-prefix-caching --enable-chunked-prefill \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  > "$RESULTS/baseline/server.log" 2>&1 < /dev/null &
echo $! > "$RESULTS/baseline/server.pid"
```

Poll until healthy, then run the existing fixed three-repetition client:

```bash
until curl --fail --silent http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
curl --fail --silent http://127.0.0.1:8000/v1/models > "$RESULTS/baseline/models.json"

cd "$REPO"
/workspace/inference-doctor/.venv/bin/python \
  experiments/vllm-48035/benchmark_client.py \
  --server-url http://127.0.0.1:8000 \
  --version-label 0.19.1 \
  --output-dir "$RESULTS/baseline"
```

Copy the baseline GPU and runtime metadata into the result directory before
stopping the server.

## 6. Stop the baseline and verify GPU release

```bash
BASE_PID=$(cat "$RESULTS/baseline/server.pid")
BASE_PGID=$(ps -o pgid= -p "$BASE_PID" | tr -d ' ')
kill -TERM -- -"$BASE_PGID" 2>/dev/null || true

for _ in $(seq 1 60); do
  if ! kill -0 "$BASE_PID" 2>/dev/null; then break; fi
  sleep 2
done

ps -eo pid,pgid,args | grep '[v]llm' > "$RESULTS/baseline/processes-after-stop.txt" || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits | tee "$RESULTS/baseline/compute-apps-after-stop.csv"
nvidia-smi --query-gpu=uuid,memory.used \
  --format=csv,noheader,nounits | tee "$RESULTS/baseline/gpu-memory-after-stop.csv"
```

The compute-apps file must be empty and the GPU memory must return to the
idle baseline before starting 0.24.0. If the dedicated process group does not
exit, record the process tree, send TERM once more to that group, then use a
bounded KILL for that group only. Never use a broad `pkill`.

## 7. Run the candidate without changing the Pod

Record the GPU identity immediately before candidate startup and require an
exact UUID match:

```bash
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap \
  --format=csv,noheader,nounits | tee "$RESULTS/candidate/gpu-before.csv"

BASE_UUID=$(awk -F, '{gsub(/ /,"",$2); print $2}' "$RESULTS/metadata/gpu-before.csv")
CAND_UUID=$(awk -F, '{gsub(/ /,"",$2); print $2}' "$RESULTS/candidate/gpu-before.csv")
test -n "$BASE_UUID" && test "$BASE_UUID" = "$CAND_UUID" || {
  echo "ABORT: GPU UUID changed: baseline=$BASE_UUID candidate=$CAND_UUID" >&2
  exit 2
}
```

Start only the candidate environment with the same server arguments:

```bash
setsid env HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" \
  VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  /workspace/envs/v0240/bin/vllm serve \
  Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --max-model-len 8192 --gpu-memory-utilization 0.7 \
  --max-num-seqs 8 --max-num-batched-tokens 2048 --dtype float16 \
  --enable-prefix-caching --enable-chunked-prefill \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  > "$RESULTS/candidate/server.log" 2>&1 < /dev/null &
echo $! > "$RESULTS/candidate/server.pid"

until curl --fail --silent http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
curl --fail --silent http://127.0.0.1:8000/v1/models > "$RESULTS/candidate/models.json"

cd "$REPO"
/workspace/inference-doctor/.venv/bin/python \
  experiments/vllm-48035/benchmark_client.py \
  --server-url http://127.0.0.1:8000 \
  --version-label 0.24.0 \
  --output-dir "$RESULTS/candidate"
```

Stop the candidate with the same dedicated process-group procedure and save
the final GPU-memory evidence before export.

## 8. Aggregate, compare, and validate

Run the existing offline comparison after both result sets are complete:

```bash
cd "$REPO"
/workspace/inference-doctor/.venv/bin/python \
  experiments/vllm-48035/compare_results.py \
  --baseline-dir "$RESULTS/baseline" \
  --candidate-dir "$RESULTS/candidate" \
  --output-dir "$RESULTS/comparisons"
```

This regenerates normalized results from raw JSON, aggregates each version,
checks the embedded GPU UUIDs, and writes baseline-vs-baseline,
candidate-vs-candidate, and baseline-vs-candidate comparisons. The expected
validation shape remains:

```text
baseline vs baseline   PASS
candidate vs candidate PASS
baseline vs candidate  FAIL only if the regression exceeds 10%
```

If either same-version control fails, the cross-version result is not valid.
If the UUID check fails, the cross-version result must not be labeled valid.

## 9. Export and terminate

Create a manifest on the persistent volume, copy the complete result tree to
the local machine, and verify the raw files, metadata, logs, aggregates, all
three comparisons, and checksums before deleting the Pod:

```bash
cd "$RESULTS"
find . -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS
```

From WSL:

```bash
runpodctl receive "$POD_ID:/workspace/vllm-48035-results-same-gpu" \
  ./vllm-48035-results-same-gpu-export
```

Verify the local export contains `baseline/repetition-*`,
`candidate/repetition-*`, `environment.json`, `gpu-before.csv`, server logs,
`comparisons/`, `summary.json`, and `SHA256SUMS`. Only then delete the Pod
through the authenticated RunPod integration and confirm it is gone.
