#!/usr/bin/env bash
# Build a DEDICATED GPU venv for data-generation (extract / gen_* / augmentation).
# WHY a separate venv: the only onnxruntime-gpu that works on these Volta V100s (sm_70) is the
# CUDA-11 / cuDNN-8 build (1.18.1), which requires numpy<2 — but the training venv (model/.venv)
# runs numpy 2.x. So we isolate the GPU stack here and leave training untouched.
# Benchmark that motivated this: the embedding step runs ~47x faster on a V100 (798 -> ~38,000 clips/sec).
# Volta (V100, sm_70) needs the CUDA-11/cuDNN-8 stack below; newer onnxruntime-gpu 1.26 + cuDNN 9 fails
# with "no kernel image available". This only speeds DATA GENERATION (extract/gen_*), not cached training.
#
# RUN THIS AFTER the negsweep finishes (it touches no shared state, but free up CPU/RAM first).
# Then VALIDATE with the smoke test at the bottom before trusting it in the pipeline.
set -euo pipefail
cd "$(dirname "$0")"   # model/

GPU_VENV=".venv-gpu"
export UV_LINK_MODE=copy

echo "=== [1/4] create $GPU_VENV (python 3.11) ==="
uv venv --python 3.11 "$GPU_VENV"
source "$GPU_VENV/bin/activate"

echo "=== [2/4] install heybuddy deps, pinned to numpy<2 ==="
# NOTE: validate this resolves cleanly — piper-phonemize + torch stack on numpy<2 is the risk.
uv pip install "numpy<2"
uv pip install -r requirements.txt
uv pip install piper-phonemize || echo "WARN: piper-phonemize failed — only needed if generating NEW positives"
# match the CUDA-12.4 driver for torch (training-side libs); augmentation uses torch on GPU too
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 || true

echo "=== [3/4] install the PROVEN Volta GPU onnxruntime stack (CUDA 11 / cuDNN 8) ==="
uv pip uninstall onnxruntime || true            # remove any CPU build
uv pip install "onnxruntime-gpu==1.18.1"        # the CUDA 11 wheel
uv pip install "nvidia-cudnn-cu11==8.9.6.50"    # cuDNN 8 (NOT 9 — 9 fails on sm_70)
uv pip install nvidia-cublas-cu11 nvidia-cuda-runtime-cu11 nvidia-curand-cu11 \
               nvidia-cufft-cu11 nvidia-cusparse-cu11 nvidia-cuda-nvrtc-cu11

echo "=== [4/4] write the LD_LIBRARY_PATH wrapper ==="
# onnxruntime needs the pip nvidia libs on the loader path. Source this before any GPU command.
cat > "$GPU_VENV/gpu-env.sh" <<'WRAP'
# usage: source model/.venv-gpu/bin/activate && source model/.venv-gpu/gpu-env.sh
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(ls -d $SP/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
WRAP

echo
echo "=== DONE. Validate with the smoke test: ==="
echo "  source $GPU_VENV/bin/activate && source $GPU_VENV/gpu-env.sh"
echo '  python3 -c "import onnxruntime as o,numpy as np; \'
echo '    s=o.InferenceSession(\"heybuddy/pretrained/speech-embedding.onnx\", \'
echo '      providers=[(\"CUDAExecutionProvider\",{\"device_id\":\"0\"}),\"CPUExecutionProvider\"]); \'
echo '    print(\"ACTIVE:\", s.get_providers()[0]); \'
echo '    print(s.run(None,{s.get_inputs()[0].name: np.random.randn(64,76,32,1).astype(\"float32\")})[0].shape)"'
echo
echo "Expect: ACTIVE: CUDAExecutionProvider  +  (64, 1, 1, 96)"
echo
echo "=== THEN run data generation on GPU by adding --device-id 0, e.g.: ==="
echo "  python -m heybuddy extract negs_peoples2 MLCommons/peoples_speech --config clean --split train \\"
echo "    --audio-key audio --transcript-key text --hours 22 --streaming --trust-remote-code --device-id 0"
