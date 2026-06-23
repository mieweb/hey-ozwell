#!/usr/bin/env bash
# Production focal-loss retrain: SAME data + recipe as the shipped surgical (done) and fpx (hey) models,
# the ONLY change is the loss (HEYBUDDY_FOCAL_GAMMA=2). Reuses the exact prepared .npy files that produced
# the shipped models, so it isolates the loss effect. Outputs to *_focal.onnx — does NOT touch prod.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
export HEYBUDDY_FOCAL_GAMMA=2
P=heybuddy/precalculated
mkdir -p checkpoints/scratch-onnx

train_one () {  # phrase  posfile  negfile  ckpt  onnx  log
  local PH="$1" POSF="$2" NEGF="$3" CKPT="$4" ONNX="$5" LOG="$6"
  local POS=$(python3 -c "import numpy as np;print(len(np.load('$P/$POSF',mmap_mode='r')))")
  echo "=== focal $PH: $POS pos / $NEGF  $(date) ==="
  python -m heybuddy train "$PH" --perceptron \
    --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
    --steps 3500 --stages 3 --target-false-positive-rate 1.5 \
    --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
    --num-batch-threads 3 --augmentation-dataset-streaming \
    --training-dataset "$P/$NEGF" --validation-no-default-dataset --debug \
    > "$LOG" 2>&1
  echo "  $PH train exit=$? ($(date))"
  python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('$CKPT'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '$ONNX', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('exported $ONNX')"
}

train_one "ozwell i'm done" ozwell_i_m_done.npy negs_C.npy \
  checkpoints/ozwell_i_m_done_final.pt checkpoints/scratch-onnx/ozwelldone_surgical_focal.onnx logs/focal_done_train.log
train_one "hey ozwell" hey_ozwell.npy negs_fp_hey.npy \
  checkpoints/hey_ozwell_final.pt checkpoints/scratch-onnx/heyozwell_fpx_focal.onnx logs/focal_hey_train.log
echo "=== FOCAL RETRAIN COMPLETE $(date) ==="
