#!/usr/bin/env bash
# Middle-ratio run + ACTIVE false-positive suppression.
# Changes vs run_rebalanced.sh (10k negs):
#   - external negatives 10k -> 22k (12k libri + 10k free): ratio between the 44%/74% runs
#   - adds a held-out VALIDATION negative set (negs_val.npy) so the trainer's per-hour FP
#     target (default 1.5/hr) is actually enforced via --dynamic-negative-weight. Prior runs
#     had this disabled (--validation-no-default-dataset -> validation FP rate was nan).
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
RES=logs/FINAL_RESULTS_MIDRATIO.txt; TAG=midratio
ONNX=checkpoints/scratch-onnx/ozwell_done_${TAG}.onnx
echo "=== ${TAG} pipeline started $(date) ===" > "$RES"
echo "negs: 22k train (12k libri + 10k free) + 5k held-out val; FP target 1.5/hr ENFORCED" | tee -a "$RES"

echo "[1/3] training ($(date)) -> logs/${TAG}_train.log" | tee -a "$RES"
python -m heybuddy train "ozwell i'm done" --perceptron \
  --positive-samples 5000 --adversarial-samples 5000 \
  --steps 1000 --stages 3 \
  --validation-samples 5000 --testing-positive-samples 5000 --testing-adversarial-samples 5000 \
  --num-batch-threads 2 --augmentation-dataset-streaming \
  --training-dataset heybuddy/precalculated/negs_mid.npy \
  --validation-no-default-dataset \
  --validation-dataset heybuddy/precalculated/negs_val.npy \
  --target-false-positive-rate 1.5 --debug > logs/${TAG}_train.log 2>&1
echo "  train exit=$? ($(date))" | tee -a "$RES"

echo "[2/3] exporting ONNX..." | tee -a "$RES"
mkdir -p checkpoints/scratch-onnx
python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/ozwell_i_m_done_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('exported')
" 2>&1 | grep -E "exported|Error" | tee -a "$RES"

echo "[3/3] eval ($(date)):" | tee -a "$RES"
cd eval
echo "--- ${TAG} / ElevenLabs held-out test ---" | tee -a "../$RES"
python evaluate_wakeword.py --model ../${ONNX} \
  --positives /tmp/eval/done/pos --negatives /tmp/eval/done/neg --pretrained-dir pretrained \
  --label ${TAG} 2>&1 | grep -E "recall=|POSITIVE|NEGATIVE" | tee -a "../$RES"
echo "--- in-domain Piper ---" | tee -a "../$RES"
python evaluate_wakeword.py --model ../${ONNX} \
  --positives /tmp/eval/piper_pos --negatives /tmp/eval/done/neg --pretrained-dir pretrained \
  --label ${TAG}-piper 2>&1 | grep -E "recall=|POSITIVE|NEGATIVE" | tee -a "../$RES"
cd ..
echo "=== ${TAG} pipeline DONE $(date) ===" | tee -a "$RES"
