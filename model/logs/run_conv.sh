#!/usr/bin/env bash
# EXPERIMENT: conversational negatives. Same as run_100k EXCEPT the negative set swaps
# negs_all (LibriSpeech read-speech + freesound sounds) -> negs_conv (20k of those + 653
# ElevenLabs CONVERSATIONAL train negatives oversampled 12x = 28% conversational).
# Hypothesis: the model over-fires on conversational speech because it never trained on
# any; in-domain conversational negatives should drop the held-out FP rate.
# threads=3 (RAM now 59GB); reuses the cached 100k positives so it's far faster than yesterday.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
RES=logs/FINAL_RESULTS_CONV.txt; TAG=conv
ONNX=checkpoints/scratch-onnx/ozwell_done_${TAG}.onnx
echo "=== ${TAG} pipeline started $(date) ===" > "$RES"
echo "100k pos + 100k adv + negs_conv (20k base + 28% ElevenLabs conversational)" | tee -a "$RES"

echo "[1/3] training ($(date)) -> logs/${TAG}_train.log" | tee -a "$RES"
python -m heybuddy train "ozwell i'm done" --perceptron \
  --positive-samples 100000 --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 5000 --stages 3 \
  --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 3 --augmentation-dataset-streaming \
  --training-dataset heybuddy/precalculated/negs_conv.npy \
  --validation-no-default-dataset --debug > logs/${TAG}_train.log 2>&1
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
