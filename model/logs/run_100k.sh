#!/usr/bin/env bash
# FULL 100k production run for "ozwell i'm done".
# Goal: move the whole recall<->FP tradeoff curve UP via positive diversity (the lesson
# from the 5k fast runs: knob-tuning just slides along a too-low curve; only data moves it).
# - 100k diverse Piper positives + 100k adversarial near-phrase negatives (both peak-normed
#   via the committed embeddings.py loudness fix)
# - plus our matched real-speech negatives (negs_all.npy = 37.5k libri + 15k freesound)
# - NO aggressive validation FP enforcement: we want to SEE the raw curve at full diversity
#   (a follow-up shorter run can clamp FP down from a higher starting curve).
# Reduced val/test sample counts to protect the 22GB RAM ceiling (we eval externally anyway).
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
RES=logs/FINAL_RESULTS_100k.txt; TAG=100k
ONNX=checkpoints/scratch-onnx/ozwell_done_${TAG}.onnx
echo "=== ${TAG} pipeline started $(date) ===" > "$RES"
echo "100k pos + 100k adv + negs_all (52.5k real-speech matched negs); 5000 steps x3 stages" | tee -a "$RES"

echo "[1/3] training ($(date)) -> logs/${TAG}_train.log" | tee -a "$RES"
python -m heybuddy train "ozwell i'm done" --perceptron \
  --positive-samples 100000 --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 5000 --stages 3 \
  --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 1 --augmentation-dataset-streaming \
  --training-dataset heybuddy/precalculated/negs_all.npy \
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
