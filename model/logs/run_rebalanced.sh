#!/usr/bin/env bash
# Rebalanced fast-signal run: identical to run_pipeline.sh EXCEPT external negatives
# are capped 52.5k -> 10k (5k LibriSpeech speech + 5k freesound) to test whether the
# ~10:1 negative:positive imbalance was crushing recall. Single-variable A/B.
# Usage: run_rebalanced.sh [probe]   (probe = tiny 30-step smoke test of the wiring)
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate

if [ "${1:-}" = "probe" ]; then
  RES=logs/PROBE_REBAL.txt; TAG=probe; STEPS=30
  POS=300; ADV=300; VAL=500; TST=500
else
  RES=logs/FINAL_RESULTS_REBAL.txt; TAG=rebal; STEPS=1000
  POS=5000; ADV=5000; VAL=5000; TST=5000
fi
ONNX=checkpoints/scratch-onnx/ozwell_done_${TAG}.onnx

echo "=== ${TAG} pipeline started $(date) ===" > "$RES"
echo "negs: 10k (5k libri + 5k free)  vs prior 52.5k" | tee -a "$RES"

echo "[1/3] training ($(date)) -> logs/${TAG}_train.log" | tee -a "$RES"
python -m heybuddy train "ozwell i'm done" --perceptron \
  --positive-samples $POS --adversarial-samples $ADV \
  --steps $STEPS --stages 3 \
  --validation-samples $VAL --testing-positive-samples $TST --testing-adversarial-samples $TST \
  --num-batch-threads 2 --augmentation-dataset-streaming \
  --training-dataset heybuddy/precalculated/negs_bal.npy \
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
