#!/usr/bin/env bash
# Unattended pipeline: waits for the LibriSpeech extract, combines speech+sound
# negatives, retrains ozwell-i'm-done at matched scale, exports, and evals.
# Runs fully in tmux so it completes regardless of the chat/SSH session.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
RES=logs/FINAL_RESULTS.txt
echo "=== pipeline started $(date) ===" > "$RES"

echo "[1/5] waiting for LibriSpeech extract (tmux exl)..." | tee -a "$RES"
while [ ! -f logs/exl.done ]; do sleep 15; done
echo "  extract finished: $(cat logs/exl.done)" | tee -a "$RES"

echo "[2/5] combining speech (libri) + sound (freesound) negatives..." | tee -a "$RES"
python3 -c "
import numpy as np, glob
libri=sorted(glob.glob('heybuddy/precalculated/negs_libri/*.npy'))
free =sorted(glob.glob('heybuddy/precalculated/negs_pk/*.npy'))
arrs=[np.load(f) for f in libri+free]
alln=np.concatenate(arrs)
np.save('heybuddy/precalculated/negs_all.npy', alln)
print('combined', alln.shape, 'from', len(libri),'libri +',len(free),'free')
" 2>&1 | grep -vE "pthread|Warning" | tee -a "$RES"

echo "[3/5] training ($(date)) -> logs/pipeline_train.log" | tee -a "$RES"
python -m heybuddy train "ozwell i'm done" --perceptron \
  --positive-samples 5000 --adversarial-samples 5000 \
  --steps 1000 --stages 3 \
  --validation-samples 5000 --testing-positive-samples 5000 --testing-adversarial-samples 5000 \
  --num-batch-threads 2 --augmentation-dataset-streaming \
  --training-dataset heybuddy/precalculated/negs_all.npy \
  --validation-no-default-dataset --debug > logs/pipeline_train.log 2>&1
echo "  train exit=$? ($(date))" | tee -a "$RES"

echo "[4/5] exporting ONNX..." | tee -a "$RES"
mkdir -p checkpoints/scratch-onnx
python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/ozwell_i_m_done_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), 'checkpoints/scratch-onnx/ozwell_done_speech.onnx', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('exported')
" 2>&1 | grep -E "exported|Error" | tee -a "$RES"

echo "[5/5] eval @ production scale ($(date)):" | tee -a "$RES"
cd eval
echo "--- RETRAINED (speech+sound negs) / ElevenLabs held-out test ---" | tee -a "../$RES"
python evaluate_wakeword.py --model ../checkpoints/scratch-onnx/ozwell_done_speech.onnx \
  --positives /tmp/eval/done/pos --negatives /tmp/eval/done/neg --pretrained-dir pretrained \
  --label retrained 2>&1 | grep -E "recall=|POSITIVE|NEGATIVE" | tee -a "../$RES"
echo "--- in-domain Piper ---" | tee -a "../$RES"
python evaluate_wakeword.py --model ../checkpoints/scratch-onnx/ozwell_done_speech.onnx \
  --positives /tmp/eval/piper_pos --negatives /tmp/eval/done/neg --pretrained-dir pretrained \
  --label retrained-piper 2>&1 | grep -E "recall=|POSITIVE|NEGATIVE" | tee -a "../$RES"
cd ..
echo "=== pipeline DONE $(date) — read this file for results ===" | tee -a "$RES"
