#!/usr/bin/env bash
# CLEAN, NO-LEAKAGE experiment. One variable changed vs run_100k:
#   negs_all (37.5k LibriSpeech READ-aloud + 15k freesound)
#   -> negs_peoples_all (37.5k People's Speech SPONTANEOUS real speech + 15k freesound)
# Diagnosis (exp D) said the model over-fires on conversational/spontaneous speech because it
# only trained on read-aloud audiobooks. People's Speech is real, spontaneous, and a DIFFERENT
# source from both Piper (train positives) and ElevenLabs (test) -> a FP drop on the held-out
# ElevenLabs test is an honest generalization signal, not leakage. No oversampling.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
RES=logs/FINAL_RESULTS_PEOPLES.txt; TAG=peoples
ONNX=checkpoints/scratch-onnx/ozwell_done_${TAG}.onnx
echo "=== ${TAG} started $(date) ===" > "$RES"
echo "A/B vs run_100k: LibriSpeech read -> People's Speech spontaneous (real, different source); test=held-out ElevenLabs" | tee -a "$RES"

echo "[1/4] extract People's Speech negatives ($(date)) -> logs/peoples_extract.log ..." | tee -a "$RES"
rm -rf heybuddy/precalculated/negs_peoples
python -m heybuddy extract negs_peoples MLCommons/peoples_speech \
  --config clean --split train --audio-key audio --transcript-key text \
  --hours 15 --streaming --trust-remote-code --device-id 0 --debug > logs/peoples_extract.log 2>&1
echo "  extract exit=$? ($(date))" | tee -a "$RES"

echo "[2/4] combine peoples + freesound -> negs_peoples_all.npy" | tee -a "$RES"
python3 -c "
import numpy as np, glob
pe=sorted(glob.glob('heybuddy/precalculated/negs_peoples/*.npy'))
fr=sorted(glob.glob('heybuddy/precalculated/negs_pk/*.npy'))
arrs=[np.load(f) for f in pe+fr]
alln=np.concatenate(arrs); np.save('heybuddy/precalculated/negs_peoples_all.npy', alln)
print('combined', alln.shape, 'from', len(pe),'peoples-files +',len(fr),'free-files',
      '| mean(frames)', round(float(alln[:,:16,:].mean()),2))
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"

echo "[3/4] training ($(date)) -> logs/peoples_train.log (reuses cached 100k positives)" | tee -a "$RES"
python -m heybuddy train "ozwell i'm done" --perceptron \
  --positive-samples 100000 --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 5000 --stages 3 --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 3 --augmentation-dataset-streaming \
  --training-dataset heybuddy/precalculated/negs_peoples_all.npy \
  --validation-no-default-dataset --debug > logs/peoples_train.log 2>&1
echo "  train exit=$? ($(date))" | tee -a "$RES"

echo "[4/4] export + eval ($(date)):" | tee -a "$RES"
mkdir -p checkpoints/scratch-onnx
python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/ozwell_i_m_done_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('exported')
" 2>&1 | grep -E "exported|Error" | tee -a "$RES"
cd eval
echo "--- ${TAG} / ElevenLabs HELD-OUT test (honest — different source from training) ---" | tee -a "../$RES"
python evaluate_wakeword.py --model ../${ONNX} \
  --positives /tmp/eval/done/pos --negatives /tmp/eval/done/neg --pretrained-dir pretrained \
  --label ${TAG} 2>&1 | grep -E "recall=|POSITIVE|NEGATIVE" | tee -a "../$RES"
echo "--- in-domain Piper ---" | tee -a "../$RES"
python evaluate_wakeword.py --model ../${ONNX} \
  --positives /tmp/eval/piper_pos --negatives /tmp/eval/done/neg --pretrained-dir pretrained \
  --label ${TAG}-piper 2>&1 | grep -E "recall=|POSITIVE|NEGATIVE" | tee -a "../$RES"
cd ..
echo "=== ${TAG} DONE $(date) ===" | tee -a "$RES"
