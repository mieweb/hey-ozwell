#!/usr/bin/env bash
# TRACK 1 (free): does POSITIVE diversity lift held-out recall off 64%?
# Inject 15k diverse Piper positives (en_GB-vctk = 109 UK speakers + 5 extra en_US voices,
# varied speed/noise) into the 100k libritts-en-us positives. Test on held-out ElevenLabs
# (untouched; speaker-leaky with its own train split, so kept strictly as test).
# Same engine (Piper) so this tests accent/voice diversity, NOT cross-vendor — that needs the cloud key.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
RES=logs/FINAL_RESULTS_DIVERSE.txt; TAG=diverse
ONNX=checkpoints/scratch-onnx/ozwell_done_${TAG}.onnx
echo "=== ${TAG} started $(date) ===" > "$RES"
echo "+15k diverse Piper positives (VCTK 109 UK + 5 US) into 100k libritts; test=held-out ElevenLabs" | tee -a "$RES"

echo "[1/4] generate diverse positives ($(date)) -> logs/diverse_gen.log ..." | tee -a "$RES"
python gen_diverse_positives.py 15000 heybuddy/precalculated/diverse_pos.npy > logs/diverse_gen.log 2>&1
echo "  gen exit=$? : $(tail -1 logs/diverse_gen.log)" | tee -a "$RES"

echo "[2/4] back up clean 100k + inject diverse into positive cache" | tee -a "$RES"
python3 -c "
import numpy as np, shutil, os
src='heybuddy/precalculated/ozwell_i_m_done.npy'
bak='heybuddy/precalculated/ozwell_i_m_done.libritts100k.npy'
if not os.path.exists(bak): shutil.copy(src, bak)   # preserve the clean 100k libritts
base=np.load(bak); div=np.load('heybuddy/precalculated/diverse_pos.npy')
both=np.concatenate([base,div]); np.save(src, both)
print('positives now', both.shape, '=', len(base),'libritts +',len(div),'diverse')
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"

POS=$(python3 -c "import numpy as np; print(len(np.load('heybuddy/precalculated/ozwell_i_m_done.npy')))" 2>/dev/null)
echo "[3/4] retrain with $POS positives ($(date)) -> logs/diverse_train.log" | tee -a "$RES"
python -m heybuddy train "ozwell i'm done" --perceptron \
  --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 5000 --stages 3 --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 3 --augmentation-dataset-streaming \
  --training-dataset heybuddy/precalculated/negs_all.npy \
  --validation-no-default-dataset --debug > logs/diverse_train.log 2>&1
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
echo "--- ${TAG} / ElevenLabs HELD-OUT test ---" | tee -a "../$RES"
python evaluate_wakeword.py --model ../${ONNX} --positives /tmp/eval/done/pos --negatives /tmp/eval/done/neg \
  --pretrained-dir pretrained --label ${TAG} 2>&1 | grep -E "recall=|POSITIVE|NEGATIVE" | tee -a "../$RES"
echo "--- in-domain Piper ---" | tee -a "../$RES"
python evaluate_wakeword.py --model ../${ONNX} --positives /tmp/eval/piper_pos --negatives /tmp/eval/done/neg \
  --pretrained-dir pretrained --label ${TAG}-piper 2>&1 | grep -E "recall=|POSITIVE|NEGATIVE" | tee -a "../$RES"
cd ..
echo "=== ${TAG} DONE $(date) ===" | tee -a "$RES"
