#!/usr/bin/env bash
# HEY OZWELL accent retrain — applies the PROVEN ozwell-done recipe (config C, 160k negs) to the START word.
# Builds a NEW model from scratch; does NOT touch Amanda's shipped hey-ozwell. Reuses negs_C (phrase-agnostic).
# Isolated artifacts: hey_*.npy positives + /tmp/eval/hey_accent/ test clips (ozwell-done artifacts untouched).
# Run in tmux:  tmux new -s heyozwell 'bash logs/run_heyozwell.sh'
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
P=heybuddy/precalculated
RES=logs/FINAL_RESULTS_HEYOZWELL.txt
echo "=== hey ozwell accent retrain $(date) ===" > "$RES"

rm -rf /tmp/eval/hey_accent   # clean any smoke-test leftovers so the test set is exactly the held-out voices

# [1/6] American base: trigger heybuddy's Piper positive + adversarial generation (throwaway 50-step train).
#       Generates hey_ozwell.npy / _adv / _val / _tst caches; we back up the clean 100k base.
echo "[1/6] generate 100k American Piper base ($(date))..." | tee -a "$RES"
python -m heybuddy train "hey ozwell" --perceptron \
  --positive-samples 100000 --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 50 --stages 1 --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 3 --augmentation-dataset-streaming \
  --training-dataset $P/negs_C.npy --validation-no-default-dataset --debug \
  > logs/heyozwell_base.log 2>&1
echo "  base gen exit=$? : hey_ozwell.npy = $(python3 -c "import numpy as np; print(np.load('$P/hey_ozwell.npy').shape)" 2>/dev/null)" | tee -a "$RES"
if [ ! -f "$P/hey_ozwell.npy" ]; then echo "  FATAL: base positives not generated — aborting" | tee -a "$RES"; exit 1; fi
cp "$P/hey_ozwell.npy" "$P/hey_ozwell.libritts100k.npy"

# [2/6] diverse Piper positives (VCTK UK + US voices)
echo "[2/6] diverse Piper positives ($(date))..." | tee -a "$RES"
python gen_diverse_positives_hey.py 15000 "$P/hey_diverse_pos.npy" > logs/heyozwell_diverse.log 2>&1
echo "  diverse exit=$? : $(tail -1 logs/heyozwell_diverse.log)" | tee -a "$RES"

# [3/6] Google accents (+ held-out test wavs -> /tmp/eval/hey_accent)
echo "[3/6] Google accent positives ($(date))..." | tee -a "$RES"
python gen_accent_full_hey.py > logs/heyozwell_accent.log 2>&1
echo "  google exit=$? : $(tail -1 logs/heyozwell_accent.log)" | tee -a "$RES"

# [4/6] Azure accents (+ test wavs)
echo "[4/6] Azure accent positives ($(date))..." | tee -a "$RES"
python gen_azure_accents_hey.py > logs/heyozwell_azure.log 2>&1
echo "  azure exit=$? : $(tail -1 logs/heyozwell_azure.log)" | tee -a "$RES"

# [5/6] assemble full positive cache = base + diverse + google + azure
echo "[5/6] assemble positives ($(date))..." | tee -a "$RES"
python3 -c "
import os, numpy as np
P='$P/'
parts=['hey_ozwell.libritts100k.npy','hey_diverse_pos.npy','hey_accent_pos.npy','hey_azure_accent_pos.npy']
have=[f for f in parts if os.path.exists(P+f)]
allp=np.concatenate([np.load(P+f) for f in have]).astype('float32'); np.save(P+'hey_ozwell.npy', allp)
print('hey_ozwell.npy', allp.shape, '=', [(f, len(np.load(P+f))) for f in have])
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"
POS=$(python3 -c "import numpy as np; print(len(np.load('$P/hey_ozwell.npy')))")

# [6/6] train the proven config (160k negs = negs_C), reusing hey_ozwell.npy + adversarial cache
TAG=heyozwell_C; ONNX=checkpoints/scratch-onnx/${TAG}.onnx
echo "[6/6] train $POS pos / 160k neg ($(date)) -> logs/heyozwell_train.log" | tee -a "$RES"
python -m heybuddy train "hey ozwell" --perceptron \
  --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 3500 --stages 3 --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 3 --augmentation-dataset-streaming \
  --training-dataset $P/negs_C.npy --validation-no-default-dataset --debug \
  > logs/heyozwell_train.log 2>&1
echo "  train exit=$? ($(date))" | tee -a "$RES"

mkdir -p checkpoints/scratch-onnx
python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/hey_ozwell_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('exported')" 2>&1 | grep -E "exported|Error" | tee -a "$RES"

# eval: held-out hey-ozwell accent recall (en-US folder doubles as the American test)
( cd eval && python3 - <<PYEOF 2>&1 | grep -vE "pthread|onnxruntime:Default|Warning|warn|^INFO|DEBUG"
import glob, os
from evaluate_wakeword import WakeWordEvaluator
ev=WakeWordEvaluator("../${ONNX}","pretrained")
def line(n,d):
    p=ev.score_folder(d)
    if len(p): print("  {:18s} @0.5={:5.1f}%  @0.7={:5.1f}%  @0.85={:5.1f}%  (n={})".format(n,(p>=0.5).mean()*100,(p>=0.7).mean()*100,(p>=0.85).mean()*100,len(p)))
print("=== hey ozwell held-out accent recall ===")
for d in sorted(glob.glob("/tmp/eval/hey_accent/*")):
    if len(glob.glob(d+'/*.wav'))>=4: line("accent "+os.path.basename(d), d)
PYEOF
) | tee -a "$RES"

# FP/hour on the same held-out real speech used for ozwell-done
( cd eval && python3 fp_per_hour.py --model "../${ONNX}" --audio-dir /tmp/fphour/peoples_1h \
    --pretrained-dir pretrained --label "$TAG" --thresholds "0.5,0.7,0.85,0.9" 2>&1 | grep -E "over|^  0\." ) | tee -a "$RES"

echo "=== HEY OZWELL DONE $(date) — compare accent recall vs baseline (97% Am -> 11% IN) ===" | tee -a "$RES"
