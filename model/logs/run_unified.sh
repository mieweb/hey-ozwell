#!/usr/bin/env bash
# OVERNIGHT UNIFIED RETRAIN — runs unattended after track-1 + both accent builds finish.
# Positives = 100k libritts(American) + 15k Piper-diverse(UK/US) + 3x(Google+Azure accents).
# Eval on held-out ElevenLabs(American) + in-domain Piper + every per-accent test set.
# Writes one results file: logs/FINAL_RESULTS_UNIFIED.txt
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
RES=logs/FINAL_RESULTS_UNIFIED.txt; TAG=unified
ONNX=checkpoints/scratch-onnx/ozwell_done_${TAG}.onnx
echo "=== ${TAG} queued $(date) — waiting for track-1 + Google + Azure builds ===" > "$RES"

# ---- 1) wait (max ~9h) for track-1 DONE and Azure DONE (Azure implies Google finished) ----
deadline=$((SECONDS + 32400))
while true; do
  if grep -q "diverse DONE" logs/FINAL_RESULTS_DIVERSE.txt 2>/dev/null \
     && grep -q "DONE: azure_accent_pos" logs/azure_gen.log 2>/dev/null; then break; fi
  if [ "$SECONDS" -ge "$deadline" ]; then echo "  WAIT TIMEOUT $(date) — proceeding with whatever is ready" | tee -a "$RES"; break; fi
  sleep 60
done
echo "[1/4] prerequisites ready $(date)" | tee -a "$RES"

# ---- 2) build unified positive cache from modular parts ----
python3 -c "
import numpy as np, os
P='heybuddy/precalculated/'
parts=[np.load(P+'ozwell_i_m_done.libritts100k.npy')]; desc=['100k libritts']
if os.path.exists(P+'diverse_pos.npy'):
    parts.append(np.load(P+'diverse_pos.npy')); desc.append(str(len(parts[-1]))+' piper-diverse')
acc=[np.load(P+f) for f in ['accent_pos.npy','azure_accent_pos.npy'] if os.path.exists(P+f)]
if acc:
    acc=np.concatenate(acc); parts.append(np.repeat(acc,3,axis=0)); desc.append('3x'+str(len(acc))+' accents(google+azure)')
allp=np.concatenate(parts).astype('float32'); np.save(P+'ozwell_i_m_done.npy', allp)
print('unified positives', allp.shape, '=', ' + '.join(desc))
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"

POS=$(python3 -c "import numpy as np; print(len(np.load('heybuddy/precalculated/ozwell_i_m_done.npy')))" 2>/dev/null)
echo "[2/4] retrain with $POS positives $(date) -> logs/${TAG}_train.log" | tee -a "$RES"
python -m heybuddy train "ozwell i'm done" --perceptron \
  --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 5000 --stages 3 --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 3 --augmentation-dataset-streaming \
  --training-dataset heybuddy/precalculated/negs_all.npy \
  --validation-no-default-dataset --debug > logs/${TAG}_train.log 2>&1
echo "  train exit=$? $(date)" | tee -a "$RES"

echo "[3/4] export ONNX" | tee -a "$RES"
mkdir -p checkpoints/scratch-onnx
python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/ozwell_i_m_done_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('exported')" 2>&1 | grep -E "exported|Error" | tee -a "$RES"

echo "[4/4] eval — recall by test set $(date):" | tee -a "$RES"
cd eval
python3 - <<PYEOF 2>&1 | grep -vE "pthread|onnxruntime:Default|Warning|warn|INFO|DEBUG" | tee -a "../$RES"
import glob, os, numpy as np
from evaluate_wakeword import WakeWordEvaluator
ev=WakeWordEvaluator("../${ONNX}","pretrained")
neg=ev.score_folder("/tmp/eval/done/neg")
print("--- UNIFIED model: recall by test set | per-clip neg FPR@0.5 = {:.0f}% ---".format((neg>=0.5).mean()*100))
def line(name,posd):
    p=ev.score_folder(posd)
    if not len(p): print(f"  {name:26s} (no clips)"); return
    print(f"  {name:26s} recall@0.5={ (p>=0.5).mean()*100:5.1f}%  @0.7={ (p>=0.7).mean()*100:5.1f}%  (n={len(p)})")
line("ElevenLabs (American)","/tmp/eval/done/pos")
line("Piper (in-domain)","/tmp/eval/piper_pos")
for d in sorted(glob.glob("/tmp/eval/accent/*")):
    line("accent "+os.path.basename(d), d)
PYEOF
cd ..
echo "=== ${TAG} DONE $(date) — read this file for results ===" | tee -a "$RES"
