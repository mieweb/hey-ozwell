#!/usr/bin/env bash
# Deliberate, balanced ozwell-done retrain: config-C base (127k) + BALANCED ElevenLabs accent positives.
# GUARDRAILS (Jonathan: don't ruin existing progress):
#  - base positives PRESERVED (backup + restore), accents only ADDED on top
#  - negatives (config-C 160k) stay > positives -> no trigger-happy regression
#  - NO-REGRESSION eval on the ORIGINAL config-C sets (American + Google/Azure accents) vs prod
#  - plus the NEW n=390 cross-vendor test + multi-source FP. Adopt only if cross-vendor UP and originals + FP HELD.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
P=heybuddy/precalculated
RES=logs/FINAL_RESULTS_SURGICAL.txt
ACC="$P/eleven_surgical_pos_done.npy"
ONNX="checkpoints/scratch-onnx/ozwelldone_surgical.onnx"
PRODDONE="../../prod/js/models/ozwell-i'm-done.onnx"
echo "=== ozwell-done balanced accent retrain $(date) ===" > "$RES"
[ -f "$ACC" ] || { echo "FATAL: $ACC missing (run eleven_embed_balanced.py)" | tee -a "$RES"; exit 1; }

# build combined positives (preserve base)
[ -f "$P/ozwell_i_m_done.prebase.npy" ] || cp "$P/ozwell_i_m_done.npy" "$P/ozwell_i_m_done.prebase.npy"
read POS NACC NEG <<< $(python3 -c "
import numpy as np
b=np.load('$P/ozwell_i_m_done.prebase.npy'); a=np.load('$ACC'); neg=np.load('$P/negs_C.npy')
c=np.concatenate([b,a]).astype('float32'); np.save('$P/ozwell_i_m_done.npy', c)
print(len(c), len(a), len(neg))")
echo ">>> ${POS} pos (base $((POS-NACC)) + ${NACC} balanced accent) / ${NEG} neg  [pos<neg? balanced]" | tee -a "$RES"

python -m heybuddy train "ozwell i'm done" --perceptron \
  --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 3500 --stages 3 --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 3 --augmentation-dataset-streaming \
  --training-dataset "$P/negs_C.npy" --validation-no-default-dataset --debug \
  > logs/surgical_train.log 2>&1
echo "  train exit=$? ($(date))" | tee -a "$RES"
python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/ozwell_i_m_done_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('  exported ${ONNX}')" 2>&1 | grep -E "exported|Error" | tee -a "$RES"
cp "$P/ozwell_i_m_done.prebase.npy" "$P/ozwell_i_m_done.npy"   # restore base

# ---- EVAL: new model vs prod, on all suites ----
( cd eval && python3 - <<PYEOF 2>&1 | grep -vE "pthread|onnxruntime|Warning|warn|^INFO|DEBUG"
import glob, os, numpy as np
from evaluate_wakeword import WakeWordEvaluator
NEW=WakeWordEvaluator("../${ONNX}","pretrained"); PROD=WakeWordEvaluator("${PRODDONE}","pretrained")
def rec(ev,d,thr):
    p=ev.score_folder(d); return ((p>=thr).mean()*100 if len(p) else float('nan')), len(p)
print("### NO-REGRESSION: original config-C sets @0.5 (prod -> new) ###")
orig=[("American","/tmp/eval/done/pos")]+[("acc "+os.path.basename(d),d) for d in sorted(glob.glob("/tmp/eval/accent/*")) if len(glob.glob(d+'/*.wav'))>=4]
for name,d in orig:
    if not glob.glob(d+'/*.wav'): continue
    (rp,_),(rn,n)=rec(PROD,d,0.5),rec(NEW,d,0.5)
    print(f"  {name:18s} n={n:3d}  prod={rp:3.0f}% -> new={rn:3.0f}%")
print("\n### NEW n=390 cross-vendor ElevenLabs @0.5 (prod -> new) ###")
pa=[]; na=[]
for d in sorted(glob.glob("/tmp/eleven_big/test/*/ozwell_done")):
    if not glob.glob(d+'/*.wav'): continue
    acc=d.split('/')[-2]; (rp,_),(rn,n)=rec(PROD,d,0.5),rec(NEW,d,0.5)
    pa.append(PROD.score_folder(d)); na.append(NEW.score_folder(d))
    print(f"  {acc:11s} n={n:3d}  prod={rp:3.0f}% -> new={rn:3.0f}%")
pp,nn=np.concatenate(pa),np.concatenate(na)
print(f"  {'ALL':11s} n={len(pp):3d}  prod={(pp>=0.5).mean()*100:3.0f}% -> new={(nn>=0.5).mean()*100:3.0f}%")
PYEOF
) | tee -a "$RES"
echo "" | tee -a "$RES"; echo ">>> multi-source FP/hr (new model) @0.5:" | tee -a "$RES"
for s in voxpopuli_test peoples_test thirdparty_ami combined; do
  [ -d "/tmp/fphour/$s" ] || continue
  echo "  [$s]:" | tee -a "$RES"
  ( cd eval && python3 fp_per_hour.py --model "../${ONNX}" --audio-dir "/tmp/fphour/$s" \
      --pretrained-dir pretrained --label ozwbig --thresholds "0.5" 2>&1 | grep -E "^  0\." ) | tee -a "$RES"
done
echo "" | tee -a "$RES"; echo "=== SURGICAL DONE $(date) — adopt only if cross-vendor UP, originals HELD, FP HELD ===" | tee -a "$RES"
