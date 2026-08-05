#!/usr/bin/env bash
# Balanced accent retrain for BOTH phrases using ElevenLabs cross-vendor accent positives.
# BALANCE (per Jonathan's directive — avoid the trigger-happy unbalanced run): negatives stay >= positives.
#   hey-ozwell : ~126k base + ~6k accent = ~132k pos / negs_fp_hey 208k  (fpx recipe, cross-talk-safe)
#   ozwell-done: ~127k base + ~6k accent = ~133k pos / negs_C 160k       (config-C ORIGINAL; NO counterpart)
# Accent positives = ElevenLabs (cross-vendor, disjoint train/test voices), embedded+augmented x13.
# Eval = held-out ElevenLabs accent recall (honest, disjoint voices) + MULTI-SOURCE FP (vox/peoples/AMI/combined).
# Base positive caches are backed up (*.prebase.npy) and RESTORED after each train.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
P=heybuddy/precalculated
RES=logs/FINAL_RESULTS_ELAC.txt
echo "=== ElevenLabs accent retrain (both phrases, balanced) $(date) ===" > "$RES"
for f in eleven_accent_pos_hey.npy eleven_accent_pos_done.npy; do
  [ -f "$P/$f" ] || { echo "FATAL: $P/$f missing (run eleven_embed_accents.py)" | tee -a "$RES"; exit 1; }
done

train_phrase () {
  local PHRASE="$1" SLUG="$2" BASE="$3" ACC="$4" NEGS="$5" CKPT="$6" TAG="$7" TESTSUB="$8"
  local ONNX="checkpoints/scratch-onnx/${TAG}.onnx"
  echo "" | tee -a "$RES"; echo "########## ${TAG} (${PHRASE}) ##########" | tee -a "$RES"
  # backup base once, build combined positives in-place
  [ -f "$P/${BASE}.prebase.npy" ] || cp "$P/${BASE}.npy" "$P/${BASE}.prebase.npy"
  local POS
  POS=$(python3 -c "
import numpy as np
b=np.load('$P/${BASE}.prebase.npy'); a=np.load('$P/${ACC}')
c=np.concatenate([b,a]).astype('float32'); np.save('$P/${BASE}.npy', c)
print(len(c))")
  local NC; NC=$(python3 -c "import numpy as np;print(len(np.load('$P/${NEGS}')))")
  echo ">>> ${POS} pos (incl $(python3 -c "import numpy as np;print(len(np.load('$P/${ACC}')))") accent) / ${NC} neg ($(date))" | tee -a "$RES"
  python -m heybuddy train "$PHRASE" --perceptron \
    --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
    --steps 3500 --stages 3 --target-false-positive-rate 1.5 \
    --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
    --num-batch-threads 3 --augmentation-dataset-streaming \
    --training-dataset "$P/${NEGS}" --validation-no-default-dataset --debug \
    > "logs/${TAG}_train.log" 2>&1
  echo "  train exit=$? ($(date))" | tee -a "$RES"
  python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/${CKPT}'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('  exported ${ONNX}')" 2>&1 | grep -E "exported|Error" | tee -a "$RES"
  # restore pristine base cache
  cp "$P/${BASE}.prebase.npy" "$P/${BASE}.npy"
  # --- eval: held-out ElevenLabs accent recall (disjoint voices) ---
  echo "  >>> held-out ElevenLabs accent recall:" | tee -a "$RES"
  ( cd eval && python3 - <<PYEOF 2>&1 | grep -vE "pthread|onnxruntime|Warning|warn|^INFO|DEBUG"
import glob, os, numpy as np
from evaluate_wakeword import WakeWordEvaluator
ev=WakeWordEvaluator("../${ONNX}","pretrained")
allp=[]
for d in sorted(glob.glob("/tmp/eleven_pos/test/*/${TESTSUB}")):
    w=glob.glob(d+"/*.wav")
    if not w: continue
    p=ev.score_folder(d); allp.append(p); acc=d.split("/")[-2]
    print(f"    {acc:11s} n={len(p):2d}  @0.5={(p>=0.5).mean()*100:3.0f}%  @0.8={(p>=0.8).mean()*100:3.0f}%  @0.9={(p>=0.9).mean()*100:3.0f}%")
if allp:
    p=np.concatenate(allp)
    print(f"    {'ALL':11s} n={len(p):2d}  @0.5={(p>=0.5).mean()*100:3.0f}%  @0.8={(p>=0.8).mean()*100:3.0f}%  @0.9={(p>=0.9).mean()*100:3.0f}%")
PYEOF
  ) | tee -a "$RES"
  # --- eval: MULTI-SOURCE FP ---
  echo "  >>> multi-source FP/hr:" | tee -a "$RES"
  for s in voxpopuli_test peoples_test thirdparty_ami combined; do
    [ -d "/tmp/fphour/$s" ] || continue
    echo "    [$s]:" | tee -a "$RES"
    ( cd eval && python3 fp_per_hour.py --model "../${ONNX}" --audio-dir "/tmp/fphour/$s" \
        --pretrained-dir pretrained --label "$TAG" --thresholds "0.8,0.9" 2>&1 | grep -E "^  0\." ) | tee -a "$RES"
  done
}

train_phrase "hey ozwell"      hey_ozwell      hey_ozwell      eleven_accent_pos_hey.npy  negs_fp_hey.npy  hey_ozwell_final.pt      heyozwell_elac   hey_ozwell
train_phrase "ozwell i'm done" ozwell_i_m_done ozwell_i_m_done eleven_accent_pos_done.npy negs_C.npy       ozwell_i_m_done_final.pt ozwelldone_elac  ozwell_done

echo "" | tee -a "$RES"; echo "=== ELAC DONE $(date) — compare recall to cross-vendor baseline (ozwell-done was JP50/ZH67/ES75/ALL83) AND check multi-source FP didn't regress ===" | tee -a "$RES"
