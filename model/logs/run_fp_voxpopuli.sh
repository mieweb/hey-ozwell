#!/usr/bin/env bash
# hey-ozwell retrain: winning fpx recipe + VoxPopuli diverse real negatives.
# Goal: close the LIVE conversational-FP gap (synthetic 0.6/hr underestimates real speech).
# VoxPopuli = accented European-parliament conversational speech, a DIFFERENT distribution than
# People's Speech (negs_C) — generic, NOT "ozwell"-sharing, so it's SAFE for recall (unlike
# counterpart/ozwell-adjacent negs). Negatives = negs_C + 40k counterpart + 8k confusables + ALL VoxPopuli.
# Re-verifies: hey recall per accent (must HOLD vs fpx 100/100/100/100) + cross-trigger (~2%) + FP/hr (<1, target < fpx 0.6).
# ozwell-done LEFT on its ORIGINAL model.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
P=heybuddy/precalculated
RES=logs/FINAL_RESULTS_FPVOX.txt
COUNTERPART=40000
echo "=== hey-ozwell retrain: fpx + VoxPopuli negs $(date) ===" > "$RES"

if [ ! -f "$P/confusable_negs.npy" ]; then echo "FATAL: $P/confusable_negs.npy missing" | tee -a "$RES"; exit 1; fi
NVOX=$(ls "$P"/negs_voxpopuli/*.npy 2>/dev/null | wc -l)
if [ "$NVOX" -eq 0 ]; then echo "FATAL: no VoxPopuli shards in $P/negs_voxpopuli/" | tee -a "$RES"; exit 1; fi
echo "VoxPopuli shards found: $NVOX" | tee -a "$RES"

python3 -c "
import numpy as np, glob
P='$P/'; rng=np.random.default_rng(0)
negs=np.load(P+'negs_C.npy')                 # [N,17,96] labeled
done=np.load(P+'ozwell_i_m_done.npy')        # counterpart for hey-ozwell [n,16,96]
conf=np.load(P+'confusable_negs.npy')        # FP hard negs [n,16,96]
vox=np.concatenate([np.load(f) for f in sorted(glob.glob(P+'negs_voxpopuli/*.npy'))])  # [n,17,96] labeled
def labeled(a16, n):
    a=a16[rng.choice(len(a16), size=min(n,len(a16)), replace=False)]
    tok=negs[rng.choice(len(negs), size=len(a), replace=True), -1:, :]
    return np.concatenate([a, tok], axis=1)
nh=np.concatenate([negs, labeled(done,$COUNTERPART), labeled(conf,len(conf)), vox]).astype('float32')
np.save(P+'negs_fp_vox.npy', nh)
print('negs_fp_vox', nh.shape, '= negs_C(%d) + %d counterpart + %d confusable + %d voxpopuli'%(len(negs),$COUNTERPART,len(conf),len(vox)))
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"

POS_HEY=$(python3 -c "import numpy as np;print(len(np.load('$P/hey_ozwell.npy')))")
TAG=heyozwell_fpvox; ONNX="checkpoints/scratch-onnx/${TAG}.onnx"
echo ">>> TRAIN ${TAG}: ${POS_HEY} pos / negs_fp_vox ($(date))" | tee -a "$RES"
python -m heybuddy train "hey ozwell" --perceptron \
  --positive-samples "$POS_HEY" --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 3500 --stages 3 --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 3 --augmentation-dataset-streaming \
  --training-dataset "$P/negs_fp_vox.npy" --validation-no-default-dataset --debug \
  > logs/fpvox_${TAG}.log 2>&1
echo "  train exit=$? ($(date))" | tee -a "$RES"
python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/hey_ozwell_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('  exported ${ONNX}')" 2>&1 | grep -E "exported|Error" | tee -a "$RES"

echo "" | tee -a "$RES"; echo ">>> EVAL ${TAG} @0.8 (compare fpx: recall 100/100/100/100, cross-trigger 2%, FP 0.6/hr):" | tee -a "$RES"
( cd eval && python3 - <<PYEOF 2>&1 | grep -vE "pthread|onnxruntime|Warning|warn|^INFO|DEBUG"
import glob, os, numpy as np
from evaluate_wakeword import WakeWordEvaluator
ev=WakeWordEvaluator("../${ONNX}","pretrained")
def rec(d,thr):
    p=ev.score_folder(d); return (p>=thr).mean()*100 if len(p) else None, len(p)
print("  own recall (hey accents):")
for d in sorted(glob.glob("/tmp/eval/hey_accent/*")):
    if len(glob.glob(d+'/*.wav'))>=4: r,n=rec(d,0.8); print(f"    {os.path.basename(d)}: {r:.0f}% (n={n})")
a=[ev.score_folder(d) for d in sorted(glob.glob("/tmp/eval/accent/*")) if glob.glob(d+'/*.wav')]
p=np.concatenate(a)
print(f"  CROSS-TRIGGER on 'ozwell done' @0.8: {(p>=0.8).mean()*100:.0f}% (n={len(p)}) [want ~0]")
PYEOF
) | tee -a "$RES"
( cd eval && python3 fp_per_hour.py --model "../${ONNX}" --audio-dir /tmp/fphour/peoples_1h \
    --pretrained-dir pretrained --label "$TAG" --thresholds "0.8" 2>&1 | grep -E "over|^  0\." ) | tee -a "$RES"
echo "" | tee -a "$RES"; echo "=== FPVOX DONE $(date) — adopt if FP < 0.6/hr AND recall held AND cross-trigger low ===" | tee -a "$RES"
