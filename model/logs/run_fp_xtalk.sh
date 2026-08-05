#!/usr/bin/env bash
# COMBINED hey-ozwell retrain: fix cross-talk AND false alarms together.
# Negatives = negs_C + counterpart (ozwell-done positives, cross-talk) + confusable_negs (FP: "yo bro" etc).
# All in LABELED [n,17,96] form via a BORROWED token row (so the exclude filter keeps them).
# ozwell-done is LEFT ON ITS ORIGINAL model (counterpart negs hurt its recall — see xtalk results).
# Re-verifies: hey recall per accent (must hold) + cross-trigger (must stay ~0) + FP/hr (target <1).
# Prereq: run gen_confusables.py first to make confusable_negs.npy.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
P=heybuddy/precalculated
RES=logs/FINAL_RESULTS_FPXTALK.txt
COUNTERPART=40000
echo "=== combined hey-ozwell retrain (cross-talk + FP) $(date) ===" > "$RES"

if [ ! -f "$P/confusable_negs.npy" ]; then echo "FATAL: $P/confusable_negs.npy missing — run gen_confusables.py first" | tee -a "$RES"; exit 1; fi

python3 -c "
import numpy as np
P='$P/'; rng=np.random.default_rng(0)
negs=np.load(P+'negs_C.npy')                 # [N,17,96] labeled
done=np.load(P+'ozwell_i_m_done.npy')        # counterpart for hey-ozwell
conf=np.load(P+'confusable_negs.npy')        # FP hard negs
def labeled(a16, n):
    a=a16[rng.choice(len(a16), size=min(n,len(a16)), replace=False)]
    tok=negs[rng.choice(len(negs), size=len(a), replace=True), -1:, :]
    return np.concatenate([a, tok], axis=1)
nh=np.concatenate([negs, labeled(done,$COUNTERPART), labeled(conf,len(conf))]).astype('float32')
np.save(P+'negs_fp_hey.npy', nh)
print('negs_fp_hey', nh.shape, '= negs_C + ',$COUNTERPART,'counterpart +',len(conf),'confusable')
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"

POS_HEY=$(python3 -c "import numpy as np;print(len(np.load('$P/hey_ozwell.npy')))")
TAG=heyozwell_fpx; ONNX="checkpoints/scratch-onnx/${TAG}.onnx"
echo ">>> TRAIN ${TAG}: ${POS_HEY} pos / negs_fp_hey ($(date))" | tee -a "$RES"
python -m heybuddy train "hey ozwell" --perceptron \
  --positive-samples "$POS_HEY" --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 3500 --stages 3 --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 3 --augmentation-dataset-streaming \
  --training-dataset "$P/negs_fp_hey.npy" --validation-no-default-dataset --debug \
  > logs/xtalk_${TAG}.log 2>&1
echo "  train exit=$? ($(date))" | tee -a "$RES"
python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/hey_ozwell_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('  exported ${ONNX}')" 2>&1 | grep -E "exported|Error" | tee -a "$RES"

echo "" | tee -a "$RES"; echo ">>> EVAL ${TAG} @0.8 (compare: original = recall 100/100/100/96, xtalk = +0 cross-trigger but FP 2.5/hr):" | tee -a "$RES"
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
echo "" | tee -a "$RES"; echo "=== FPXTALK DONE $(date) — adopt if cross-trigger low AND FP<1 AND recall held ===" | tee -a "$RES"
