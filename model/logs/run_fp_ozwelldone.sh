#!/usr/bin/env bash
# ozwell-i'm-done FP-hardening: stop it firing on "ozwell + wrong ending" (e.g. "ozwell deez nuts").
# Negatives = negs_C + confusables + ozwell_wrong_negs ("ozwell + wrong ending"). NO counterpart
# ("hey ozwell" tanked this model's recall 92->63). All as LABELED [n,17,96] via borrowed token rows.
# ADOPT ONLY IF recall holds (American + accents) — these share "ozwell", so over-suppression is the risk.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
P=heybuddy/precalculated
RES=logs/FINAL_RESULTS_OZWDONE_FP.txt
echo "=== ozwell-done FP hardening $(date) ===" > "$RES"
if [ ! -f "$P/ozwell_wrong_negs.npy" ]; then echo "FATAL: ozwell_wrong_negs.npy missing — run gen_ozwell_wrong.py first" | tee -a "$RES"; exit 1; fi

# baseline: does the CURRENT ozwell-done fire on "ozwell + wrong ending"? (the bug)
echo ">>> BASELINE: current ozwell-done on 'ozwell + wrong ending' clips:" | tee -a "$RES"
( cd eval && python3 -c "
import glob, numpy as np
from evaluate_wakeword import WakeWordEvaluator
ev=WakeWordEvaluator('../checkpoints/scratch-onnx/ozwell_done_negsweep_C_extract.onnx','pretrained')
p=ev.score_folder('/tmp/eval/ozwell_wrong')
print(f'  fires on ozwell-wrong @0.5: {(p>=0.5).mean()*100:.0f}% (n={len(p)}) [the bug — want LOW after fix]')
" 2>&1 | grep -vE "pthread|onnxruntime|Warning|warn|^INFO|DEBUG" ) | tee -a "$RES"

# build negatives (NO counterpart)
python3 -c "
import numpy as np
P='$P/'; rng=np.random.default_rng(0)
negs=np.load(P+'negs_C.npy'); conf=np.load(P+'confusable_negs.npy'); ozw=np.load(P+'ozwell_wrong_negs.npy')
def labeled(a16,n):
    a=a16[rng.choice(len(a16),size=min(n,len(a16)),replace=False)]
    tok=negs[rng.choice(len(negs),size=len(a),replace=True),-1:,:]
    return np.concatenate([a,tok],axis=1)
nd=np.concatenate([negs, labeled(conf,len(conf)), labeled(ozw,len(ozw))]).astype('float32')
np.save(P+'negs_fp_done.npy', nd)
print('negs_fp_done', nd.shape, '= negs_C +',len(conf),'confusable +',len(ozw),'ozwell-wrong')
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"

POS=$(python3 -c "import numpy as np;print(len(np.load('$P/ozwell_i_m_done.npy')))")
TAG=ozwell_done_fpx; ONNX="checkpoints/scratch-onnx/${TAG}.onnx"
echo ">>> TRAIN ${TAG}: ${POS} pos / negs_fp_done ($(date))" | tee -a "$RES"
python -m heybuddy train "ozwell i'm done" --perceptron \
  --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
  --steps 3500 --stages 3 --target-false-positive-rate 1.5 \
  --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
  --num-batch-threads 3 --augmentation-dataset-streaming \
  --training-dataset "$P/negs_fp_done.npy" --validation-no-default-dataset --debug \
  > logs/xtalk_${TAG}.log 2>&1
echo "  train exit=$? ($(date))" | tee -a "$RES"
python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/ozwell_i_m_done_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('  exported ${ONNX}')" 2>&1 | grep -E "exported|Error" | tee -a "$RES"

echo "" | tee -a "$RES"; echo ">>> AFTER FIX ${TAG} @0.5 (orig = American 92/IN 83/GB 94/AU 100/US 96, FP 0.6/hr):" | tee -a "$RES"
( cd eval && python3 - <<PYEOF 2>&1 | grep -vE "pthread|onnxruntime|Warning|warn|^INFO|DEBUG"
import glob, os, numpy as np
from evaluate_wakeword import WakeWordEvaluator
ev=WakeWordEvaluator("../${ONNX}","pretrained")
def rec(d): p=ev.score_folder(d); return ((p>=0.5).mean()*100, len(p)) if len(p) else (None,0)
print("  own recall (must hold):")
r,n=rec("/tmp/eval/done/pos"); print(f"    American: {r:.0f}% (n={n})")
for d in sorted(glob.glob("/tmp/eval/accent/*")):
    if len(glob.glob(d+'/*.wav'))>=4: r,n=rec(d); print(f"    {os.path.basename(d)}: {r:.0f}% (n={n})")
r,n=rec("/tmp/eval/ozwell_wrong"); print(f"  REJECTION: fires on 'ozwell-wrong' @0.5: {r:.0f}% (n={n}) [want LOW]")
PYEOF
) | tee -a "$RES"
( cd eval && python3 fp_per_hour.py --model "../${ONNX}" --audio-dir /tmp/fphour/peoples_1h \
    --pretrained-dir pretrained --label "$TAG" --thresholds "0.5" 2>&1 | grep -E "over|^  0\." ) | tee -a "$RES"
echo "" | tee -a "$RES"; echo "=== OZWDONE_FP DONE $(date) — adopt ONLY if recall held AND ozwell-wrong rejection dropped ===" | tee -a "$RES"
