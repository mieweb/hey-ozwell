#!/usr/bin/env bash
# CROSS-TALK FIX: "hey ozwell" and "ozwell i'm done" share the word "ozwell" and co-fire on one
# utterance. Fix = add each phrase's POSITIVES as hard NEGATIVES for the OTHER model so they learn
# to distinguish. Reuses cached embeddings (NO generation/augmentation) -> fast (~train only).
# Verifies: own-phrase recall NOT hurt + cross-trigger rate dropped + FP/hr still <1.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
P=heybuddy/precalculated
RES=logs/FINAL_RESULTS_XTALK.txt
COUNT=40000   # counterpart hard-negatives to add (tunable, empirical — like the negsweep)
echo "=== cross-talk fix $(date) (counterpart negs=$COUNT) ===" > "$RES"

# [1] build counterpart-augmented negative sets (negs_C + COUNT of the OTHER phrase's positives).
# negs_C is LABELED [N,17,96] (last row=tokens, used by an exclude filter); positives are [M,16,96].
# Give each counterpart clip a token row BORROWED from a real negs_C negative (known to pass the
# exclude filter) so it survives as a negative — the model only sees the 16 audio frames anyway.
python3 -c "
import numpy as np
P='$P/'; C=$COUNT; rng=np.random.default_rng(0)
negs=np.load(P+'negs_C.npy'); hey=np.load(P+'hey_ozwell.npy'); done=np.load(P+'ozwell_i_m_done.npy')
def labeled_neg(audio16, n):
    a=audio16[rng.choice(len(audio16), size=min(n,len(audio16)), replace=False)]   # [c,16,96]
    tok=negs[rng.choice(len(negs), size=len(a), replace=True), -1:, :]             # [c,1,96] borrowed token rows
    return np.concatenate([a, tok], axis=1)                                        # [c,17,96]
nd=np.concatenate([negs, labeled_neg(hey,C)]).astype('float32'); np.save(P+'negs_xtalk_done.npy', nd)  # done trains vs hey
nh=np.concatenate([negs, labeled_neg(done,C)]).astype('float32'); np.save(P+'negs_xtalk_hey.npy', nh)  # hey trains vs done
print('negs_xtalk_done', nd.shape, '| negs_xtalk_hey', nh.shape)
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"

# [2] baseline cross-trigger (current config-C models, BEFORE the fix)
echo "" | tee -a "$RES"; echo ">>> BASELINE cross-trigger (current models):" | tee -a "$RES"
( cd eval && python3 - <<'PYEOF' 2>&1 | grep -vE "pthread|onnxruntime|Warning|warn|^INFO|DEBUG"
import glob, os, numpy as np
from evaluate_wakeword import WakeWordEvaluator
def pooled(ev,parent):
    a=[ev.score_folder(d) for d in sorted(glob.glob(parent+"/*")) if glob.glob(d+"/*.wav")]
    return np.concatenate(a) if a else np.array([])
ev=WakeWordEvaluator("../checkpoints/scratch-onnx/ozwell_done_negsweep_C_extract.onnx","pretrained")
p=pooled(ev,"/tmp/eval/hey_accent"); print(f"  ozwell-done fires on 'hey ozwell' clips @0.5: {(p>=0.5).mean()*100:.0f}% (n={len(p)}) [want LOW]")
ev=WakeWordEvaluator("../checkpoints/scratch-onnx/heyozwell_C.onnx","pretrained")
p=pooled(ev,"/tmp/eval/accent"); print(f"  hey-ozwell fires on 'ozwell done' clips @0.8: {(p>=0.8).mean()*100:.0f}% (n={len(p)}) [want LOW]")
PYEOF
) | tee -a "$RES"

POS_DONE=$(python3 -c "import numpy as np;print(len(np.load('$P/ozwell_i_m_done.npy')))")
POS_HEY=$(python3 -c "import numpy as np;print(len(np.load('$P/hey_ozwell.npy')))")

train_one () {
  local PHRASE="$1" POS="$2" NEG="$3" TAG="$4" CKPT="$5"
  local ONNX="checkpoints/scratch-onnx/${TAG}.onnx"
  echo "" | tee -a "$RES"; echo ">>> TRAIN ${TAG}: ${POS} pos / ${NEG} ($(date))" | tee -a "$RES"
  python -m heybuddy train "$PHRASE" --perceptron \
    --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
    --steps 3500 --stages 3 --target-false-positive-rate 1.5 \
    --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
    --num-batch-threads 3 --augmentation-dataset-streaming \
    --training-dataset "$P/${NEG}" --validation-no-default-dataset --debug \
    > "logs/xtalk_${TAG}.log" 2>&1
  echo "  train exit=$? ($(date))" | tee -a "$RES"
  # CKPT is the safe-name checkpoint base (passed in to avoid the apostrophe in "ozwell i'm done")
  python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/${CKPT}_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('  exported ${ONNX}')" 2>&1 | grep -E "exported|Error" | tee -a "$RES"
}

train_one "ozwell i'm done" "$POS_DONE" "negs_xtalk_done.npy" "ozwell_done_xtalk" "ozwell_i_m_done"
train_one "hey ozwell"      "$POS_HEY"  "negs_xtalk_hey.npy"  "heyozwell_xtalk"   "hey_ozwell"

# [3] AFTER FIX — own recall (must hold) + cross-trigger (must drop)
echo "" | tee -a "$RES"; echo ">>> AFTER FIX:" | tee -a "$RES"
( cd eval && python3 - <<'PYEOF' 2>&1 | grep -vE "pthread|onnxruntime|Warning|warn|^INFO|DEBUG"
import glob, os, numpy as np
from evaluate_wakeword import WakeWordEvaluator
def rec(ev,d,thr):
    p=ev.score_folder(d); return (p>=thr).mean()*100 if len(p) else None, len(p)
def pooled(ev,parent):
    a=[ev.score_folder(d) for d in sorted(glob.glob(parent+"/*")) if glob.glob(d+"/*.wav")]
    return np.concatenate(a) if a else np.array([])
print("ozwell-done (xtalk) @0.5 — own recall:")
ev=WakeWordEvaluator("../checkpoints/scratch-onnx/ozwell_done_xtalk.onnx","pretrained")
r,n=rec(ev,"/tmp/eval/done/pos",0.5); print(f"   American: {r:.0f}% (n={n})")
for d in sorted(glob.glob("/tmp/eval/accent/*")):
    if len(glob.glob(d+'/*.wav'))>=4: r,n=rec(ev,d,0.5); print(f"   {os.path.basename(d)}: {r:.0f}% (n={n})")
p=pooled(ev,"/tmp/eval/hey_accent"); print(f"   CROSS-TRIGGER on 'hey ozwell' @0.5: {(p>=0.5).mean()*100:.0f}% (n={len(p)}) [want LOW]")
print("hey-ozwell (xtalk) @0.8 — own recall:")
ev=WakeWordEvaluator("../checkpoints/scratch-onnx/heyozwell_xtalk.onnx","pretrained")
for d in sorted(glob.glob("/tmp/eval/hey_accent/*")):
    if len(glob.glob(d+'/*.wav'))>=4: r,n=rec(ev,d,0.8); print(f"   {os.path.basename(d)}: {r:.0f}% (n={n})")
p=pooled(ev,"/tmp/eval/accent"); print(f"   CROSS-TRIGGER on 'ozwell done' @0.8: {(p>=0.8).mean()*100:.0f}% (n={len(p)}) [want LOW]")
PYEOF
) | tee -a "$RES"

# [4] FP/hour (confirm no regression)
for M in "ozwell_done_xtalk 0.5" "heyozwell_xtalk 0.8"; do set -- $M
  echo "" | tee -a "$RES"; echo ">>> FP/hour ${1} (op thr ${2}):" | tee -a "$RES"
  ( cd eval && python3 fp_per_hour.py --model "../checkpoints/scratch-onnx/${1}.onnx" \
      --audio-dir /tmp/fphour/peoples_1h --pretrained-dir pretrained --label "${1}" \
      --thresholds "${2}" 2>&1 | grep -E "over|^  0\." ) | tee -a "$RES"
done

echo "" | tee -a "$RES"; echo "=== XTALK DONE $(date) ===" | tee -a "$RES"
