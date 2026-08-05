#!/usr/bin/env bash
# NEGATIVE-RATIO SWEEP (FP reduction). Recall (incl. accents) is solved; FP regressed because positives
# (127k) outnumber negatives (52.5k) AND negatives are read-speech while the model fires on conversational.
# Keep x1 positives FIXED (best accents); vary negatives toward MORE + CONVERSATIONAL:
#   B ~105k = negs_all(libri+free) + negs_peoples_all(People's Speech conversational)
#   C ~155k = B + a fresh People's Speech extract
# Each -> recall (American + per-accent) AND FP/hour. Pick the ratio that lands BOTH (recall>=90%, FP<1/hr).
# Baseline for comparison = x1 (negs 52.5k): FP ~18/hr @0.5, accent recall ~90-100%.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
RES=/home/jlocala/hey-ozwell/model/logs/FINAL_RESULTS_NEGSWEEP.txt
echo "=== negative-ratio sweep $(date) ===" > "$RES"

# fixed x1 positives
python3 -c "
import numpy as np
P='heybuddy/precalculated/'
pos=np.concatenate([np.load(P+f) for f in ['ozwell_i_m_done.libritts100k.npy','diverse_pos.npy','accent_pos.npy','azure_accent_pos.npy']]).astype('float32')
np.save(P+'ozwell_i_m_done.npy', pos); print('x1 positives', pos.shape)
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"
POS=$(python3 -c "import numpy as np; print(len(np.load('heybuddy/precalculated/ozwell_i_m_done.npy')))")

# negs_B (conversational-inclusive ~105k)
python3 -c "
import numpy as np
P='heybuddy/precalculated/'
b=np.concatenate([np.load(P+'negs_all.npy'), np.load(P+'negs_peoples_all.npy')]).astype('float32')
np.save(P+'negs_B.npy', b); print('negs_B', b.shape)
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"

run_cfg () {
  local NEG=$1 LABEL=$2 TAG="negsweep_$2"
  local ONNX="checkpoints/scratch-onnx/ozwell_done_${TAG}.onnx"
  local NC; NC=$(python3 -c "import numpy as np; print(len(np.load('heybuddy/precalculated/${NEG}')))")
  echo "" | tee -a "$RES"; echo ">>> CONFIG ${LABEL}: ${POS} pos / ${NC} neg  ($(date))" | tee -a "$RES"
  python -m heybuddy train "ozwell i'm done" --perceptron \
    --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
    --steps 3500 --stages 3 --target-false-positive-rate 1.5 \
    --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
    --num-batch-threads 3 --augmentation-dataset-streaming \
    --training-dataset "heybuddy/precalculated/${NEG}" \
    --validation-no-default-dataset --debug > "logs/${TAG}_train.log" 2>&1
  echo "  train exit=$? $(date)" | tee -a "$RES"
  python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/ozwell_i_m_done_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('  exported')" 2>&1 | grep -E "exported|Error" | tee -a "$RES"
  ( cd eval && python3 - <<PYEOF 2>&1 | grep -vE "pthread|onnxruntime:Default|Warning|warn|INFO|DEBUG"
import glob, os, numpy as np
from evaluate_wakeword import WakeWordEvaluator
ev=WakeWordEvaluator("../${ONNX}","pretrained")
neg=ev.score_folder("/tmp/eval/done/neg")
print("  [${LABEL}] recall | per-clip neg FPR@0.5={:.0f}%".format((neg>=0.5).mean()*100))
def line(n,d):
    p=ev.score_folder(d)
    if len(p): print("    {:22s} @0.5={:5.1f}%  @0.7={:5.1f}%  (n={})".format(n,(p>=0.5).mean()*100,(p>=0.7).mean()*100,len(p)))
line("American","/tmp/eval/done/pos")
for d in sorted(glob.glob("/tmp/eval/accent/*")):
    if len(glob.glob(d+'/*.wav'))>=4: line("accent "+os.path.basename(d), d)
PYEOF
  ) | tee -a "$RES"
  ( cd eval && python3 fp_per_hour.py --model "../${ONNX}" --audio-dir /tmp/fphour/peoples_1h \
      --pretrained-dir pretrained --label "${LABEL}" --thresholds "0.5,0.7,0.9" 2>&1 | grep -E "over|^  0\." ) | tee -a "$RES"
}

# B first (no extract needed) so training starts immediately
run_cfg "negs_B.npy" "B_105k" || echo "B failed" | tee -a "$RES"

# extract more People's Speech, build negs_C (~155k), run C
echo "" | tee -a "$RES"; echo "[extract] more People's Speech for config C ($(date))..." | tee -a "$RES"
rm -rf heybuddy/precalculated/negs_peoples2
python -m heybuddy extract negs_peoples2 MLCommons/peoples_speech --config clean --split train \
  --audio-key audio --transcript-key text --hours 22 --streaming --trust-remote-code --device-id 0 --debug \
  > logs/negsweep_extract.log 2>&1 || true
python3 -c "
import numpy as np, glob
P='heybuddy/precalculated/'
extra=[np.load(f) for f in sorted(glob.glob(P+'negs_peoples2/*.npy'))]
c=np.concatenate([np.load(P+'negs_B.npy')]+extra).astype('float32') if extra else np.load(P+'negs_B.npy')
np.save(P+'negs_C.npy', c); print('negs_C', c.shape, 'from B +', len(extra), 'extract files')
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"
run_cfg "negs_C.npy" "C_extract" || echo "C failed" | tee -a "$RES"

echo "" | tee -a "$RES"
echo "=== NEGSWEEP DONE $(date) — compare B(105k) vs C(155k) vs baseline x1(52.5k neg, FP~18/hr) ===" | tee -a "$RES"
