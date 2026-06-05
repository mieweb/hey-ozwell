#!/usr/bin/env bash
# OVERNIGHT ACCENT-WEIGHT SWEEP — after track-1 + both accent builds finish, retrains 3 configs
# varying ONLY the accent oversample factor (x1 ~10% / x3 ~25% / x6 ~40% of positives), each
# evaluated on held-out American + every per-accent test set. Configs are isolated: one failing
# doesn't stop the others. Results -> logs/FINAL_RESULTS_SWEEP.txt
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
RES=/home/jlocala/hey-ozwell/model/logs/FINAL_RESULTS_SWEEP.txt
echo "=== accent-weight SWEEP queued $(date) — waiting for track-1 + Google + Azure builds ===" > "$RES"

deadline=$((SECONDS + 32400))
while true; do
  if grep -q "diverse DONE" logs/FINAL_RESULTS_DIVERSE.txt 2>/dev/null \
     && grep -q "DONE: azure_accent_pos" logs/azure_gen.log 2>/dev/null; then break; fi
  [ "$SECONDS" -ge "$deadline" ] && { echo "WAIT TIMEOUT $(date) — proceeding with what's ready" | tee -a "$RES"; break; }
  sleep 60
done
echo "prerequisites ready $(date)" | tee -a "$RES"

run_config () {
  local MULT=$1
  local TAG="sweep_x${MULT}"
  local ONNX="checkpoints/scratch-onnx/ozwell_done_${TAG}.onnx"
  echo "" | tee -a "$RES"
  echo ">>> CONFIG accent x${MULT}  (build positives $(date))" | tee -a "$RES"
  python3 -c "
import numpy as np, os
P='heybuddy/precalculated/'
parts=[np.load(P+'ozwell_i_m_done.libritts100k.npy')]
if os.path.exists(P+'diverse_pos.npy'): parts.append(np.load(P+'diverse_pos.npy'))
acc=[np.load(P+f) for f in ['accent_pos.npy','azure_accent_pos.npy'] if os.path.exists(P+f)]
if acc:
    acc=np.concatenate(acc); parts.append(np.repeat(acc, ${MULT}, axis=0))
allp=np.concatenate(parts).astype('float32'); np.save(P+'ozwell_i_m_done.npy', allp)
print('  positives', allp.shape, '(accent x${MULT})')
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"
  local POS
  POS=$(python3 -c "import numpy as np; print(len(np.load('heybuddy/precalculated/ozwell_i_m_done.npy')))" 2>/dev/null)
  echo "  retrain x${MULT} ($POS positives) $(date) -> logs/${TAG}_train.log" | tee -a "$RES"
  python -m heybuddy train "ozwell i'm done" --perceptron \
    --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
    --steps 5000 --stages 3 --target-false-positive-rate 1.5 \
    --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
    --num-batch-threads 3 --augmentation-dataset-streaming \
    --training-dataset heybuddy/precalculated/negs_all.npy \
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
print("  [accent x${MULT}] recall by set | neg FPR@0.5={:.0f}%".format((neg>=0.5).mean()*100))
def line(n,d):
    p=ev.score_folder(d)
    if len(p): print("    {:24s} @0.5={:5.1f}%  @0.7={:5.1f}%  (n={})".format(n,(p>=0.5).mean()*100,(p>=0.7).mean()*100,len(p)))
line("ElevenLabs(American)","/tmp/eval/done/pos")
for d in sorted(glob.glob("/tmp/eval/accent/*")): line("accent "+os.path.basename(d), d)
PYEOF
  ) | tee -a "$RES"
}

for M in 1 3 6; do
  run_config "$M" || echo "  !! config x$M errored, continuing" | tee -a "$RES"
done
echo "" | tee -a "$RES"
echo "=== SWEEP DONE $(date) — compare the 3 configs above (accent x1 / x3 / x6) ===" | tee -a "$RES"
