#!/usr/bin/env bash
# REBALANCE sweep: keep the CLEAN surgical positives fixed (which gave 92% recall / 6.4 FP),
# vary ONLY the negatives to pull FP back down toward prod's 3.6/hr WITHOUT losing recall.
# This is an empirical sweep (no formula) over 2 negative levels:
#   conf    = negs_C(160k) + confusables(8k)                 ~168k
#   voxconf = negs_C(160k) + VoxPopuli(55k) + confusables(8k) ~223k
# All non-"ozwell"-sharing negatives (safe for ozwell-done recall). Eval: n=390 recall + multi-source FP.
# Compare to prod (86% / 3.6/hr) and surgical (92% / 6.4/hr). Adopt the one that holds recall + lowest FP.
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
P=heybuddy/precalculated
RES=logs/FINAL_RESULTS_REBAL.txt
PRODDONE="../../prod/js/models/ozwell-i'm-done.onnx"
echo "=== ozwell-done REBALANCE sweep (clean surgical positives, vary negatives) $(date) ===" > "$RES"

# fixed positives = config-C base + clean surgical accents
[ -f "$P/ozwell_i_m_done.prebase.npy" ] || cp "$P/ozwell_i_m_done.npy" "$P/ozwell_i_m_done.prebase.npy"
POS=$(python3 -c "
import numpy as np
b=np.load('$P/ozwell_i_m_done.prebase.npy'); a=np.load('$P/eleven_surgical_pos_done.npy')
np.save('$P/ozwell_i_m_done.npy', np.concatenate([b,a]).astype('float32')); print(len(b)+len(a))")
echo "positives (fixed) = ${POS} (127384 base + 1790 clean surgical accent)" | tee -a "$RES"

# build the two negative sets
python3 -c "
import numpy as np, glob
P='$P/'; rng=np.random.default_rng(0)
negs=np.load(P+'negs_C.npy')                                   # [160k,17,96]
conf=np.load(P+'confusable_negs.npy')                          # [8k,16,96] -> needs token row
tok=negs[rng.choice(len(negs),size=len(conf),replace=True),-1:,:]
conf_l=np.concatenate([conf,tok],axis=1)                       # [8k,17,96]
vox=np.concatenate([np.load(f) for f in sorted(glob.glob(P+'negs_voxpopuli/*.npy'))])  # [~55k,17,96]
np.save(P+'negs_rebal_conf.npy',    np.concatenate([negs,conf_l]).astype('float32'))
np.save(P+'negs_rebal_voxconf.npy', np.concatenate([negs,vox,conf_l]).astype('float32'))
print('negs_rebal_conf', len(negs)+len(conf_l), '| negs_rebal_voxconf', len(negs)+len(vox)+len(conf_l))
" 2>&1 | grep -vE "Warning|pthread" | tee -a "$RES"

run_cfg () {
  local NEG=$1 TAG=$2; local ONNX="checkpoints/scratch-onnx/ozwelldone_${TAG}.onnx"
  local NC; NC=$(python3 -c "import numpy as np;print(len(np.load('$P/${NEG}')))")
  echo "" | tee -a "$RES"; echo "########## ${TAG}: ${POS} pos / ${NC} neg ($(date)) ##########" | tee -a "$RES"
  python -m heybuddy train "ozwell i'm done" --perceptron \
    --positive-samples "$POS" --adversarial-samples 100000 --adversarial-phrases 250 \
    --steps 3500 --stages 3 --target-false-positive-rate 1.5 \
    --validation-samples 2000 --testing-positive-samples 2000 --testing-adversarial-samples 2000 \
    --num-batch-threads 3 --augmentation-dataset-streaming \
    --training-dataset "$P/${NEG}" --validation-no-default-dataset --debug > "logs/rebal_${TAG}_train.log" 2>&1
  echo "  train exit=$? ($(date))" | tee -a "$RES"
  python3 -c "
import torch
from heybuddy.wakeword import WakeWordMLPModel
m=WakeWordMLPModel.from_file('checkpoints/ozwell_i_m_done_final.pt'); m.to('cpu').eval()
torch.onnx.export(m, torch.randn(m.input_shape).unsqueeze(0), '${ONNX}', opset_version=19, input_names=['input'], output_names=['output'], dynamo=False)
print('  exported')" 2>&1 | grep -E "exported|Error" | tee -a "$RES"
  ( cd eval && python3 - <<PYEOF 2>&1 | grep -vE "pthread|onnxruntime|Warning|warn|^INFO|DEBUG"
import glob, numpy as np
from evaluate_wakeword import WakeWordEvaluator
ev=WakeWordEvaluator("../${ONNX}","pretrained"); allp=[]
for d in sorted(glob.glob("/tmp/eleven_big/test/*/ozwell_done")):
    if not glob.glob(d+'/*.wav'): continue
    p=ev.score_folder(d); allp.append(p); acc=d.split('/')[-2]
    print(f"    {acc:11s} {(p>=0.5).mean()*100:3.0f}%")
p=np.concatenate(allp); print(f"    {'ALL':11s} {(p>=0.5).mean()*100:3.0f}%  (prod 86, surgical 92)")
PYEOF
  ) | tee -a "$RES"
  echo "    multi-source FP/hr @0.5 (prod 3.6, surgical 6.4):" | tee -a "$RES"
  for s in voxpopuli_test peoples_test thirdparty_ami combined; do
    [ -d "/tmp/fphour/$s" ] || continue
    printf "      %-16s" "$s" | tee -a "$RES"
    ( cd eval && python3 fp_per_hour.py --model "../${ONNX}" --audio-dir "/tmp/fphour/$s" \
        --pretrained-dir pretrained --label "$TAG" --thresholds "0.5" 2>&1 | grep -E "^  0\." | sed 's/^/ /' ) | tee -a "$RES"
  done
}

run_cfg "negs_rebal_conf.npy"    "rebal_conf"
run_cfg "negs_rebal_voxconf.npy" "rebal_voxconf"
cp "$P/ozwell_i_m_done.prebase.npy" "$P/ozwell_i_m_done.npy"   # restore base
echo "" | tee -a "$RES"; echo "=== REBAL DONE $(date) — pick the config that holds ~92% recall with FP nearest prod's 3.6/hr ===" | tee -a "$RES"
