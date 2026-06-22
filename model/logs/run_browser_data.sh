#!/usr/bin/env bash
# Browser-faithful retrain data. Clean A/B split:
#   TRAIN negs (bulk + hard) from AMI + peoples_1h ; HELD-OUT eval negs from voxpopuli_test + peoples_test.
set -e
cd "$(dirname "$0")/.."
source .venv-gpu/bin/activate
NVD=.venv-gpu/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $NVD -name 'lib' -type d 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH
export OMP_NUM_THREADS=2
TRAIN="/tmp/fphour/thirdparty_ami /tmp/fphour/peoples_1h"
EVAL="/tmp/fphour/voxpopuli_test /tmp/fphour/peoples_test"

# hard negs (re-mined from TRAIN corpora only, so eval stays clean), both phrases, GPU0/GPU1
python mine_browser_ff.py --phrase done --audio-dirs $TRAIN --thr 0.3 --gpu --device-id 0 \
    --out precalculated/mined_ff_train_done.npy > logs/d_hard_done.log 2>&1 &
python mine_browser_ff.py --phrase hey  --audio-dirs $TRAIN --thr 0.3 --gpu --device-id 1 \
    --out precalculated/mined_ff_train_hey.npy  > logs/d_hard_hey.log 2>&1 &
# bulk train negs (GPU2) + held-out eval negs (GPU3)
python gen_browser_negs.py --audio-dirs $TRAIN --keep 0.25 --cap 40000 --gpu --device-id 2 \
    --out precalculated/browser_negs_train.npy > logs/d_bulk.log 2>&1 &
python gen_browser_negs.py --audio-dirs $EVAL  --keep 0.5  --cap 30000 --gpu --device-id 3 \
    --out precalculated/browser_negs_eval.npy  > logs/d_evalneg.log 2>&1 &
wait
echo "ALL browser-faithful negative data done."
tail -2 logs/d_hard_done.log logs/d_hard_hey.log logs/d_bulk.log logs/d_evalneg.log
