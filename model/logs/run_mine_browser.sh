#!/usr/bin/env bash
# Browser-faithful hard-negative mining over spontaneous + read corpora, GPU, both phrases.
# Harvest thr 0.3 = fires + near-fires (the hardest, boundary negatives). Output -> precalculated/mined_ff_browser_{done,hey}.npy
set -e
cd "$(dirname "$0")/.."
source .venv-gpu/bin/activate
NVD=.venv-gpu/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $NVD -name 'lib' -type d 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH
export OMP_NUM_THREADS=2
CORPORA="/tmp/fphour/thirdparty_ami /tmp/fphour/voxpopuli_test /tmp/fphour/peoples_1h /tmp/fphour/peoples_test"

python mine_browser_ff.py --phrase done --audio-dirs $CORPORA --thr 0.3 --gpu --device-id 0 \
    --out precalculated/mined_ff_browser_done.npy > logs/mine_browser_done.log 2>&1 &
echo "done-mine PID $! (GPU0) -> logs/mine_browser_done.log"

python mine_browser_ff.py --phrase hey --audio-dirs $CORPORA --thr 0.3 --gpu --device-id 1 \
    --out precalculated/mined_ff_browser_hey.npy > logs/mine_browser_hey.log 2>&1 &
echo "hey-mine PID $! (GPU1) -> logs/mine_browser_hey.log"
echo "mining ~2.5h of audio across 4 corpora, both phrases, in background."
