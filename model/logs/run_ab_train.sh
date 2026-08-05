#!/usr/bin/env bash
# Wait for all browser-faithful data, then A/B train (baseline vs +hard-negs) for both phrases.
# Fully detached (setsid nohup) -> survives laptop/connection close. Result -> logs/ab_train.log
cd "$(dirname "$0")/.."
NEED="precalculated/nbf_done_train/pos_clean.npy precalculated/nbf_hey_train/pos_clean.npy \
precalculated/nbf_done_test/pos_clean.npy precalculated/nbf_hey_test/pos_clean.npy \
precalculated/browser_negs_train.npy precalculated/browser_negs_eval.npy \
precalculated/mined_ff_train_done.npy precalculated/mined_ff_train_hey.npy"
until ls $NEED >/dev/null 2>&1; do sleep 10; done
sleep 3
source .venv/bin/activate
echo "=== all data ready $(date) ==="; ls -la $NEED | awk '{print $5, $9}'
echo "############ DONE ############"; OMP_NUM_THREADS=4 python train_browser.py --phrase done 2>&1 | grep -vi "pthread\|affinity"
echo "############ HEY ############";  OMP_NUM_THREADS=4 python train_browser.py --phrase hey  2>&1 | grep -vi "pthread\|affinity"
echo "=== A/B COMPLETE $(date) ==="
