#!/usr/bin/env bash
# Reproducible retrain for "ozwell i'm done". See docs/retrain-ozwell-im-done.md.
#
# Usage (run from the model/ directory, inside tmux for the long runs):
#   tmux new -s retrain
#   scripts/retrain.sh            # full run  (100k samples, 5000 steps) ~hours
#   scripts/retrain.sh fast       # fast signal (5k samples, 1000 steps) ~15-30 min
#   scripts/retrain.sh sanity     # tiny end-to-end smoke test (200/50)  ~3 min
#
# Mandatory, non-obvious settings baked in (do not remove):
#   --perceptron            : without it the CLI bug sets architecture=False -> crash
#   --num-batch-threads 2   : defaults (12) + 25k val/test OOM-kill on a 22 GB box
# Trained model lands in checkpoints/ozwell_i_m_done_final.pt (NOT exports/heybuddy/).
set -uo pipefail
cd "$(dirname "$0")/.."                      # -> model/
source .venv/bin/activate

MODE="${1:-full}"
PHRASE="ozwell i'm done"
mkdir -p logs

case "$MODE" in
  full)   POS=100000; ADV=100000; STEPS=5000; VT=10000;  EXTRA="--adversarial-phrases 250 --target-false-positive-rate 1.5" ;;
  fast)   POS=5000;   ADV=5000;   STEPS=1000; VT=5000;   EXTRA="" ;;
  sanity) POS=200;    ADV=200;    STEPS=50;   VT=500;    EXTRA="" ;;
  *) echo "unknown mode: $MODE (use full|fast|sanity)"; exit 2 ;;
esac

LOG="logs/retrain_${MODE}.log"
echo "=== retrain mode=$MODE started $(date) ===" | tee "$LOG"
git rev-parse HEAD 2>/dev/null | tee -a "$LOG"

python -m heybuddy train "$PHRASE" \
  --perceptron \
  --positive-samples "$POS" --adversarial-samples "$ADV" \
  --steps "$STEPS" --stages 3 \
  --validation-samples "$VT" \
  --testing-positive-samples "$VT" --testing-adversarial-samples "$VT" \
  --num-batch-threads 2 \
  --augmentation-dataset-streaming \
  --debug $EXTRA 2>&1 | tee -a "$LOG"

code=${PIPESTATUS[0]}
echo "=== retrain mode=$MODE EXIT_CODE=$code  $(date) ===" | tee -a "$LOG"
echo "checkpoint: $(ls -la checkpoints/ozwell_i_m_done_final.pt 2>/dev/null || echo MISSING)"
exit "$code"
