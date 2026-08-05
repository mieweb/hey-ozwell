#!/usr/bin/env bash
# Fast-signal retrain for "ozwell i'm done" (intermediate run before the full 100k).
# Goal: a quick read on whether the corrected pipeline beats the ~30% baseline.
# Memory guardrails: --num-batch-threads 2, val/test 5k (22 GB RAM ceiling; threads=12 OOM'd).
# Architecture forced via --perceptron (click default-resolution bug yields architecture=False).
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate

echo "=== fast-signal retrain started $(date) ==="
python -m heybuddy train "ozwell i'm done" \
  --perceptron \
  --positive-samples 5000 \
  --adversarial-samples 5000 \
  --steps 1000 --stages 3 \
  --validation-samples 5000 \
  --testing-positive-samples 5000 \
  --testing-adversarial-samples 5000 \
  --num-batch-threads 2 \
  --augmentation-dataset-streaming \
  --debug
code=$?
echo "=== fast-signal retrain EXIT_CODE=$code  $(date) ==="
