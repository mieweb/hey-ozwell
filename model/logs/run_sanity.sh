#!/usr/bin/env bash
# Corrected sanity run for "ozwell i'm done".
# Fixes the OOM from the prior attempt: caps batch threads (12 -> 2) and
# shrinks the validation/testing sets (25k -> 500), which were left at default
# and peaked RAM at 22 GB. Training-set sizes already small (200/200).
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate

echo "=== sanity run started $(date) ==="
python -m heybuddy train "ozwell i'm done" \
  --perceptron \
  --positive-samples 200 --adversarial-samples 200 \
  --validation-samples 500 \
  --testing-positive-samples 500 --testing-adversarial-samples 500 \
  --steps 50 \
  --num-batch-threads 2 \
  --augmentation-dataset-streaming \
  --debug
code=$?
echo "=== sanity run EXIT_CODE=$code  $(date) ==="
