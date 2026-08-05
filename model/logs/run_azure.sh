#!/usr/bin/env bash
set -uo pipefail
cd /home/jlocala/hey-ozwell/model
source .venv/bin/activate
echo "[azure] waiting for Google accent build to finish..."
while ! grep -q "DONE TRAIN" logs/accent_gen.log 2>/dev/null; do sleep 30; done
echo "[azure] Google build done; starting Azure generation $(date)"
python gen_azure_accents.py 15
