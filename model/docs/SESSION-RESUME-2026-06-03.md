# Resume context — ozwell-i'm-done retrain (2026-06-03)

Drop this into a fresh Claude session if the chat is lost. Work from `~/hey-ozwell/model`,
`source .venv/bin/activate`. Read [audio-scale-mismatch.md](audio-scale-mismatch.md) and
[retrain-ozwell-im-done.md](retrain-ozwell-im-done.md) for detail.

## The big result of the day
The shipped `ozwell i'm done` wake word was broken (~24% recall). Root cause found and fixed:
a **train/inference audio-loudness mismatch**. The mel→embedding pipeline's output scale depends on
per-clip **loudness**, and the data sources were at different loudness (Piper positives ~1.0,
freesound ~0.1, mic/eval ~0.7) → the model was separating by loudness, not content.

**Fix (implemented + verified):** peak-normalize every clip to 1.0 right before the mel model, in
**both** the training pipeline and the eval harness, so loudness is irrelevant.
- `heybuddy/embeddings.py` (`SpeechEmbeddings.__call__`): replaced upstream `audio_tensor *= 32767.0`
  with per-clip peak-normalization. This one function covers training positives AND extracted negatives.
- `eval/evaluate_wakeword.py` (`load_16k_mono`): same peak-normalization.
- Verified: loud Piper (peak 1.0) and quiet freesound (peak 0.07) now produce matching embeddings (~53).

After the fix, retraining at consistent scale: **recall 24% → 93%** at production scale. Recall is fixed.

## Current open problem (what's being worked on)
After the recall fix, FPR was **71%** — the model fires on general speech. Cause: the training
**negatives were the wrong domain** (freesound sound-effects + near-wake-word phrases), so the model
never learned to reject ordinary speech. The eval negatives are general speech → it fires on them.

**In progress:** building a correct negative set of REAL speech + sounds, all peak-normalized:
- `heybuddy extract` real human speech from **LibriSpeech** (`openslr/librispeech_asr`, config `clean`,
  split `train.100`, `--transcript-key text`) → `heybuddy/precalculated/negs_libri/` (running in tmux `exl`).
- Already have freesound sound negatives in `heybuddy/precalculated/negs_pk/0.npy` (peak-normed).

### Next steps (do these)
1. Wait for tmux `exl` to finish (check `cat logs/exl.done`). Verify embedding-frame scale ~50-70
   (frames `[:, :16, :]`; the 17th frame is BERT token labels — ignore it for scale checks).
2. Combine speech + sound negatives into one training set, e.g.:
   `python -c "import numpy as np; a=np.load('heybuddy/precalculated/negs_libri/0.npy'); b=np.load('heybuddy/precalculated/negs_pk/0.npy'); np.save('heybuddy/precalculated/negs_all.npy', np.concatenate([a,b]))"`
3. Retrain (fast signal) with the combined negatives:
   `python -m heybuddy train "ozwell i'm done" --perceptron --positive-samples 5000 --adversarial-samples 5000 --steps 1000 --stages 3 --validation-samples 5000 --testing-positive-samples 5000 --testing-adversarial-samples 5000 --num-batch-threads 2 --augmentation-dataset-streaming --training-dataset heybuddy/precalculated/negs_all.npy --validation-no-default-dataset --debug`
4. Export + eval on the (peak-normed) harness. Test clips: `/tmp/eval/done/{pos,neg}` (ElevenLabs,
   held-out) and `/tmp/eval/piper_pos` (in-domain). Target: keep recall ~90%, drop FPR well under 20%.
   Export: `PYTHONPATH=. python tools/convert.py --input checkpoints/ozwell_i_m_done_final.pt --output checkpoints/scratch-onnx/<name>.onnx` (or use the inline legacy `dynamo=False` exporter — see runbook).
   Eval: `cd eval && python evaluate_wakeword.py --model <onnx> --positives /tmp/eval/done/pos --negatives /tmp/eval/done/neg --pretrained-dir pretrained --label X`
5. Also worth using the REAL `data.zip` train negatives (556 ElevenLabs clips, same source as eval):
   extract to `/tmp/train/done/neg` from `model/data/data.zip` `data/ozwell-i'm-done/train/negative/`.

## Known gaps (NOT done yet)
- **`prod/js` does NOT have the peak-norm.** The browser runtime still feeds un-normalized audio to the
  mel model. The fix must be added there too (in `audio.js`/`hey-buddy.js`, before `spectrogram.run`)
  or the model won't behave the same in the real app. REQUIRED before production.
- These are 5k fast-signal runs. The **full production retrain** (100k samples) hasn't been run.
- `--validation-no-default-dataset` means the per-hour FP target isn't actively enforced (validation
  FP/HR was nan); fine for fast signals, revisit for the real run.

## Environment gotchas (all fixed, keep in mind)
Python 3.11 venv; torch cu124; deps added: `tokenizers phonemizer onnxscript pyinstrument` (+ the
ones from the runbook). FFmpeg errors in logs are HARMLESS (the AAC-codec aug that needs it is dead
code; PyAV handles reading). Checkpoints land in `model/checkpoints/`, not `exports/`. `unzip` not
installed — use Python `zipfile`. Disk is only 49 GB — the default precalc negative sets are 20 GB,
do NOT download them; generate local negatives instead. Run long jobs in tmux.

## How to check what's running after a reconnect
`tmux ls`  → attach with `tmux attach -t <name>`.  `nvidia-smi` for GPU activity.
Latest trained model: `ls -la model/checkpoints/ozwell_i_m_done_final.pt`.
