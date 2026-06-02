# Runbook: Retrain `ozwell i'm done`

**Goal:** the shipped `ozwell i'm done` model scores ~20–30% recall (see [`../eval/`](../eval/)) — effectively non-functional. Retrain it with a full, diverse, **recorded** config and verify with the eval harness.

**Why this should help:** the Piper generator is *capable* of high diversity (904 speakers, speed/noise variation, default 100k positives), and the heybuddy trainer already does negative-weighting + FPR targeting. The shipped model's recipe was **never captured**, so the leading theory is it was under-configured. This run uses near-default (large, diverse) settings and **logs the exact command**, fixing both problems at once.

> ⚠️ This was written from reading the code on macOS — it has **not** been run on Linux. Spots most likely to need a tweak on first contact are marked ⚠️.

---

## 0. Prereqs
- Linux container (Ubuntu), **GPU (CUDA)**, root/sudo, outbound internet.
- This repo cloned; work from the `model/` directory.

## 1. Install
```bash
sudo apt-get update && sudo apt-get install -y espeak-ng        # piper phonemizer system dep
cd model
pip install -r requirements.txt
pip install piper-phonemize                                     # ⚠️ Linux-only — the thing that fails on macOS
# ⚠️ if torch doesn't see the GPU, install the CUDA build, e.g.:
#    pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must print True
python -m heybuddy --help                                       # must succeed (this is what dies on macOS)
```

## 2. Sanity: tiny end-to-end run BEFORE the real one
Confirms data-gen (Piper TTS) → train → checkpoint works, fast, before spending GPU time.
```bash
python -m heybuddy train "ozwell i'm done" \
    --positive-samples 200 --adversarial-samples 200 --steps 50 --debug
```
Watch for: Piper model downloads, positives generate, training loop runs, a checkpoint is written.
⚠️ Note where the `.pt` lands (checkpoint dir / `exports/heybuddy/`) — you'll point `convert.py` at it.

## 3. Listen to the generated Piper positives (the deferred data-quality check)
Dump a handful of synthesized positives and actually listen — confirms diversity/quality directly.
```python
# save as /tmp/dump_positives.py, run: python /tmp/dump_positives.py
import soundfile as sf
from heybuddy.dataset.piper import PiperSpeechGenerator
gen = PiperSpeechGenerator(phrase="ozwell i'm done")   # ⚠️ confirm constructor args vs your version
for i, s in zip(range(20), gen(num_samples=20)):
    sf.write(f"/tmp/piper_{i:02d}.wav", s["audio"]["array"], s["audio"]["sampling_rate"])
print("wrote 20 clips to /tmp/piper_*.wav")
```
Listen (`afplay`/`aplay`) — are they varied (different voices/speeds) and is the "I'm" contraction clean? If many are garbled, that's a data-gen problem to fix before the full run.

## 4. The real retrain (near-default = large + diverse), and RECORD it
```bash
# Record the exact command + commit so the recipe is never lost again:
git rev-parse HEAD | tee training-run.log
python -m heybuddy train "ozwell i'm done" \
    --positive-samples 100000 \
    --adversarial-samples 100000 \
    --adversarial-phrases 250 \
    --steps 5000 --stages 3 \
    --target-false-positive-rate 1.5 \
    --wandb-entity "<your-wandb-or-omit>" \
    2>&1 | tee -a training-run.log
```
Defaults (already large/diverse): positives/adversarials 100k, 250 adversarial phrases, 3 stages, 5000 steps, augmentation (MIT impulse + background datasets) on. Architecture = `perceptron` (the MLP). Start here; only deviate with a reason.
⚠️ 100k samples on first run may be slow — if you just want a fast signal first, rerun §2 with `--positive-samples 5000 --steps 1000` and check the harness before committing to the full run.

## 5. Export to ONNX
```bash
# ⚠️ point --input at the .pt produced in step 4 (verify the exact path/name)
python tools/convert.py \
    --input exports/heybuddy/ozwell_i_m_done_final.pt \
    --output "../prod/js/models/ozwell-i'm-done.onnx"
```

## 6. Verify with the eval harness (this is the whole point)
```bash
cd eval
mkdir -p pretrained
HF=https://huggingface.co/benjamin-paine/hey-buddy/resolve/main/pretrained
curl -sL -o pretrained/mel-spectrogram.onnx  "$HF/mel-spectrogram.onnx"
curl -sL -o pretrained/speech-embedding.onnx "$HF/speech-embedding.onnx"
cd ..
unzip -o -j data/data.zip "data/ozwell-i'm-done/test/positive/*.wav" -d /tmp/eval/done/pos
unzip -o -j data/data.zip "data/ozwell-i'm-done/test/negative/*.wav" -d /tmp/eval/done/neg
cd eval
python evaluate_wakeword.py \
    --model "../../prod/js/models/ozwell-i'm-done.onnx" \
    --positives /tmp/eval/done/pos --negatives /tmp/eval/done/neg \
    --pretrained-dir pretrained --label "ozwell i'm done (retrained)"
python plot_eval.py --pretrained-dir pretrained --outdir figures \
    --phrase "ozwell i'm done (retrained)" "../../prod/js/models/ozwell-i'm-done.onnx" /tmp/eval/done/pos /tmp/eval/done/neg
```

## 7. Read the result honestly
- **Recall climbs out of the ~20–30% overlap** → it was an under-configured run; this fixes it. Compare before/after figures.
- **Still stuck** → it's the Piper→ElevenLabs domain gap or the phrase itself (the contraction). Escalate to: real human recordings in the test set, and/or reconsidering the phrase. Note: testing on ElevenLabs while training on Piper is a domain mismatch — a fairer check is to also hold out some Piper-generated positives.

## Caveats carried from eval
- The data.zip test set is **synthetic** (ElevenLabs) — optimistic vs. real clinicians.
- Per-clip FPR ≠ FP/hour; the `--target-false-positive-rate` flag targets the real per-hour notion during training, but validate separately on continuous audio later.
