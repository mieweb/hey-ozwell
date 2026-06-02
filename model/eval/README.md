# Wake-Word Evaluation Harness

Offline evaluation of the trained wake-word ONNX models against labeled audio.
**Inference only — no `piper-phonemize` required**, so this runs on macOS (where the
heybuddy training CLI cannot). It replicates the runtime preprocessing in
[`prod/js/src/models/`](../../prod/js/src/models/): mel-spectrogram → speech-embedding
→ wake-word classifier.

This produced the **first recorded metrics** for the shipped MLP models (June 2026);
before this they had none.

## Setup

```bash
cd model/eval
pip install onnxruntime soundfile scipy numpy   # all inference-only, install fine on macOS

# 1. Fetch the shared upstream models (the JS runtime loads these from Hugging Face)
mkdir -p pretrained
HF=https://huggingface.co/benjamin-paine/hey-buddy/resolve/main/pretrained
curl -sL -o pretrained/mel-spectrogram.onnx  "$HF/mel-spectrogram.onnx"
curl -sL -o pretrained/speech-embedding.onnx "$HF/speech-embedding.onnx"

# 2. Extract the labeled test clips from the dataset (note the apostrophe in the path)
cd ../..    # repo root
unzip -o -j model/data/data.zip "data/hey-ozwell/test/positive/*.wav"      -d /tmp/eval/hey/pos
unzip -o -j model/data/data.zip "data/hey-ozwell/test/negative/*.wav"      -d /tmp/eval/hey/neg
unzip -o -j model/data/data.zip "data/ozwell-i'm-done/test/positive/*.wav" -d /tmp/eval/done/pos
unzip -o -j model/data/data.zip "data/ozwell-i'm-done/test/negative/*.wav" -d /tmp/eval/done/neg
```

## Run

```bash
cd model/eval

python evaluate_wakeword.py \
  --model ../../prod/js/models/hey-ozwell.onnx \
  --positives /tmp/eval/hey/pos --negatives /tmp/eval/hey/neg \
  --label "hey ozwell"

python evaluate_wakeword.py \
  --model "../../prod/js/models/ozwell-i'm-done.onnx" \
  --positives /tmp/eval/done/pos --negatives /tmp/eval/done/neg \
  --label "ozwell i'm done"
```

## Baseline results (June 2026, synthetic test audio)

| Phrase | recall @0.5 | per-clip FPR @0.5 | verdict |
|---|---|---|---|
| `hey ozwell` (start) | ~98–99% | ~20% | strong recall; weakness is hard-negative false positives |
| `ozwell i'm done` (stop) | ~20–30% | ~15% | barely detects — flagged for retraining |

Numbers are rounded ranges: CPU inference isn't bit-exact run-to-run, and `ozwell i'm done`
has many clips near the 0.5 boundary, so its recall swings ~10 points per run (itself a sign
of how uncertain the model is). The figures and qualitative conclusions are stable.

## Caveats — do not over-quote these numbers

- **Synthetic audio.** The clips are ElevenLabs TTS: clean, accent-free, silent room.
  Real-world recall will be lower and FPR behavior different. Treat as an optimistic ceiling.
- **`per-clip FPR` ≠ `false positives per hour`.** Each clip is scored as max over ~13
  sliding windows, so it gets many chances to fire. The `<1 FP/hour` production target
  needs a separate streaming test against real continuous audio.
- **Held-out status unverified.** The shipped models' training recipe was never captured
  (see [`../README.md`](../README.md)), so we cannot prove the test split was unseen.

## TODO

- Streaming FPR test against real continuous audio (the real `<1 FP/hour` measure).
- Evaluate against real human recordings, not just synthetic TTS.
- Precision-recall curve generation per phrase (threshold tuning).
