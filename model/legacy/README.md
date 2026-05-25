# Legacy: Custom Conv2d Wake-Word Pipeline (Lineage B)

> **Status**: Archived. Superseded by the heybuddy MLP pipeline at [`model/`](../).
> Preserved for reference and possible future revival.

This folder contains the custom-Conv2d wake-word training and evaluation pipeline contributed by [@amandamarg](https://github.com/amandamarg). It is **not** compatible with the JavaScript runtime in [`prod/js/`](../../prod/js/), which expects the heybuddy MLP architecture (mel-spectrogram → speech-embedding → tiny MLP classifier).

The original Lineage B README is preserved at [`README-original.md`](./README-original.md).

## Why archived

When this repo was consolidated as the single source of truth (see the root [`README.md`](../../README.md)), we chose the heybuddy lineage because:

- The JavaScript runtime is proven end-to-end with the heybuddy MLP models.
- Models are tiny (~27 KB vs ~15 MB for Conv2d).
- The full Hey Buddy training framework is now vendored in [`model/heybuddy/`](../heybuddy/).

The Conv2d pipeline is left intact here so that:

1. The reproducible training recipe (with logs and eval results) isn't lost.
2. A future contributor could revisit Conv2d if benchmarks justify it.
3. The 4-wake-word coverage (`hey-ozwell`, `go-ozwell`, `ozwell-go`, `ozwell-i'm-done`) — broader than the heybuddy lineage's current 2 — is documented.

## What's here

| Path | Description |
|---|---|
| `tools/train.py`, `train_all.py`, `prepare_data.py` | Custom training scripts (mel-spectrogram → Conv2d → softmax) |
| `download_tools/` | ElevenLabs TTS download scripts (the canonical training data lives at [`model/data/`](../data/)) |
| `exports/` | Trained `.pth` + exported `.onnx` artifacts for 4 wake words (LFS-stored, ~60 MB total) |
| `logs/training/`, `logs/data_prep/`, `logs/testing/` | Run logs from amandamarg's training sessions |
| `results/` | Evaluation reports and confusion matrix for `hey-ozwell` |
| `testing/evaluate.py` | Evaluation script |
| `negative_phrases.csv` | Negative-example phrase list used during dataset generation |
| `requirements.txt` | Python deps for this legacy pipeline |
| `test_components.py` | Component smoke tests |
| `README-original.md` | The original Lineage B README authored by @amandamarg |

## Running the legacy pipeline (if you ever need to)

```bash
cd model/legacy
pip install -r requirements.txt

# 1. Generate training data (or use model/data/data.zip if compatible)
python download_tools/download.py

# 2. Prepare manifests
python tools/prepare_data.py

# 3. Train one or all wake words
python tools/train_all.py
# or: python tools/train.py --wake-word hey-ozwell

# 4. Evaluate
python testing/evaluate.py
```

## Provenance

Original commits live on the `main` branch history; see `git log --follow model/legacy/tools/train.py`.
