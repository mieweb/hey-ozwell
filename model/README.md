# Model — Hey Ozwell Wake-Word Training

This folder is the **single source of truth** for the Hey Ozwell wake-word model: training framework, training data, and exported artifacts that the JS runtime in [`prod/js/`](../prod/js/) consumes.

## Lineage

We use the **heybuddy MLP** architecture (a fork of [painebenjamin/hey-buddy](https://github.com/painebenjamin/hey-buddy), Apache-2.0).

- Per-wake-word classifier: tiny MLP over a window of 16 speech-embedding frames (96 dims each) → ~27 KB ONNX
- Shared upstream pipeline at inference time: mel-spectrogram → speech-embedding → MLP classifier → Silero VAD gating
- The shared upstream ONNX models (`silero-vad`, `mel-spectrogram`, `speech-embedding`) are loaded by the JS runtime directly from Hugging Face: <https://huggingface.co/benjamin-paine/hey-buddy>

The earlier custom Conv2d pipeline is preserved under [`legacy/`](./legacy/).

## Layout

| Path | Purpose |
|---|---|
| `heybuddy/` | Vendored Hey Buddy Python package (training, conversion, datasets, modules). Upstream: <https://github.com/painebenjamin/hey-buddy>. Apache-2.0. |
| `tools/convert.py` | `.pt` → `.onnx` conversion, output lands in `../prod/js/models/` |
| `exports/heybuddy/*.pt` | Trained PyTorch checkpoints (LFS) — currently `hey_ozwell_final.pt`, `ozwell_i_m_done_final.pt` |
| `data/` | Training dataset (LFS-stored `data.zip`, see [`data/README.md`](./data/README.md)) |
| `docs/onnx_export/` | PyTorch→ONNX export reports captured during the original conversion |
| `legacy/` | Archived custom Conv2d pipeline (4 wake words, ~60 MB models) |
| `requirements.txt` | Python deps for training and conversion |

## Quickstart: convert existing `.pt` → runtime ONNX

```bash
cd model
pip install -r requirements.txt

# Default emits ozwell-i'm-done.onnx into prod/js/models/
python tools/convert.py

# Or specify both:
python tools/convert.py \
    --input exports/heybuddy/hey_ozwell_final.pt \
    --output ../prod/js/models/hey-ozwell.onnx
```

## Training

> **Status — TODO**: the original training invocations were not captured in the upstream `hey-ozwell-demo` repo. The CLI surface below was reverse-engineered from [`heybuddy/__main__.py`](./heybuddy/__main__.py). Treat as a starting template until validated against amandamarg's original workflow.

```bash
cd model
pip install -r requirements.txt
unzip data/data.zip -d data/   # extracts positive + negative audio

# Train one wake word
python -m heybuddy train "hey ozwell" \
    --steps 5000 \
    --threshold 0.5 \
    --positive-samples 1000 \
    --adversarial-samples 1000

# Convert to ONNX (drops into prod/js/models/)
python tools/convert.py \
    --input ./hey-ozwell-final.pt \
    --output ../prod/js/models/hey-ozwell.onnx
```

The full set of `train` options (W&B logging, augmentation probabilities, batch sizes, validation/testing splits) lives in [`heybuddy/__main__.py`](./heybuddy/__main__.py) — run `python -m heybuddy train --help` for the live list.

## Roadmap

- [ ] Validate training recipe end-to-end on `data/data.zip` and reproduce `hey-ozwell.onnx`.
- [ ] Add CI smoke test: `python -c "from heybuddy.wakeword import WakeWordMLPModel"`.
- [ ] Document hardware requirements (GPU memory, training duration) once validated.

## Credits

- **Hey Buddy** by Benjamin Paine — <https://github.com/painebenjamin/hey-buddy> (Apache-2.0). Vendored in `heybuddy/`.
- **Wake-word demo + initial Ozwell models** by [@amandamarg](https://github.com/amandamarg) — <https://github.com/amandamarg/hey-ozwell-demo>.
- **Training data generation tooling** also by [@amandamarg](https://github.com/amandamarg) — <https://github.com/amandamarg/hey-ozwell-data> (mirrored into `data/`).
