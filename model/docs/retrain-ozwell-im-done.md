# Runbook: Retrain `ozwell i'm done`

**Goal:** the shipped `ozwell i'm done` model scores ~20–30% recall (see [`../eval/`](../eval/)) —
effectively non-functional. Retrain it with a full, diverse, **recorded** config and verify with
the eval harness.

**Why this should help:** the Piper generator is *capable* of high diversity (default 100k positives,
speed/noise/speaker variation), and the heybuddy trainer already does negative-weighting + FPR targeting.
The shipped model's recipe was **never captured**, so the leading theory is narrow-data overfitting /
under-configuration. This run uses near-default (large, diverse) settings and **logs the exact command**.

> ✅ **Verified end-to-end on Linux 2026-06-03** (4× Tesla V100-32GB, CUDA 12.4 driver, ~22 GB RAM).
> Every command below was actually run. Spots that still depend on your box are marked ⚠️.

---

## 0. Prereqs
- Linux container, **GPU (CUDA)**, outbound internet.
- This repo cloned; **work from the `model/` directory** unless noted.
- **~22 GB RAM minimum.** The feature/augmentation generator is RAM-hungry; at 4 GB it OOM-kills
  (kernel SIGKILL, no Python traceback — check `cat /sys/fs/cgroup/memory.events` for `oom_kill`).
  Memory peaked ~17.6 GB on the fast-signal run below; keep `--num-batch-threads` low (see §4).

## 1. Install (Python 3.11 + uv)
`piper-phonemize` has **no wheels for Python 3.12/3.13 or macOS** → must use 3.11.
```bash
cd model
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# Match torch's CUDA build to the DRIVER. For a CUDA 12.4 driver use cu124
# (the default cu130 build fails on this driver):
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
# Extra runtime deps the training path imports but requirements.txt misses:
uv pip install piper-phonemize phonemizer tokenizers \
    torch-audiomentations torch-pitch-shift torchaudio torchcodec
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must print True
python -m heybuddy --help                                            # must succeed
```
Notes / hard-won:
- **`tokenizers`** is needed by `BERTTokenizer` (negative-sampling token exclusion). Missing → the run
  starts training but spams `ModuleNotFoundError: No module named 'tokenizers'` per batch.
- **`phonemizer`** (separate from `piper-phonemize`) is needed by the Piper positive generator.
- **numpy 2.x shim:** heybuddy imports `numpy.compat` (removed in numpy 2.x).
  [`heybuddy/util/numpy_util.py`](../heybuddy/util/numpy_util.py) defines a shim
  (`numpy.compat.pickle` + `isfileobj`). **Keep this patch.**
- ⚠️ `espeak-ng` system package may be required for phonemization on a fresh box
  (`sudo apt-get install -y espeak-ng`). It was already present here.

## 2. Sanity: tiny end-to-end run BEFORE the real one
Confirms data-gen → train → checkpoint → metrics, fast, before spending GPU time.
```bash
python -m heybuddy train "ozwell i'm done" \
    --perceptron \
    --positive-samples 200 --adversarial-samples 200 \
    --validation-samples 500 \
    --testing-positive-samples 500 --testing-adversarial-samples 500 \
    --steps 50 --num-batch-threads 2 \
    --augmentation-dataset-streaming --debug
```
**Two non-obvious flags that are mandatory, not optional:**
- **`--perceptron`** — without it the CLI's click default-resolution is buggy and `architecture`
  evaluates to `False`, crashing with `ValueError: Invalid architecture: False` at
  [`trainer.py:267`](../heybuddy/trainer.py#L267). (Bug: in
  [`__main__.py:175-176`](../heybuddy/__main__.py#L175-L176) `--transformer`'s explicit `default=False`
  wins over `--perceptron`'s flag_value. Passing `--perceptron` forces it.)
- **`--num-batch-threads 2`** + reduced `--validation-samples`/`--testing-*-samples` — the defaults are
  `--num-batch-threads 12` and 25 000 val/test each, which **peaks RAM past 22 GB and OOM-kills**.
  The `--positive/adversarial-samples` flags only size the *training* set; val/test default
  independently and must be reduced too.

**Checkpoint location (IMPORTANT):** the model is written to **`model/checkpoints/`**, e.g.
`model/checkpoints/ozwell_i_m_done_final.pt` (+ `_optimizer.pt`) and a `model/ozwell_i_m_done_metrics.png`.
It is **NOT** in `exports/heybuddy/` — those `*.pt` (132 bytes) are stale **git-LFS pointers**, not models.
Expected sanity result: a real ~1 MB `.pt`, exit 0, Stage-3 recall ~0.95.

## 3. (Optional) Listen to the generated Piper positives
Data-quality spot check — dump a few synthesized positives and listen.
```python
# python /tmp/dump_positives.py
import soundfile as sf
from heybuddy.dataset.piper import PiperSpeechGenerator
gen = PiperSpeechGenerator(phrase="ozwell i'm done")   # ⚠️ confirm constructor args vs your version
for i, s in zip(range(20), gen(num_samples=20)):
    sf.write(f"/tmp/piper_{i:02d}.wav", s["audio"]["array"], s["audio"]["sampling_rate"])
```
Listen (`aplay /tmp/piper_*.wav`) — varied voices/speeds? Is the "I'm" contraction clean?

## 4. The real retrain — RUN IN tmux, and RECORD it
**Always run long training in tmux** so a VS Code window reload / disconnect can't kill it.
A convenience script lives at [`scripts/retrain.sh`](../scripts/retrain.sh) (see §4b); the explicit
command is:
```bash
tmux new -s retrain                 # detach later with Ctrl-b d ; reattach: tmux attach -t retrain
cd model && source .venv/bin/activate
git rev-parse HEAD | tee logs/training-run.log
python -m heybuddy train "ozwell i'm done" \
    --perceptron \
    --positive-samples 100000 \
    --adversarial-samples 100000 \
    --adversarial-phrases 250 \
    --steps 5000 --stages 3 \
    --target-false-positive-rate 1.5 \
    --validation-samples 10000 \
    --testing-positive-samples 10000 --testing-adversarial-samples 10000 \
    --num-batch-threads 2 \
    --augmentation-dataset-streaming --debug \
    2>&1 | tee -a logs/training-run.log
```
- **Keep `--perceptron` and `--num-batch-threads 2`.** The augmentation phase memory scales with sample
  count; at 100k watch `cat /sys/fs/cgroup/memory.current` and lower threads / val-test further if it
  climbs toward ~20 GB. (Fast-signal at 5k peaked ~17.6 GB.)
- **Fast-signal first (recommended):** run the above with `--positive-samples 5000 --adversarial-samples 5000
  --steps 1000 --validation-samples 5000 --testing-*-samples 5000` (~15–30 min) and evaluate before
  committing to the full 100k run (hours).
- `--augmentation-dataset-streaming` streams the freesound/MIT background+impulse datasets instead of
  downloading ~460 MB parquet files first (saves RAM/disk).

### 4b. Reproducible scripts
[`scripts/retrain.sh`](../scripts/retrain.sh) runs the full recipe; `scripts/retrain.sh fast` runs the
fast-signal variant. Both source `.venv`, set the memory guardrails, and log to `logs/`.

## 5. Export to ONNX
`tools/convert.py` imports `heybuddy`, so it must see `model/` on `sys.path` — run it with
**`PYTHONPATH=.` from `model/`** (a bare `python tools/convert.py` fails with
`ModuleNotFoundError: No module named 'heybuddy'`). Point `--input` at `checkpoints/` (§2), and
**do NOT overwrite the shipped baseline** while testing — export to a separate file first, compare, then
promote.
```bash
PYTHONPATH=. python tools/convert.py \
    --input checkpoints/ozwell_i_m_done_final.pt \
    --output "../prod/js/models/ozwell-i'm-done.retrained.onnx"
# Promote only after eval (§6) shows it beats the baseline:
#   cp "../prod/js/models/ozwell-i'm-done.retrained.onnx" "../prod/js/models/ozwell-i'm-done.onnx"
```

## 6. Verify with the eval harness (this is the whole point)
Inference-only (no piper-phonemize). See [`../eval/README.md`](../eval/README.md).
```bash
# (a) Pull LFS data + baseline models (they ship as 132-byte LFS pointers):
cd ..    # repo root
git lfs pull --include="model/data/data.zip,prod/js/models/hey-ozwell.onnx,prod/js/models/ozwell-i'm-done.onnx"

# (b) Stage the shared pretrained ONNX (already downloaded under heybuddy/pretrained/):
cd model
mkdir -p eval/pretrained
cp heybuddy/pretrained/mel-spectrogram.onnx  eval/pretrained/
cp heybuddy/pretrained/speech-embedding.onnx eval/pretrained/
#   (or curl them from HF per eval/README.md if heybuddy/pretrained/ is absent)

# (c) Extract test clips. NOTE: `unzip` is NOT installed on this box — use Python zipfile.
#     The archive also contains __MACOSX/ resource-fork junk (._*) to skip.
cd ..
python3 - <<'PY'
import zipfile, os
z = zipfile.ZipFile('model/data/data.zip')
sets = {'/tmp/eval/done/pos': "data/ozwell-i'm-done/test/positive/",
        '/tmp/eval/done/neg': "data/ozwell-i'm-done/test/negative/"}
for out, pre in sets.items():
    os.makedirs(out, exist_ok=True)
    for n in z.namelist():
        if n.startswith(pre) and n.lower().endswith('.wav') and '__MACOSX' not in n \
           and not os.path.basename(n).startswith('._'):
            open(os.path.join(out, os.path.basename(n)), 'wb').write(z.read(n))
    print(out, len(os.listdir(out)))
PY

# (d) Evaluate retrained vs baseline:
cd model/eval
python evaluate_wakeword.py --model "../../prod/js/models/ozwell-i'm-done.retrained.onnx" \
    --positives /tmp/eval/done/pos --negatives /tmp/eval/done/neg \
    --pretrained-dir pretrained --label "ozwell i'm done (retrained)"
python evaluate_wakeword.py --model "../../prod/js/models/ozwell-i'm-done.onnx" \
    --positives /tmp/eval/done/pos --negatives /tmp/eval/done/neg \
    --pretrained-dir pretrained --label "ozwell i'm done (baseline)"
python plot_eval.py --pretrained-dir pretrained --outdir figures \
    --phrase "ozwell i'm done (retrained)" "../../prod/js/models/ozwell-i'm-done.retrained.onnx" \
    /tmp/eval/done/pos /tmp/eval/done/neg
```

## 7. Read the result honestly
- **Recall climbs out of the ~20–30% overlap** → it was under-configured; this fixes it. Compare figures.
- **Still stuck** → likely the Piper→ElevenLabs domain gap or the phrase/contraction itself. Escalate to
  real human recordings in the test set, and/or also hold out some Piper-generated positives (training on
  Piper while testing on ElevenLabs is a domain mismatch).

## Hard-won fixes (quick reference)
| Symptom | Cause | Fix |
|---|---|---|
| OOM, no traceback, process vanishes | `--num-batch-threads 12` + 25k val/test peak >22 GB | threads `2`, reduce `--validation-samples`/`--testing-*-samples` |
| `ValueError: Invalid architecture: False` | click default-resolution bug | pass `--perceptron` explicitly |
| `ModuleNotFoundError: tokenizers` (×N per batch) | missing dep for `BERTTokenizer` | `uv pip install tokenizers` |
| `ImportError: Phonemizer ... required` | missing dep | `uv pip install phonemizer` |
| `numpy.compat` import error | numpy 2.x removed it | keep the `numpy_util.py` shim |
| `convert.py` → `No module named 'heybuddy'` | CWD not on sys.path | `PYTHONPATH=. python tools/convert.py` |
| `exports/heybuddy/*.pt` are 132 bytes | git-LFS pointers, not models | trained models land in `checkpoints/` |
| `unzip: command not found` | not installed | use Python `zipfile`; skip `__MACOSX/`/`._*` |
| Training dies on window reload | not in tmux | always `tmux new -s retrain` |

## Caveats carried from eval
- The data.zip test set is **synthetic** (ElevenLabs) — optimistic vs. real clinicians.
- Per-clip FPR ≠ FP/hour; `--target-false-positive-rate` targets the per-hour notion during training,
  but validate separately on continuous audio later.
