# Speaker-Verification Gate on the Wake Word — Spike + Build Runbook

**Status: feasibility (Phase 1) DONE → GO. Phase 2 (browser delivery) is a build task, not yet started.**
_If this goes up as a PR for visibility: **DON'T MERGE** — it's a spike writeup + runbook, no shippable code yet._

## What this feature is

When "hey ozwell" fires, verify the speaker is the **enrolled doctor** before driving Ozwell.
Anyone can trigger the wake word; only the doctor's voice should act on it. Fully on-device
(PHI — no audio leaves the browser). This is **1:1 speaker verification**, NOT transcription and
NOT diarization: enroll the doctor once → on each wake, cosine-compare the live voice embedding to
the enrolled embedding → gate `driveOzwell`.

The wake utterance is already in hand: `hey-buddy.js` captures it via `onRecording(audioSamples)`
(16 kHz mono `Float32Array`).

---

## Phase 1 — feasibility + margin (DONE, June 11)

Measured entirely in Python on the Mac, inference-only (same pattern as [`model/eval/`](../../../model/eval/README.md)),
using sherpa-onnx's own `SpeakerEmbeddingExtractor` — so the **internal fbank is real** and the
hand-computed-feature scale-mismatch bug class is off the table. Harness: `/tmp/hey-eval/sv_margin.py`.

Data: 500 synthetic "hey ozwell" clips across 38 ElevenLabs voices (`model/data/data.zip`,
`hey-ozwell/{train,test}/positive`). Filename `NNNN_Voice.wav` → the voice name is the speaker
label, so **genuine** trials = same-voice pairs, **impostor** = different-voice pairs.

### Two findings

**1. `pip install sherpa-onnx` works on macOS and exposes the extractor standalone.**
v1.13.2, `sherpa_onnx.SpeakerEmbeddingExtractor` present. The whole margin test runs without any
browser/WASM build.

**2. Speaker verification on the ~1s wake phrase is feasible — and the best model is NeMo TitaNet-small, not WeSpeaker.**
"enroll" = the realistic flow (enroll = mean of 3 clips → score held-out same-voice vs every other
voice) on the **loudest ~1 s window**:

| Model (sherpa-onnx) | EER (raw ~1s pairs) | EER (enroll-centroid, ~1s) | genuine vs impostor | margin* |
|---|---|---|---|---|
| **`nemo_en_titanet_small.onnx`** | **3.3 %** | **1.3 %** | 0.66 vs 0.21 | **+0.30** |
| `wespeaker_en_voxceleb_resnet34_LM.onnx` _(original pick)_ | 14.3 % | 8.6 % | 0.80 vs 0.57 | +0.06 |
| `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` | 27.2 % | 20.5 % | 0.90 vs 0.76 | −0.05 (unusable) |

\*margin = genuine-mean minus impostor-95th-percentile (daylight between the distributions).

- Short-utterance robust: TitaNet barely degrades full→1s (3.04 % → 3.31 % EER) — a 1-second
  "hey ozwell" carries enough speaker identity.
- This does **not** reopen the sherpa-onnx vs WavLM decision — the extractor auto-detects the model
  type; it just picks the best in-inventory model for a ~1s phrase.
- **Decision: ship `nemo_en_titanet_small.onnx`** (38 MB). Threshold ≈ 0.45–0.55 (re-tune in browser).

### Caveat — don't over-quote 1.3 %
Synthetic TTS voices are cleaner and more self-consistent than a real person across sessions/mics,
and a real impostor *trying* to mimic the doctor isn't modeled. Real-world EER will be higher. But
TitaNet's margin (0.66 vs 0.21) has the headroom to absorb that; WeSpeaker's (tails touching) does
not. **Real-voice confirmation = the in-browser demo test**, not this synthetic number.

### Verdict: GO
The verification runs **once per wake event** on the recorded ~1s buffer (not in the per-frame hot
loop), so its latency is a one-shot, negligible.

---

## Phase 2 — get the extractor into the browser (the only remaining risk)

**Key finding:** there is **no prebuilt in-browser speaker-embedding WASM.** The `wasm/speaker-diarization`
browser build does NOT export the embedding-extractor symbols (its `CMakeLists.txt` `EXPORTED_FUNCTIONS`
lists only `...OfflineSpeakerDiarization...`). The C-API exists, so a browser build is possible but
needs a **custom Emscripten build**. The `sherpa-onnx` npm package is the native Node addon, not browser WASM.

**The shortcut:** diarization already links the embedding extractor internally (diarization = VAD +
segmentation + **speaker embedding** + clustering), so the symbols are already in the static lib.
The minimal change is to **add them to the diarization build's export list** — no new build target.

> Do this on the **os.mieweb.org Linux container** (VMID 117, user `jlocala`). Emscripten builds are
> far smoother on Linux than on this Mac, and it keeps the Mac clean.

### Steps (on the container)

```bash
# 1. Emscripten toolchain
git clone https://github.com/emscripten-core/emsdk.git && cd emsdk
./emsdk install latest && ./emsdk activate latest && source ./emsdk_env.sh
cd ..

# 2. sherpa-onnx source (separate from the hey-ozwell clone)
git clone https://github.com/k2-fsa/sherpa-onnx && cd sherpa-onnx
```

**3. Edit `wasm/speaker-diarization/CMakeLists.txt`** — append these 9 names to the
`EXPORTED_FUNCTIONS` list (note the leading `_`; keep the existing diarization entries):

```
_SherpaOnnxCreateSpeakerEmbeddingExtractor,
_SherpaOnnxDestroySpeakerEmbeddingExtractor,
_SherpaOnnxCreateSpeakerEmbeddingExtractorStream,
_SherpaOnnxSpeakerEmbeddingExtractorAcceptWaveform,
_SherpaOnnxSpeakerEmbeddingExtractorInputFinished,
_SherpaOnnxSpeakerEmbeddingExtractorIsReady,
_SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding,
_SherpaOnnxSpeakerEmbeddingExtractorDimension,
_SherpaOnnxDestroyOnlineStream
```
`_malloc`, `_free`, and the `HEAPF32`/`HEAP32` views are already exported by this build (the
diarization wrapper uses them), so nothing else to add.

```bash
# 4. Build (the script sets the WASM + C-API flags; output -> install/bin/wasm/speaker-diarization)
./build-wasm-simd-speaker-diarization.sh
```

Artifacts land in `build-wasm-simd-speaker-diarization/install/bin/wasm/speaker-diarization/`
(`.wasm`, `.js`, `.data`). Copy the `.wasm`/`.js` glue into `prod/js/` and serve them (add a static
mount in `server.js` if needed). Drop `nemo_en_titanet_small.onnx` into `prod/js/models/`.

### Build-risk fallback
If the custom build hits a wall, the documented fallback is to run `nemo_en_titanet_small.onnx`
directly in the already-loaded `onnxruntime-web` with a **vetted JS log-mel fbank** — **but** that
reintroduces the exact feature-scale-mismatch bug class this project already hit, so it is the
fallback, not the plan.

---

## Phase 3 — JS wrapper + integration (after the build)

### Thin wrapper (model on `sherpa-onnx-speaker-diarization.js`)
The C-API call sequence per utterance:

```js
// extractor created once at load from the config struct {model, num_threads, debug, provider}.
// VERIFY the struct field order/offsets against sherpa-onnx/c-api/c-api.h before marshaling.
const stream = Module._SherpaOnnxCreateSpeakerEmbeddingExtractorStream(extractor);
const ptr = Module._malloc(samples.length * 4);
Module.HEAPF32.set(samples, ptr / 4);                              // 16k mono Float32
Module._SherpaOnnxSpeakerEmbeddingExtractorAcceptWaveform(extractor, stream, 16000, ptr, samples.length);
Module._SherpaOnnxSpeakerEmbeddingExtractorInputFinished(extractor, stream);
const dim = Module._SherpaOnnxSpeakerEmbeddingExtractorDimension(extractor);
const ep  = Module._SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(extractor, stream); // const float*
const emb = Module.HEAPF32.subarray(ep / 4, ep / 4 + dim).slice(); // copy out of heap
Module._free(ptr); Module._SherpaOnnxDestroyOnlineStream(stream);
// L2-normalize emb -> cosine == dot product
```

### Wiring into the existing voiceprint plumbing
Reuse the enroll→store-vectors→cosine→threshold scaffolding already built for the content voiceprint;
only the embedding *source* changes (speaker encoder instead of the wake-word embedding) and the gate
moves to fire-once-per-wake:

- **Enroll** ([`src/index.js`](../src/index.js) `runEnrollment`/`captureRep`): on each captured rep,
  feed the recorded ~1s buffer through the wrapper → store the speaker embedding in `localStorage`
  (vectors, not audio — same as today). The doctor's enrolled centroid = mean of the reps.
- **Gate** ([`hey-buddy.js`](../src/hey-buddy.js) `onRecording` / `driveOzwell` in `src/index.js`):
  when a wake word fires, embed the `onRecording` buffer, cosine vs the enrolled centroid, and only
  call `driveOzwell` if it clears the threshold (start ~0.45–0.55, tune live with the readout). On a
  miss: ignore — a non-doctor said the phrase.

Note this is a different axis from the current content-voiceprint *boost*: that one helps the doctor's
accented "hey ozwell" trigger; this one *blocks* a non-doctor's clean "hey ozwell". They compose —
content-voiceprint as a recall fallback, speaker-verification as the act/no-act gate.

---

## Pointers
- Spike harness + venv + models + clips: `/tmp/hey-eval/` (`sv_margin.py`, `sv-venv`, `sv-models/`, `sv-clips/`).
- Models release: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
- C-API: `sherpa-onnx/sherpa-onnx/c-api/c-api.h` (`SherpaOnnx...SpeakerEmbeddingExtractor...`).
- Diarization wasm wrapper to mirror: `sherpa-onnx/wasm/speaker-diarization/sherpa-onnx-speaker-diarization.js`.
