#!/usr/bin/env python3
"""
Build the SPEAKER-VERIFICATION cohort for AS-norm (score normalization on the WHO gate).

A cohort = a crowd of OTHER voices' TitaNet embeddings. At verify time the live clip (and the
enrolled centroid) are scored against this crowd; normalizing by the crowd's mean/spread cancels
the channel offset, so the WHO threshold stays stable across rooms.

PARITY: embeddings come from the SAME library + model the browser runtime uses (sherpa-onnx +
nemo_en_titanet_small.onnx), so cohort scores are comparable to live scores. We cut ~1.2s segments
(close to a wake-utterance length) from real multi-speaker corpora and embed each.
"""
import sherpa_onnx, soundfile as sf, numpy as np, glob, json, os, sys

MODEL = "/home/jlocala/sherpa-onnx/wasm/speaker-diarization/assets/nemo_en_titanet_small.onnx"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sv-cohort.json"
SR = 16000
SEG = int(1.2 * SR)
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 250
DIRS = ["/tmp/fphour/thirdparty_ami", "/tmp/fphour/voxpopuli_test",
        "/tmp/fphour/peoples_1h", "/tmp/fphour/peoples_test"]

cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=MODEL, num_threads=2, debug=False, provider="cpu")
ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
print("embedding dim:", ext.dim)


def embed(samples, sr):
    s = ext.create_stream()
    s.accept_waveform(sr, samples)
    s.input_finished()
    e = np.array(ext.compute(s), dtype="float64")
    n = np.linalg.norm(e)
    return (e / n) if n > 0 else e   # L2-normalize, matching the runtime


files = []
for d in DIRS:
    files += sorted(glob.glob(os.path.join(d, "*.wav")))
rng = np.random.default_rng(0)
rng.shuffle(files)
print(f"{len(files)} source clips across {len(DIRS)} corpora")

cohort = []
for f in files:
    a, sr = sf.read(f)
    if a.ndim > 1:
        a = a.mean(1)
    a = a.astype("float32")
    got = 0
    for st in range(0, max(0, len(a) - SEG), SEG):
        seg = a[st:st + SEG]
        if np.sqrt(np.mean(seg ** 2)) < 0.01:   # skip near-silence
            continue
        cohort.append([round(float(x), 5) for x in embed(seg, sr)])
        got += 1
        if got >= 2:    # at most 2 segments per file -> more distinct voices
            break
    if len(cohort) >= TARGET:
        break

json.dump(cohort, open(OUT, "w"))
arr = np.array(cohort)
# sanity: cohort-to-cohort cosine spread (should be low mean, not all ~1)
import itertools
sims = [float(np.dot(arr[i], arr[j])) for i, j in itertools.islice(itertools.combinations(range(len(arr)), 2), 2000)]
print(f"cohort {arr.shape} -> {OUT}")
print(f"pairwise cosine among cohort: mean {np.mean(sims):.2f} (distinct voices => not near 1.0)")
