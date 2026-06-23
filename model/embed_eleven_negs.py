#!/usr/bin/env python3
"""
Embed the 556 ElevenLabs conversational train negatives into heybuddy's negative
format ([N,17,96]: 16-frame speech embedding + 1 BERT-token row), exactly matching
how `heybuddy extract` builds negs_libri/negs_pk, so they can be used as a
--training-dataset. Transcript is empty (these are generic negatives, not the phrase),
so the token row won't collide with the wake word during negative-sampling exclusion.
"""
import glob, os, sys
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heybuddy.dataset.precalculated import PrecalculatedLabeledTrainingDatasetGenerator

SRC = "/tmp/train/done_neg_eleven"
OUT = "heybuddy/precalculated/negs_eleven.npy"
SR = 16000

gen = PrecalculatedLabeledTrainingDatasetGenerator(
    dataset_path="unused", transcript_key="transcript", device_id=0, sample_rate=SR,
)
SEG = gen.samples_per_batch  # 23040 = 1.44s
print(f"segment length = {SEG} samples ({SEG/SR:.2f}s)")

# Build 1.44s segments (right-padded) from every wav — same chunking as the generator loop
batch = []
for p in sorted(glob.glob(os.path.join(SRC, "*.wav"))):
    a, sr = sf.read(p, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR); a = resample_poly(a, SR // g, sr // g).astype("float32")
    for i in range(0, max(len(a), 1), SEG):
        seg = a[i:i + SEG]
        if seg.shape[0] < SEG:
            seg = np.concatenate([seg, np.zeros(SEG - seg.shape[0], "float32")])
        batch.append((seg.astype("float32"), {"transcript": ""}))

print(f"{len(batch)} segments from {len(glob.glob(os.path.join(SRC,'*.wav')))} wavs; embedding...")
emb = gen.speech_embeddings([a for a, s in batch], spectrogram_batch_size=32,
                            embedding_batch_size=32, remove_nan=False)  # [n,16,96]
labeled = gen.label_embeddings(embeddings=emb, batch=batch)            # [n,17,96]
labeled = labeled[~np.isnan(labeled).any(axis=(1, 2))]
np.save(OUT, labeled.astype("float32"))
print(f"saved {OUT} shape={labeled.shape} mean(frames)={float(labeled[:, :16, :].mean()):.2f}")
