#!/usr/bin/env python3
"""Generate "ozwell + WRONG ending" hard negatives for ozwell-i'm-done (e.g. "ozwell deez nuts").
Teaches the model to require the actual ending ("...I'm done"), not just the "ozwell" sound.
TRAIN phrases -> ozwell_wrong_negs.npy ; held-out TEST phrases/voices -> /tmp/eval/ozwell_wrong/ wavs
(so we can honestly measure the rejection rate). NOTE: these share "ozwell" -> recall risk; the
retrain re-verifies recall and we only adopt if it holds."""
import sys, glob, os
import numpy as np
from scipy.signal import resample_poly
from math import gcd
from huggingface_hub import hf_hub_download
from piper import PiperVoice, SynthesisConfig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heybuddy.embeddings import get_speech_embeddings

SR = 16000; SEG = int(SR * 1.44)
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 6000

# "ozwell" + a wrong ending (NOT "I'm done"). Train on these:
TRAIN_PHRASES = ["ozwell hold on", "ozwell what's up", "ozwell never mind", "ozwell are you there",
                 "ozwell let's go", "ozwell over here", "ozwell come here", "ozwell not now",
                 "ozwell one second", "ozwell sounds good"]
# Held out for honest rejection test (different phrases + we use different voices):
TEST_PHRASES  = ["ozwell deez nuts", "ozwell good morning"]

VOICES = [("en/en_GB/vctk/medium/en_GB-vctk-medium", True),
          ("en/en_US/ryan/high/en_US-ryan-high", False),
          ("en/en_US/lessac/medium/en_US-lessac-medium", False),
          ("en/en_US/amy/medium/en_US-amy-medium", False),
          ("en/en_US/joe/medium/en_US-joe-medium", False)]
def load_voice(stem):
    for ext in (".onnx", ".onnx.json"): p = hf_hub_download("rhasspy/piper-voices", stem + ext)
    return PiperVoice.load(glob.glob(f"{os.path.dirname(p)}/*.onnx")[0])
rng = np.random.default_rng(0)
voices = []
for stem, multi in VOICES:
    v = load_voice(stem); voices.append((v, v.config.num_speakers if multi else 1))

import soundfile as sf
def synth(phrase, vi=None, sid=None):
    v, n = voices[vi if vi is not None else rng.integers(len(voices))]
    cfg = SynthesisConfig(speaker_id=(sid if sid is not None else (int(rng.integers(n)) if n>1 else None)),
                          length_scale=float(rng.uniform(0.85,1.3)), noise_scale=float(rng.uniform(0.5,0.9)),
                          noise_w_scale=float(rng.uniform(0.6,1.0)))
    chunks = list(v.synthesize(phrase, syn_config=cfg))
    a = np.concatenate([c.audio_float_array for c in chunks]).astype("float32"); sr0 = chunks[0].sample_rate
    g = gcd(sr0, SR); a = resample_poly(a, SR//g, sr0//g).astype("float32")
    if len(a) >= SEG: a = a[:SEG]
    else: pad = SEG-len(a); a = np.concatenate([np.zeros(pad//2,"float32"), a, np.zeros(pad-pad//2,"float32")])
    return a

emb_model = get_speech_embeddings(device_id=0)

# --- TRAIN negatives ---
buf, BATCH, out = [], 256, []
for i in range(COUNT):
    buf.append(synth(TRAIN_PHRASES[rng.integers(len(TRAIN_PHRASES))]))
    if len(buf)==BATCH or i==COUNT-1:
        out.append(np.asarray(emb_model(buf, spectrogram_batch_size=32, embedding_batch_size=32, remove_nan=False), dtype="float32")); buf=[]
emb = np.concatenate(out); emb = emb[~np.isnan(emb).any(axis=(1,2))]
np.save("heybuddy/precalculated/ozwell_wrong_negs.npy", emb)
print(f"train negs: ozwell_wrong_negs.npy {emb.shape}")

# --- held-out TEST wavs (every voice/speaker for the test phrases) ---
d = "/tmp/eval/ozwell_wrong"; os.makedirs(d, exist_ok=True)
nt = 0
for p in TEST_PHRASES:
    for vi,(v,n) in enumerate(voices):
        for sid in ([None] if n==1 else list(range(0, n, max(1,n//6)))[:6]):
            a = synth(p, vi=vi, sid=sid)
            sf.write(f"{d}/{p.replace(' ','_')}_{vi}_{sid}.wav", a, SR); nt += 1
print(f"test wavs: {nt} in {d}")
