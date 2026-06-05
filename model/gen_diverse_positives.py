#!/usr/bin/env python3
"""
Track 1: generate accent/voice-DIVERSE synthetic positives for "ozwell i'm done" and
inject them into the cached positive set, to test whether positive diversity lifts the
held-out (ElevenLabs) recall off 64%.

Sources (free, local Piper): en_GB-vctk (109 UK speakers, varied British accents) +
a few extra en_US voices. Varied length_scale (speed) + noise_scale (voice variation).
Output: [N,16,96] embeddings (peak-normalized, 1.44s segments) — same format as the
cached Piper positives, which heybuddy reuses if the cache has >= requested samples.
"""
import sys, glob, os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from huggingface_hub import hf_hub_download
from piper import PiperVoice, SynthesisConfig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heybuddy.embeddings import get_speech_embeddings

PHRASE = "ozwell I'm done"
SR = 16000
SEG = int(SR * 1.44)  # 23040
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
OUT = sys.argv[2] if len(sys.argv) > 2 else "heybuddy/precalculated/diverse_pos.npy"

VOICES = [  # (repo path stem, is_multispeaker)
    ("en/en_GB/vctk/medium/en_GB-vctk-medium", True),       # 109 UK speakers
    ("en/en_US/ryan/high/en_US-ryan-high", False),
    ("en/en_US/lessac/medium/en_US-lessac-medium", False),
    ("en/en_US/amy/medium/en_US-amy-medium", False),
    ("en/en_US/hfc_female/medium/en_US-hfc_female-medium", False),
    ("en/en_US/joe/medium/en_US-joe-medium", False),
]

def load_voice(stem):
    for ext in (".onnx", ".onnx.json"):
        p = hf_hub_download("rhasspy/piper-voices", stem + ext)
    return PiperVoice.load(glob.glob(f"{os.path.dirname(p)}/*.onnx")[0])

rng = np.random.default_rng(0)
voices = []
for stem, multi in VOICES:
    v = load_voice(stem)
    n = v.config.num_speakers if multi else 1
    voices.append((v, n))
total_speakers = sum(n for _, n in voices)
print(f"loaded {len(voices)} voice models, {total_speakers} total speakers")

def gen_one():
    v, n = voices[rng.integers(len(voices))]
    cfg = SynthesisConfig(
        speaker_id=int(rng.integers(n)) if n > 1 else None,
        length_scale=float(rng.uniform(0.85, 1.3)),
        noise_scale=float(rng.uniform(0.5, 0.9)),
        noise_w_scale=float(rng.uniform(0.6, 1.0)),
    )
    chunks = list(v.synthesize(PHRASE, syn_config=cfg))
    a = np.concatenate([c.audio_float_array for c in chunks]).astype("float32")
    sr0 = chunks[0].sample_rate
    g = gcd(sr0, SR); a = resample_poly(a, SR // g, sr0 // g).astype("float32")
    if len(a) >= SEG:
        a = a[:SEG]
    else:
        pad = SEG - len(a); a = np.concatenate([np.zeros(pad // 2, "float32"), a, np.zeros(pad - pad // 2, "float32")])
    return a

emb_model = get_speech_embeddings(device_id=0)
buf, BATCH = [], 256
out_chunks = []
for i in range(COUNT):
    buf.append(gen_one())
    if len(buf) == BATCH or i == COUNT - 1:
        e = emb_model(buf, spectrogram_batch_size=32, embedding_batch_size=32, remove_nan=False)  # [b,16,96]
        out_chunks.append(np.asarray(e, dtype="float32"))
        buf = []
        if (i + 1) % 1024 < BATCH:
            print(f"  {i+1}/{COUNT}")
emb = np.concatenate(out_chunks)
emb = emb[~np.isnan(emb).any(axis=(1, 2))]
np.save(OUT, emb)
print(f"saved {OUT} shape={emb.shape} mean(frames)={float(emb[:, :16, :].mean()):.2f}")
