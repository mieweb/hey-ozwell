#!/usr/bin/env python3
"""Generate PHONETIC/CONFUSABLE hard negatives (FP hardening) via local Piper TTS.
Output: [N,16,96] embeddings -> heybuddy/precalculated/confusable_negs.npy (same format as positives).

IMPORTANT — these are DISTINCT near-miss words, NOT the phrase or its core word. We deliberately
EXCLUDE anything too close to the wake phrase (e.g. "ozwell" alone, "all is well", bare "hey"/"well"),
because training those as negatives would hurt recall (the model would learn to reject the real phrase).
Recall is re-verified after the retrain to catch any over-suppression."""
import sys, glob, os
import numpy as np
from scipy.signal import resample_poly
from math import gcd
from huggingface_hub import hf_hub_download
from piper import PiperVoice, SynthesisConfig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heybuddy.embeddings import get_speech_embeddings

SR = 16000; SEG = int(SR * 1.44)
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
OUT = sys.argv[2] if len(sys.argv) > 2 else "heybuddy/precalculated/confusable_negs.npy"

# SAFE confusables. (Add real mined triggers you log to these lists over time.)
# DISTINCT words/phrases — safe to train in isolation (clearly different from the wake phrase):
PHRASES = [
    "roswell", "oswald", "nozzle", "gospel", "hostel",
    "well done", "almost done", "are we done", "i'm gone", "all done",
    "yo bro", "you know", "let's go", "hey there", "okay", "hold on", "what's up", "no way",
]
# NEAR-HOMOPHONES ("...as well" / "oh well") — too close to "ozwell" to train BARE (would hurt recall),
# so train them IN SENTENCE CONTEXT: matches how they actually false-fire in conversation, and the
# surrounding words give the model context to keep them distinct from the deliberate wake phrase.
PHRASES += [
    "i'd like that as well", "me as well", "that works as well", "i think so as well",
    "yeah that one as well", "oh well it happens", "oh well never mind",
]
# EXCLUDED entirely (too close -> would hurt recall): bare "as well", "ozwell", "hey", "well", "all is well"

VOICES = [
    ("en/en_GB/vctk/medium/en_GB-vctk-medium", True),
    ("en/en_US/ryan/high/en_US-ryan-high", False),
    ("en/en_US/lessac/medium/en_US-lessac-medium", False),
    ("en/en_US/amy/medium/en_US-amy-medium", False),
    ("en/en_US/joe/medium/en_US-joe-medium", False),
]
def load_voice(stem):
    for ext in (".onnx", ".onnx.json"):
        p = hf_hub_download("rhasspy/piper-voices", stem + ext)
    return PiperVoice.load(glob.glob(f"{os.path.dirname(p)}/*.onnx")[0])

rng = np.random.default_rng(0)
voices = []
for stem, multi in VOICES:
    v = load_voice(stem); voices.append((v, v.config.num_speakers if multi else 1))
print(f"loaded {len(voices)} voices; {len(PHRASES)} confusable phrases")

def gen_one():
    phrase = PHRASES[rng.integers(len(PHRASES))]
    v, n = voices[rng.integers(len(voices))]
    cfg = SynthesisConfig(
        speaker_id=int(rng.integers(n)) if n > 1 else None,
        length_scale=float(rng.uniform(0.85, 1.3)),
        noise_scale=float(rng.uniform(0.5, 0.9)),
        noise_w_scale=float(rng.uniform(0.6, 1.0)),
    )
    chunks = list(v.synthesize(phrase, syn_config=cfg))
    a = np.concatenate([c.audio_float_array for c in chunks]).astype("float32")
    sr0 = chunks[0].sample_rate
    g = gcd(sr0, SR); a = resample_poly(a, SR // g, sr0 // g).astype("float32")
    if len(a) >= SEG: a = a[:SEG]
    else:
        pad = SEG - len(a); a = np.concatenate([np.zeros(pad//2,"float32"), a, np.zeros(pad-pad//2,"float32")])
    return a

emb_model = get_speech_embeddings(device_id=0)
buf, BATCH, out_chunks = [], 256, []
for i in range(COUNT):
    buf.append(gen_one())
    if len(buf) == BATCH or i == COUNT - 1:
        e = emb_model(buf, spectrogram_batch_size=32, embedding_batch_size=32, remove_nan=False)
        out_chunks.append(np.asarray(e, dtype="float32")); buf = []
        if (i+1) % 1024 < BATCH: print(f"  {i+1}/{COUNT}")
emb = np.concatenate(out_chunks); emb = emb[~np.isnan(emb).any(axis=(1,2))]
np.save(OUT, emb)
print(f"saved {OUT} shape={emb.shape}")
