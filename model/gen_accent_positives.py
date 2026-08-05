#!/usr/bin/env python3
"""
Track 2: generate REAL accent-diverse positives for "ozwell i'm done" via Google Cloud TTS
(en-IN Indian, en-GB, en-AU, en-US), the accents our American-only Piper engine can't make.

Voices are split TRAIN vs HELD-OUT TEST (disjoint) per accent — so we can train on some
voices of an accent and honestly measure recall on *different* voices of it.
  - TRAIN -> embedded [N,16,96] in heybuddy/precalculated/accent_pos.npy (inject into positives)
  - TEST  -> wavs in /tmp/eval/accent/<accent>/ (per-accent recall measurement)
"""
import os, json, base64, io, sys, glob
import numpy as np, soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heybuddy.embeddings import get_speech_embeddings

KEY = open(os.path.expanduser("~/.tts_api_key")).read().strip()
PHRASE = "ozwell I'm done"
SR = 16000; SEG = int(SR * 1.44)
ACCENTS = {"en-IN": 8, "en-GB": 6, "en-AU": 6, "en-US": 4}   # accent -> rate-variations count
RATES = [0.80, 0.88, 0.96, 1.04, 1.12, 1.20, 1.30, 1.40]
US_VOICE_CAP = 20   # we already have US (libritts); cap it to spend budget on IN/GB/AU

def voices_for(lc):
    r = json.load(urllib.request.urlopen(f"https://texttospeech.googleapis.com/v1/voices?key={KEY}&languageCode={lc}", timeout=30))
    return sorted(v["name"] for v in r["voices"] if lc in v["languageCodes"])

import time
def synth(voice, lc, rate):
    body = {"input": {"text": PHRASE}, "voice": {"languageCode": lc, "name": voice},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": SR, "speakingRate": rate}}
    data = json.dumps(body).encode()
    for attempt in range(5):                      # retry on 429 / transient errors
        req = urllib.request.Request(f"https://texttospeech.googleapis.com/v1/text:synthesize?key={KEY}",
                                     data=data, headers={"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=45))
            a, _ = sf.read(io.BytesIO(base64.b64decode(r["audioContent"])), dtype="float32")
            return a if a.ndim == 1 else a.mean(1)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None

def fit(a):
    a = a[:SEG] if len(a) >= SEG else np.concatenate([np.zeros((SEG-len(a))//2,"float32"), a, np.zeros(SEG-len(a)-(SEG-len(a))//2,"float32")])
    return a.astype("float32")

train_jobs, test_jobs = [], []
for lc, nrates in ACCENTS.items():
    vs = voices_for(lc)
    if lc == "en-US": vs = vs[:US_VOICE_CAP]
    n_test = max(6, len(vs)//5)
    test_vs, train_vs = vs[:n_test], vs[n_test:]
    for v in train_vs:
        for rate in RATES[:nrates]:
            train_jobs.append((v, lc, rate))
    for v in test_vs:
        for rate in (0.95, 1.15):
            test_jobs.append((v, lc, rate))
    print(f"{lc}: {len(train_vs)} train / {len(test_vs)} test voices")

print(f"generating {len(train_jobs)} train + {len(test_jobs)} test clips via Google TTS...")
def run(job): v, lc, rate = job; return (job, synth(v, lc, rate))
with ThreadPoolExecutor(max_workers=6) as ex:
    train_res = list(ex.map(run, train_jobs))
    test_res = list(ex.map(run, test_jobs))

# TEST -> wavs per accent
for (v, lc, rate), a in test_res:
    if a is None: continue
    d = f"/tmp/eval/accent/{lc}"; os.makedirs(d, exist_ok=True)
    sf.write(f"{d}/{v}_{rate}.wav", a, SR)
ntest = sum(1 for _, a in test_res if a is not None)

# TRAIN -> embed -> save
auds = [fit(a) for (_, a) in train_res if a is not None]
emb_model = get_speech_embeddings(device_id=0)
out = []
for i in range(0, len(auds), 256):
    e = emb_model(auds[i:i+256], spectrogram_batch_size=32, embedding_batch_size=32, remove_nan=False)
    out.append(np.asarray(e, dtype="float32"))
emb = np.concatenate(out); emb = emb[~np.isnan(emb).any(axis=(1,2))]
np.save("heybuddy/precalculated/accent_pos.npy", emb)
print(f"TRAIN: saved accent_pos.npy {emb.shape} mean(frames)={float(emb[:,:16,:].mean()):.2f}")
print(f"TEST:  {ntest} clips -> /tmp/eval/accent/<accent>/")
