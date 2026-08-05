#!/usr/bin/env python3
"""
Azure Speech accent-diverse positives (second engine) for "ozwell i'm done".
Adds DISTINCT accented voices Google didn't have (incl. en-NG Nigerian, en-IE Irish, en-ZA).
Same structure as gen_accent_full.py: disjoint train/test voices, prosody-rate variation,
realistic audio augmentation, embed -> azure_accent_pos.npy ; test wavs -> /tmp/eval/accent/<lc>/azure_*.wav
"""
import os, sys, json, glob, io, time
import numpy as np, soundfile as sf, torch, torchaudio
from concurrent.futures import ThreadPoolExecutor
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heybuddy.embeddings import get_speech_embeddings

KEY = open(os.path.expanduser("~/.azure_speech_key")).read().strip()
REGION = "eastus2"
PHRASE = "hey ozwell"; SR = 16000; SEG = int(SR * 1.44)
AUG = int(sys.argv[1]) if len(sys.argv) > 1 else 15
ACCENTS = ["en-IN", "en-GB", "en-AU", "en-US", "en-NG", "en-IE", "en-ZA"]
RATES = [-20, 0, 18, 33]
NOISE = sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))
rng = np.random.default_rng(1)
EP = f"https://{REGION}.tts.speech.microsoft.com/cognitiveservices/v1"

def all_voices():
    req = urllib.request.Request(f"https://{REGION}.tts.speech.microsoft.com/cognitiveservices/voices/list",
                                 headers={"Ocp-Apim-Subscription-Key": KEY})
    return json.load(urllib.request.urlopen(req, timeout=30))

def synth(voice, lc, rate):
    pros = PHRASE if rate == 0 else f"<prosody rate='{rate:+d}%'>{PHRASE}</prosody>"
    ssml = f"<speak version='1.0' xml:lang='{lc}'><voice name='{voice}'>{pros}</voice></speak>"
    for k in range(5):
        try:
            req = urllib.request.Request(EP, data=ssml.encode(), headers={
                "Ocp-Apim-Subscription-Key": KEY, "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm"})
            a, _ = sf.read(io.BytesIO(urllib.request.urlopen(req, timeout=45).read()), dtype="float32")
            return a if a.ndim == 1 else a.mean(1)
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None

def fit(a):
    a = a[:SEG] if len(a) >= SEG else np.concatenate([np.zeros((SEG-len(a))//2,"float32"), a, np.zeros(SEG-len(a)-(SEG-len(a))//2,"float32")])
    return a.astype("float32")

def augment(a):
    x = torch.tensor(a)
    ns = int(rng.integers(-3, 4))
    if ns != 0: x = torchaudio.functional.pitch_shift(x, SR, ns)
    rate = float(rng.uniform(0.9, 1.12))
    if abs(rate-1) > 0.01:
        x = torch.nn.functional.interpolate(x[None,None], size=int(len(x)/rate), mode="linear", align_corners=False)[0,0]
    y = x.numpy().astype("float32")
    if NOISE and rng.random() < 0.8:
        nz,_ = sf.read(NOISE[rng.integers(len(NOISE))], dtype="float32")
        if nz.ndim > 1: nz = nz.mean(1)
        if len(nz) < len(y): nz = np.tile(nz, len(y)//len(nz)+1)
        nz = nz[:len(y)]; snr = rng.uniform(5,20)
        y = y + nz*np.sqrt(((y**2).mean()+1e-9)/((nz**2).mean()+1e-9)/(10**(snr/10)))
    return fit(y)

# voices by accent
byacc = {}
for v in all_voices():
    lc = v["Locale"]
    if lc in ACCENTS: byacc.setdefault(lc, []).append(v["ShortName"])
train_jobs, test_jobs = [], []
for lc in ACCENTS:
    vs = sorted(byacc.get(lc, []))
    if not vs: continue
    n_test = max(1, len(vs)//5)
    test_vs, train_vs = vs[:n_test], vs[n_test:]
    if lc == "en-US": train_vs = train_vs[:20]
    for v in train_vs:
        for rt in RATES: train_jobs.append((v, lc, rt))
    for v in test_vs: test_jobs.append((v, lc, 0))
    print(f"{lc}: {len(train_vs)} train / {len(test_vs)} test voices", flush=True)

print(f"Azure: synth {len(train_jobs)} base train + {len(test_jobs)} test...", flush=True)
with ThreadPoolExecutor(max_workers=3) as ex:
    base_train = list(ex.map(lambda j: (j, synth(*j)), train_jobs))
    base_test = list(ex.map(lambda j: (j, synth(*j)), test_jobs))

nt = 0
for (v, lc, rt), a in base_test:
    if a is None: continue
    d = f"/tmp/eval/hey_accent/{lc}"; os.makedirs(d, exist_ok=True)
    sf.write(f"{d}/azure_{v}.wav", a, SR); nt += 1

bases = [a for _, a in base_train if a is not None]
print(f"got {len(bases)} base; augment x{AUG} + embed...", flush=True)
auds = []
for a in bases:
    auds.append(fit(a))
    for _ in range(AUG): auds.append(augment(a))
emb_model = get_speech_embeddings(device_id=0)
out = []
for i in range(0, len(auds), 256):
    out.append(np.asarray(emb_model(auds[i:i+256], spectrogram_batch_size=32, embedding_batch_size=32, remove_nan=False), dtype="float32"))
emb = np.concatenate(out); emb = emb[~np.isnan(emb).any(axis=(1,2))]
np.save("heybuddy/precalculated/hey_azure_accent_pos.npy", emb)
print(f"DONE: azure_accent_pos.npy {emb.shape} mean={float(emb[:,:16,:].mean()):.2f} | TEST {nt} clips", flush=True)
