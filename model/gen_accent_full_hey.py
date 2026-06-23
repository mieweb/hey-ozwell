#!/usr/bin/env python3
"""
Full accent-diverse positive build (Google TTS), with realistic audio AUGMENTATION.
- ALL Google voices per accent (en-IN/GB/AU/US, all families), disjoint TRAIN/TEST voice split.
- Each TRAIN base clip -> K augmented variants (pitch shift + speed + real background-speech noise
  sampled from People's Speech) -> embed. Augmentation = realistic multiplier past Google's voice
  ceiling, staying on the speech manifold (NOT SMOTE/duplication).
- TEST clips (held-out voices) -> wavs in /tmp/eval/accent/<lc>/ for honest per-accent recall.
Usage: gen_accent_full.py [AUG_PER_CLIP] [VOICE_LIMIT_PER_ACCENT]   (defaults 12, none)
"""
import os, sys, json, base64, io, glob, time
import numpy as np, soundfile as sf, torch, torchaudio
from concurrent.futures import ThreadPoolExecutor
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heybuddy.embeddings import get_speech_embeddings

KEY = open(os.path.expanduser("~/.tts_api_key")).read().strip()
PHRASE = "hey ozwell"; SR = 16000; SEG = int(SR * 1.44)
AUG = int(sys.argv[1]) if len(sys.argv) > 1 else 12
VLIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0 = all voices
ACCENTS = ["en-IN", "en-GB", "en-AU", "en-US"]
NONCHIRP_RATES = [0.85, 1.0, 1.15, 1.30]                # Chirp voices: rate 1.0 only
NOISE = sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))  # real speech-babble backgrounds
rng = np.random.default_rng(0)

def get_voices(lc):
    r = json.load(urllib.request.urlopen(f"https://texttospeech.googleapis.com/v1/voices?key={KEY}&languageCode={lc}", timeout=30))
    return sorted(v["name"] for v in r["voices"] if lc in v["languageCodes"])

def synth(voice, lc, rate):
    body = {"input": {"text": PHRASE}, "voice": {"languageCode": lc, "name": voice},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": SR, "speakingRate": rate}}
    data = json.dumps(body).encode()
    for k in range(5):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                f"https://texttospeech.googleapis.com/v1/text:synthesize?key={KEY}",
                data=data, headers={"Content-Type": "application/json"}), timeout=45))
            a, _ = sf.read(io.BytesIO(base64.b64decode(r["audioContent"])), dtype="float32")
            return a if a.ndim == 1 else a.mean(1)
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None

def fit(a):
    a = a[:SEG] if len(a) >= SEG else np.concatenate([np.zeros((SEG-len(a))//2,"float32"), a, np.zeros(SEG-len(a)-(SEG-len(a))//2,"float32")])
    return a.astype("float32")

def augment(a):
    """one realistic variant: pitch shift + speed + background speech-babble."""
    x = torch.tensor(a)
    n_steps = int(rng.integers(-3, 4))
    if n_steps != 0:
        x = torchaudio.functional.pitch_shift(x, SR, n_steps)
    rate = float(rng.uniform(0.9, 1.12))
    if abs(rate - 1) > 0.01:
        L = int(len(x) / rate)
        x = torch.nn.functional.interpolate(x[None, None], size=L, mode="linear", align_corners=False)[0, 0]
    y = x.numpy().astype("float32")
    if NOISE and rng.random() < 0.8:                       # mix in real speech-babble at random SNR
        nz, _ = sf.read(NOISE[rng.integers(len(NOISE))], dtype="float32")
        if nz.ndim > 1: nz = nz.mean(1)
        if len(nz) < len(y): nz = np.tile(nz, len(y)//len(nz)+1)
        nz = nz[:len(y)]
        snr = rng.uniform(5, 20)
        ps, pn = (y**2).mean()+1e-9, (nz**2).mean()+1e-9
        y = y + nz * np.sqrt(ps/pn/(10**(snr/10)))
    return fit(y)

# ---- 1) gather voices, disjoint train/test split ----
train_jobs, test_jobs = [], []
for lc in ACCENTS:
    vs = get_voices(lc)
    if VLIMIT: vs = vs[:VLIMIT]
    n_test = max(6, len(vs)//5)
    test_vs, train_vs = vs[:n_test], vs[n_test:]
    for v in train_vs:
        rates = NONCHIRP_RATES if ("Chirp" not in v) else [1.0]
        for rt in rates: train_jobs.append((v, lc, rt))
    for v in test_vs:
        test_jobs.append((v, lc, 1.0))
    print(f"{lc}: {len(train_vs)} train / {len(test_vs)} test voices", flush=True)

print(f"synthesizing {len(train_jobs)} base train + {len(test_jobs)} test clips...", flush=True)
with ThreadPoolExecutor(max_workers=2) as ex:
    base_train = list(ex.map(lambda j: (j, synth(*j)), train_jobs))
    base_test = list(ex.map(lambda j: (j, synth(*j)), test_jobs))

# ---- 2) test -> wavs ----
nt = 0
for (v, lc, rt), a in base_test:
    if a is None: continue
    d = f"/tmp/eval/hey_accent/{lc}"; os.makedirs(d, exist_ok=True)
    sf.write(f"{d}/{v}.wav", a, SR); nt += 1

# ---- 3) train -> augment x AUG -> embed ----
bases = [a for _, a in base_train if a is not None]
print(f"got {len(bases)} base train clips; augmenting x{AUG} -> embedding...", flush=True)
auds = []
for a in bases:
    auds.append(fit(a))                         # the clean original
    for _ in range(AUG):
        auds.append(augment(a))
emb_model = get_speech_embeddings(device_id=0)
out = []
for i in range(0, len(auds), 256):
    e = emb_model(auds[i:i+256], spectrogram_batch_size=32, embedding_batch_size=32, remove_nan=False)
    out.append(np.asarray(e, dtype="float32"))
    if i % 2048 == 0: print(f"  embed {i}/{len(auds)}", flush=True)
emb = np.concatenate(out); emb = emb[~np.isnan(emb).any(axis=(1,2))]
np.save("heybuddy/precalculated/hey_accent_pos.npy", emb)
print(f"DONE TRAIN: accent_pos.npy {emb.shape} mean={float(emb[:,:16,:].mean()):.2f} | TEST: {nt} held-out clips", flush=True)
