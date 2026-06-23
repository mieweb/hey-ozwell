#!/usr/bin/env python3
"""Balanced, deliberate embed of ozwell-done accent TRAIN clips.
- PER-ACCENT EQUAL TARGET (~TARGET positives/accent) so no accent dominates; augmentation fills
  the gap for voice-poor accents, capped at MAXAUG to avoid overfitting a handful of voices.
- GPU pitch-shift augmentation (falls back to CPU). Realistic: pitch/speed/real speech-babble noise.
- Prints a BALANCE AUDIT (voices, multiplier, final count per accent) so the mix is auditable.
Output: precalculated/eleven_accent_pos_done_big.npy  ([n,16,96], appends onto the 127k config-C base)
"""
import os, sys, glob
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heybuddy.embeddings import get_speech_embeddings

ACCENTS = ["japanese","chinese","spanish","latin","indian","filipino","korean","british","australian","american"]
SR=16000; SEG=int(SR*1.44); TARGET=2500; MAXAUG=60
NOISE = sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))
rng = np.random.default_rng(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"augmentation device: {DEV}", flush=True)

def fit(a):
    a = a[:SEG] if len(a) >= SEG else np.concatenate(
        [np.zeros((SEG-len(a))//2,"float32"), a, np.zeros(SEG-len(a)-(SEG-len(a))//2,"float32")])
    return a.astype("float32")

def augment(a):
    x = torch.tensor(a, device=DEV)
    n = int(rng.integers(-3,4))
    if n != 0: x = torchaudio.functional.pitch_shift(x, SR, n)
    rate = float(rng.uniform(0.9,1.12))
    if abs(rate-1) > 0.01:
        L = int(len(x)/rate)
        x = torch.nn.functional.interpolate(x[None,None], size=L, mode="linear", align_corners=False)[0,0]
    y = x.detach().cpu().numpy().astype("float32")
    if NOISE and rng.random() < 0.8:
        nz,_ = sf.read(NOISE[rng.integers(len(NOISE))], dtype="float32")
        if nz.ndim>1: nz=nz.mean(1)
        if len(nz)<len(y): nz=np.tile(nz, len(y)//len(nz)+1)
        nz=nz[:len(y)]; snr=rng.uniform(5,20); ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
        y = y + nz*np.sqrt(ps/pn/(10**(snr/10)))
    return fit(y)

emb_model = get_speech_embeddings(device_id=0)
auds, audit = [], []
for acc in ACCENTS:
    wavs = sorted(glob.glob(f"/tmp/eleven_big/train/{acc}/ozwell_done/*.wav"))
    if not wavs: continue
    M = int(np.clip(round(TARGET/len(wavs)), 1, MAXAUG))   # per-clip total variants
    cnt = 0
    for w in wavs:
        a,_ = sf.read(w, dtype="float32"); a = a if a.ndim==1 else a.mean(1)
        auds.append(fit(a)); cnt += 1
        for _ in range(M-1):
            auds.append(augment(a)); cnt += 1
    audit.append((acc, len(wavs), M, cnt))
    print(f"  {acc:11s} voices={len(wavs):3d}  x{M:2d}  -> {cnt}", flush=True)

out = []
for i in range(0, len(auds), 256):
    e = emb_model(auds[i:i+256], spectrogram_batch_size=32, embedding_batch_size=32, remove_nan=False)
    out.append(np.asarray(e, dtype="float32"))
    if i % 4096 == 0: print(f"  embed {i}/{len(auds)}", flush=True)
emb = np.concatenate(out); emb = emb[~np.isnan(emb).any(axis=(1,2))]
np.save("heybuddy/precalculated/eleven_accent_pos_done_big.npy", emb)
print("\n=== BALANCE AUDIT (accent | voices | xAug | positives) ===", flush=True)
for acc,v,m,c in audit: print(f"  {acc:11s} {v:3d}  x{m:2d}  {c}", flush=True)
print(f"TOTAL accent positives: {emb.shape} mean={float(emb[:,:16,:].mean()):.2f}", flush=True)
print("EMBED_BALANCED_DONE", flush=True)
