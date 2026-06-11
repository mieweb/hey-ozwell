#!/usr/bin/env python3
"""Embed + augment the ElevenLabs accent TRAIN clips into positive .npy (one per phrase).
Mirrors gen_accent_full.py exactly: fit to SEG, x AUG realistic variants (pitch/speed/real
speech-babble noise), embed -> [n,16,96]. TEST clips stay as wavs (held-out recall, no aug).
Outputs: precalculated/eleven_accent_pos_{hey,done}.npy
"""
import os, sys, glob
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heybuddy.embeddings import get_speech_embeddings

SR = 16000; SEG = int(SR * 1.44); AUG = 12
NOISE = sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))
rng = np.random.default_rng(0)

def fit(a):
    a = a[:SEG] if len(a) >= SEG else np.concatenate(
        [np.zeros((SEG-len(a))//2,"float32"), a, np.zeros(SEG-len(a)-(SEG-len(a))//2,"float32")])
    return a.astype("float32")

def augment(a):
    x = torch.tensor(a)
    n = int(rng.integers(-3, 4))
    if n != 0: x = torchaudio.functional.pitch_shift(x, SR, n)
    rate = float(rng.uniform(0.9, 1.12))
    if abs(rate-1) > 0.01:
        L = int(len(x)/rate)
        x = torch.nn.functional.interpolate(x[None,None], size=L, mode="linear", align_corners=False)[0,0]
    y = x.numpy().astype("float32")
    if NOISE and rng.random() < 0.8:
        nz,_ = sf.read(NOISE[rng.integers(len(NOISE))], dtype="float32")
        if nz.ndim > 1: nz = nz.mean(1)
        if len(nz) < len(y): nz = np.tile(nz, len(y)//len(nz)+1)
        nz = nz[:len(y)]; snr = rng.uniform(5,20)
        ps,pn = (y**2).mean()+1e-9, (nz**2).mean()+1e-9
        y = y + nz*np.sqrt(ps/pn/(10**(snr/10)))
    return fit(y)

PHRASES = {"hey_ozwell":"eleven_accent_pos_hey.npy", "ozwell_done":"eleven_accent_pos_done.npy"}
emb_model = get_speech_embeddings(device_id=0)
for slug, outname in PHRASES.items():
    wavs = sorted(glob.glob(f"/tmp/eleven_pos/train/*/{slug}/*.wav"))
    bases = []
    for w in wavs:
        a,_ = sf.read(w, dtype="float32")
        bases.append(a if a.ndim==1 else a.mean(1))
    auds = []
    for a in bases:
        auds.append(fit(a))
        for _ in range(AUG): auds.append(augment(a))
    out = []
    for i in range(0, len(auds), 256):
        e = emb_model(auds[i:i+256], spectrogram_batch_size=32, embedding_batch_size=32, remove_nan=False)
        out.append(np.asarray(e, dtype="float32"))
        if i % 2048 == 0: print(f"  [{slug}] embed {i}/{len(auds)}", flush=True)
    emb = np.concatenate(out); emb = emb[~np.isnan(emb).any(axis=(1,2))]
    np.save(f"heybuddy/precalculated/{outname}", emb)
    print(f"{outname}: {emb.shape} from {len(wavs)} raw clips x{AUG+1}  mean={float(emb[:,:16,:].mean()):.2f}", flush=True)
print("ELEVEN_EMBED_DONE", flush=True)
