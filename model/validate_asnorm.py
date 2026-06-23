#!/usr/bin/env python3
"""
Does AS-norm hold the genuine speaker's score steady when the channel shifts?

Uses Jonathan's REAL wakes as the genuine voice and the cohort as impostors. Compares RAW cosine vs
AS-normed score, on CLEAN test clips and on the SAME clips with added noise (a simulated room change).
Win condition: raw genuine score drops a lot clean->noisy (the problem we keep hitting), while the
AS-normed score stays put (the fix).
"""
import sherpa_onnx, soundfile as sf, numpy as np, glob, json

MODEL = "/home/jlocala/sherpa-onnx/wasm/speaker-diarization/assets/nemo_en_titanet_small.onnx"
SR = 16000; SEG = int(1.2 * SR)
cohort = np.array(json.load(open("/tmp/sv-cohort.json")), dtype="float64")
cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=MODEL, num_threads=2, debug=False, provider="cpu")
ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

def embed(x, sr=SR):
    s = ext.create_stream(); s.accept_waveform(sr, x.astype("float32")); s.input_finished()
    e = np.array(ext.compute(s), "float64"); n = np.linalg.norm(e); return e/n if n > 0 else e

def segs(path):
    a, sr = sf.read(path); a = a.mean(1) if a.ndim > 1 else a; a = a.astype("float32")
    out = []
    for st in range(0, max(0, len(a)-SEG), SEG//2):
        seg = a[st:st+SEG]
        if np.sqrt(np.mean(seg**2)) > 0.02: out.append(seg)
    return out

# genuine voice = Jonathan's real wakes
gsegs = segs("../real_audio/Oz-done.wav") + segs("../real_audio/Hey-oz.wav")
gemb = np.array([embed(s) for s in gsegs])
half = len(gemb)//2
centroid = gemb[:half].mean(0); centroid /= np.linalg.norm(centroid)
test = gsegs[half:]

# noise to simulate a different room/channel
noise, _ = sf.read("/tmp/fphour/peoples_1h/0007.wav"); noise = (noise.mean(1) if noise.ndim>1 else noise).astype("float32")
def add_noise(x, snr_db):
    n = np.resize(noise, len(x)); g = np.sqrt(np.mean(x**2)/(np.mean(n**2)+1e-9)/(10**(snr_db/10)))
    return x + g*n

def asnorm(emb):
    raw = float(centroid @ emb)
    cs = cohort @ emb
    return raw, (raw - cs.mean()) / (cs.std() + 1e-9)

def stats(embs):
    raws = [asnorm(e)[0] for e in embs]; zs = [asnorm(e)[1] for e in embs]
    return np.mean(raws), np.mean(zs)

clean = np.array([embed(s) for s in test])
noisy = np.array([embed(add_noise(s, 5)) for s in test])      # +5 dB = a noisier room
imp = cohort[:120]

gr_c, gz_c = stats(clean); gr_n, gz_n = stats(noisy)
ir = np.mean([centroid @ e for e in imp]); iz = np.mean([asnorm(e)[1] for e in imp])
print("                      RAW cosine     AS-norm (z)")
print(f"genuine, clean room    {gr_c:.2f}          {gz_c:.2f}")
print(f"genuine, noisy room    {gr_n:.2f}          {gz_n:.2f}")
print(f"  drop clean->noisy     {gr_c-gr_n:+.2f}         {gz_c-gz_n:+.2f}   <- smaller drop = more stable")
print(f"impostor (cohort)      {ir:.2f}          {iz:.2f}")
print(f"  genuine-vs-impostor separation (noisy): RAW {gr_n-ir:.2f}  |  AS-norm {gz_n-iz:.2f}")
