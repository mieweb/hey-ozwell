#!/usr/bin/env python3
"""
Noise/RIR-augmented, BROWSER-FAITHFUL training-data generator (retrain proof).

Two gaps this addresses at once:
  1. NOISE/REVERB robustness — augment positives with real background noise (People's Speech) and
     synthetic room reverb (pyroomacoustics), so the model learns to fire in real-room conditions.
  2. TRAIN/SERVE match — embed every clip through browser_embed (per-1.08s-buffer peak-norm + rolling
     4-buffer assembly), the EXACT representation prod/js produces live. (Old training used whole-clip norm.)

Positives -> the PEAK wake-window per clip (what fires live / what enrollment captures).
Negatives -> every window of held-out real speech.

Usage:
  python gen_noise_aug.py --phrase done --n 300 --out precalculated/nbf_done
"""
import os, sys, glob, argparse, random
import numpy as np, soundfile as sf
import scipy.signal as ss
import pyroomacoustics as pra
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))
from browser_embed import stream_embeddingbuffers

SR = 16000
WAKE = {  # peak-window selector (the deployed model for that phrase)
    "done": "checkpoints/scratch-onnx/ozwelldone_surgical.onnx",
    "hey":  "../prod/js/models/hey-ozwell.onnx",
}
POS_SUBDIR = {"done": "ozwell_done", "hey": "hey_ozwell"}


def load_wav(p):
    a, sr = sf.read(p)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        a = ss.resample(a, int(len(a) * SR / sr))
    return a.astype("float32")


def rand_rir(rng):
    """Synthetic shoebox-room impulse response (random geometry + RT60)."""
    dim = [rng.uniform(3, 8), rng.uniform(3, 7), rng.uniform(2.5, 3.5)]
    rt60 = rng.uniform(0.2, 0.6)
    try:
        e_abs, max_order = pra.inverse_sabine(rt60, dim)
    except Exception:
        e_abs, max_order = 0.4, 8
    room = pra.ShoeBox(dim, fs=SR, materials=pra.Material(e_abs), max_order=int(min(max_order, 10)))
    src = [rng.uniform(0.5, d - 0.5) for d in dim]
    mic = [rng.uniform(0.5, d - 0.5) for d in dim]
    room.add_source(src)
    room.add_microphone(np.array(mic).reshape(3, 1))
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype="float32")
    return rir / (np.abs(rir).max() + 1e-9)


def add_reverb(a, rir):
    y = ss.fftconvolve(a, rir)[:len(a)]
    return y.astype("float32")


def add_noise(a, noise, snr_db, rng):
    if len(noise) < len(a):
        noise = np.tile(noise, int(np.ceil(len(a) / len(noise))))
    start = rng.randint(0, len(noise) - len(a)) if len(noise) > len(a) else 0
    n = noise[start:start + len(a)]
    apw = np.mean(a ** 2) + 1e-9
    npw = np.mean(n ** 2) + 1e-9
    g = np.sqrt(apw / npw / (10 ** (snr_db / 10)))
    return (a + g * n).astype("float32")


def peak_window(audio, wake):
    """The single [16,96] window where the wake model is most confident (what fires live)."""
    best, bestv = None, -1.0
    for eb in stream_embeddingbuffers(audio):
        v = float(wake.run(None, {"input": eb[None].astype("float32")})[0].reshape(-1)[0])
        if v > bestv:
            bestv, best = v, eb
    return best, bestv


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", choices=["done", "hey"], required=True)
    ap.add_argument("--n", type=int, default=300, help="positive clips to use")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--out", required=True, help="output dir for *.npy")
    ap.add_argument("--noise-dir", default="/tmp/fphour/peoples_1h")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    wake = ort.InferenceSession(WAKE[args.phrase], providers=["CPUExecutionProvider"])
    sub = POS_SUBDIR[args.phrase]
    pos = sorted(glob.glob(f"/tmp/eleven_big/{args.split}/*/{sub}/*.wav"))
    rng.shuffle(pos)
    pos = pos[:args.n]
    noise_files = sorted(glob.glob(os.path.join(args.noise_dir, "*.wav")))
    print(f"[{args.phrase}] {len(pos)} positive clips, {len(noise_files)} noise files")

    conds = {"clean": [], "noise": [], "reverb": [], "both": []}
    for i, p in enumerate(pos):
        a = load_wav(p)
        if len(a) < SR // 2:
            continue
        noise = load_wav(rng.choice(noise_files))
        rir = rand_rir(rng)
        # realistic near-field SNR for positive augmentation (voice dominates) — avoids garbage peak windows
        # from clips where noise drowns the phrase. Light robustness, not the unrealistic noise-louder regime.
        snr = rng.uniform(3, 18)
        variants = {
            "clean": a,
            "noise": add_noise(a, noise, snr, rng),
            "reverb": add_reverb(a, rir),
            "both": add_noise(add_reverb(a, rir), noise, snr, rng),
        }
        for k, v in variants.items():
            w, _ = peak_window(v, wake)
            if w is not None:
                conds[k].append(w.astype("float32"))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pos)}")

    for k, v in conds.items():
        arr = np.stack(v) if v else np.zeros((0, 16, 96), "float32")
        np.save(os.path.join(args.out, f"pos_{k}.npy"), arr)
        print(f"  pos_{k}: {arr.shape}")
    print("done ->", args.out)


if __name__ == "__main__":
    main()
