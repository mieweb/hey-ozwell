#!/usr/bin/env python3
"""
Browser-faithful BULK negatives: stream real speech and subsample every window (not just firing ones).
These are the "easy" negative pool; the mined ~230 (mine_browser_ff.py) are the hard boundary negatives.

GPU: run in model/.venv-gpu with LD_LIBRARY_PATH set (see logs/run_mine_browser.sh).

Usage:
  python gen_browser_negs.py --audio-dirs /tmp/fphour/thirdparty_ami /tmp/fphour/peoples_1h \
      --keep 0.25 --cap 40000 --out precalculated/browser_negs_train.npy --gpu
"""
import os, sys, glob, argparse, random
import numpy as np, soundfile as sf, scipy.signal as ss
import onnxruntime as ort
from mine_browser_ff import mk, load_wav, stream_ebs, EMB_FRAMES, EMB_DIM, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dirs", nargs="+", required=True)
    ap.add_argument("--keep", type=float, default=0.25, help="fraction of windows to keep")
    ap.add_argument("--cap", type=int, default=40000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    mel = mk(f"{P}/mel-spectrogram.onnx", args.gpu, args.device_id)
    emb = mk(f"{P}/speech-embedding.onnx", args.gpu, args.device_id)
    print("providers:", emb.get_providers()[0])

    files = []
    for d in args.audio_dirs:
        files += sorted(glob.glob(os.path.join(d, "*.wav")))
    rng.shuffle(files)
    print(f"streaming {len(files)} clips, keep {args.keep}, cap {args.cap}")

    out = []
    for j, p in enumerate(files):
        a = load_wav(p)
        for eb in stream_ebs(a, mel, emb):
            if rng.random() < args.keep:
                out.append(eb.astype("float32"))
        if len(out) >= args.cap:
            break
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{len(files)} | {len(out)} negs")

    arr = np.stack(out[:args.cap]) if out else np.zeros((0, EMB_FRAMES, EMB_DIM), "float32")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, arr)
    print(f"BULK negs: {arr.shape} -> {args.out}")


if __name__ == "__main__":
    main()
