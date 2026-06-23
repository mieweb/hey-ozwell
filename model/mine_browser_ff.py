#!/usr/bin/env python3
"""
BROWSER-FAITHFUL hard-negative mining.

Stream real conversational speech through the EXACT prod/js pipeline (per-1.08s-buffer peak-norm + rolling
4-buffer/16-frame assembly) and harvest every window where the wake model crosses threshold = a REAL browser
false fire. These embeddings become hard negatives for a precision retrain.

Why browser-faithful: the whole-clip-norm pipeline under-fires ~10x vs the browser, so the old mining missed
the false fires that actually occur live. Mining in-format catches the real ones.

GPU: pass --gpu (run in the CUDA venv, model/.venv-gpu) to put mel+embedding on a V100 (~47x faster).

Usage:
  python mine_browser_ff.py --phrase done --audio-dirs /tmp/fphour/peoples_1h /tmp/fphour/thirdparty_ami \
      --thr 0.5 --out precalculated/mined_ff_browser_done.npy
"""
import os, sys, glob, argparse, time
import numpy as np, soundfile as sf, scipy.signal as ss
import onnxruntime as ort

SR = 16000
BATCH = 17280   # 1.08s
HOP = 1920      # 0.12s
WIN, STR = 76, 8
EMB_FRAMES, EMB_DIM = 16, 96
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "pretrained")
WAKE = {
    "done": "checkpoints/scratch-onnx/ozwelldone_surgical.onnx",
    "hey":  "../prod/js/models/hey-ozwell.onnx",
}


def mk(path, gpu, dev):
    prov = ([("CUDAExecutionProvider", {"device_id": dev})] if gpu else []) + ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    return ort.InferenceSession(path, sess_options=so, providers=prov)


def load_wav(p):
    a, sr = sf.read(p)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        a = ss.resample(a, int(len(a) * SR / sr))
    return a.astype("float32")


def buffer_embed(buf, mel, emb):
    pk = float(np.max(np.abs(buf)))
    if pk > 1e-5:
        buf = buf / pk
    m = mel.run(None, {"input": buf[None, :].astype("float32")})[0]
    mf = (m.reshape(-1, 32) / 10.0 + 2.0).astype("float32")
    nt = mf.shape[0] - (mf.shape[0] - WIN) % STR
    wins = np.stack([mf[s:s + WIN] for s in range(0, nt - WIN + 1, STR)])[..., None].astype("float32")
    return emb.run(None, {"input_1": wins})[0].reshape(-1, EMB_DIM).astype("float32")


def stream_ebs(audio, mel, emb):
    audio = audio.astype("float32")
    buf = np.zeros(BATCH, "float32")
    recent = []
    for i in range(0, len(audio), HOP):
        chunk = audio[i:i + HOP]
        buf = np.roll(buf, -len(chunk))
        buf[-len(chunk):] = chunk
        recent.append(buffer_embed(buf, mel, emb))
        if len(recent) > EMB_FRAMES // 4:
            recent.pop(0)
        if len(recent) == EMB_FRAMES // 4:
            yield np.concatenate(recent, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", choices=["done", "hey"], required=True)
    ap.add_argument("--audio-dirs", nargs="+", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="max clips (0=all)")
    args = ap.parse_args()

    mel = mk(f"{P}/mel-spectrogram.onnx", args.gpu, args.device_id)
    emb = mk(f"{P}/speech-embedding.onnx", args.gpu, args.device_id)
    wake = mk(WAKE[args.phrase], args.gpu, args.device_id)
    print("providers:", wake.get_providers()[0])

    files = []
    for d in args.audio_dirs:
        files += sorted(glob.glob(os.path.join(d, "*.wav")))
    if args.limit:
        files = files[:args.limit]
    print(f"[{args.phrase}] mining {len(files)} clips @ thr {args.thr}")

    mined, n_win, sec, t0 = [], 0, 0.0, time.time()
    for j, p in enumerate(files):
        a = load_wav(p)
        sec += len(a) / SR
        for eb in stream_ebs(a, mel, emb):
            n_win += 1
            v = float(wake.run(None, {"input": eb[None].astype("float32")})[0].reshape(-1)[0])
            if v >= args.thr:
                mined.append(eb.astype("float32"))
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(files)} | {sec/60:.1f} min audio | {len(mined)} fires | {time.time()-t0:.0f}s")

    arr = np.stack(mined) if mined else np.zeros((0, EMB_FRAMES, EMB_DIM), "float32")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, arr)
    hrs = sec / 3600
    print(f"\nMINED {arr.shape[0]} browser false-fires from {hrs:.2f}h ({n_win} windows) -> {args.out}")
    print(f"  raw false-fire rate: {arr.shape[0]/max(n_win,1)*100:.2f}% of windows  ({arr.shape[0]/max(hrs,1e-9):.0f}/hr-of-audio)")


if __name__ == "__main__":
    main()
