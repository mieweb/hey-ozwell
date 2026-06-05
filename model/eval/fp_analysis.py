#!/usr/bin/env python3
"""
False-positive error analysis for the 100k wake-word model.

Two questions:
  1. WHICH negatives fire, and are they momentary spikes (1 window) or sustained runs?
  2. Does a "require K consecutive windows >= thr" rule (like a real device's debounce)
     cut the false-positive rate without killing recall?

The base harness scores a clip by its MAX window prob (one spike anywhere = fire).
Here we keep the full per-window sequence so we can distinguish spike vs sustained.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM

MODEL = "../checkpoints/scratch-onnx/ozwell_done_100k.onnx"
PRE   = "pretrained"
POSD  = "/tmp/eval/done/pos"
NEGD  = "/tmp/eval/done/neg"


def window_seq(ev, path):
    audio = load_16k_mono(path)
    mel = ev.mel.run(None, {"input": audio[None, :]})[0]
    mf = (mel.reshape(-1, 32) / 10.0 + 2.0).astype("float32")
    if mf.shape[0] < WIN:
        return None
    n = mf.shape[0]; n_trunc = n - (n - WIN) % STRIDE
    starts = range(0, n_trunc - WIN + 1, STRIDE)
    w = np.stack([mf[s:s + WIN] for s in starts])[..., None].astype("float32")
    emb = ev.emb.run(None, {"input_1": w})[0].reshape(-1, EMB_DIM).astype("float32")
    if emb.shape[0] < EMB_FRAMES:
        return None
    return np.array([
        float(ev.wake.run(None, {"input": emb[s:s + EMB_FRAMES][None].astype("float32")})[0].reshape(-1)[0])
        for s in range(0, emb.shape[0] - EMB_FRAMES + 1)
    ])


def longest_run(seq, thr):
    best = cur = 0
    for v in seq:
        cur = cur + 1 if v >= thr else 0
        best = max(best, cur)
    return best


def collect(ev, folder):
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        s = window_seq(ev, p)
        if s is not None:
            out[os.path.basename(p)] = s
    return out


def main():
    ev = WakeWordEvaluator(MODEL, PRE)
    pos = collect(ev, POSD)
    neg = collect(ev, NEGD)
    print(f"scored pos={len(pos)} neg={len(neg)} clips; avg windows/clip={np.mean([len(s) for s in neg.values()]):.1f}")

    THR = 0.5
    # ---- worst negatives: max prob + how many windows fire + longest consecutive run ----
    rows = []
    for name, s in neg.items():
        rows.append((name, s.max(), int((s >= THR).sum()), longest_run(s, THR), len(s)))
    rows.sort(key=lambda r: -r[1])
    print(f"\n=== top 20 firing NEGATIVES @thr={THR} (max | #win>=thr | longest run | total win) ===")
    for name, mx, nover, run, tot in rows[:20]:
        print(f"  {mx:.3f}  n>={nover:2d}  run={run:2d}/{tot:2d}  {name}")
    spikes = sum(1 for r in rows if r[1] >= THR and r[3] <= 1)
    fired  = sum(1 for r in rows if r[1] >= THR)
    print(f"\nof {fired} firing negatives, {spikes} fire on a SINGLE window (spike), {fired-spikes} sustain >=2")

    # ---- recall vs FPR under MAX rule vs K-consecutive rule ----
    def rate(d, thr, k):
        return np.mean([1.0 if longest_run(s, thr) >= k else 0.0 for s in d.values()]) * 100
    print(f"\n=== recall / per-clip FPR under sustained-frame rules (thr={THR}) ===")
    print("  rule              recall   FPR")
    for k in (1, 2, 3, 4):
        r = rate(pos, THR, k); f = rate(neg, THR, k)
        tag = "MAX (current)" if k == 1 else f"{k} consecutive"
        print(f"  {tag:16s}  {r:5.1f}%  {f:5.1f}%")
    # also at a higher threshold
    THR2 = 0.7
    print(f"\n=== same, thr={THR2} ===")
    print("  rule              recall   FPR")
    for k in (1, 2, 3):
        r = rate(pos, THR2, k); f = rate(neg, THR2, k)
        tag = "MAX (current)" if k == 1 else f"{k} consecutive"
        print(f"  {tag:16s}  {r:5.1f}%  {f:5.1f}%")


if __name__ == "__main__":
    main()
