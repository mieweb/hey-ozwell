#!/usr/bin/env python3
"""Recall under DEBOUNCE — does requiring N consecutive frames keep recall?
Debounced FP was 0.0/hr across every source (the prime FP lever), but the cost is recall on short
clips. This measures that cost directly: a clip 'detects' under debounce(N) if it has a run of >=N
consecutive windows over the operating threshold (same count_events logic as fp_per_hour).
Reports recall at min_run = 1 (current/no-debounce), 2, 3 per accent, on the big held-out test set.
Usage: python debounce_recall.py <model.onnx> <test_subfolder e.g. hey_ozwell> <threshold>
"""
import sys, glob, numpy as np
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_DIM, EMB_FRAMES
from fp_per_hour import count_events

MODEL, SUB, THR = sys.argv[1], sys.argv[2], float(sys.argv[3])
ev = WakeWordEvaluator(MODEL, "pretrained")

def seq_of(path):
    audio = load_16k_mono(path)
    mel = ev.mel.run(None, {"input": audio[None, :]})[0]
    mf = (mel.reshape(-1, 32) / 10.0 + 2.0).astype("float32")
    if mf.shape[0] < WIN: return None
    n = mf.shape[0]; n_trunc = n - (n - WIN) % STRIDE
    starts = range(0, n_trunc - WIN + 1, STRIDE)
    w = np.stack([mf[s:s+WIN] for s in starts])[..., None].astype("float32")
    emb = ev.emb.run(None, {"input_1": w})[0].reshape(-1, EMB_DIM).astype("float32")
    if emb.shape[0] < EMB_FRAMES: return None
    return np.array([float(ev.wake.run(None, {"input": emb[s:s+EMB_FRAMES][None].astype("float32")})[0].reshape(-1)[0])
                     for s in range(0, emb.shape[0]-EMB_FRAMES+1)])

print(f"### {MODEL.split('/')[-1]}  thr={THR}  (recall at debounce min_run = 1 / 2 / 3) ###")
RUNS = [1, 2, 3]
tot = {r: [] for r in RUNS}
for d in sorted(glob.glob(f"/tmp/eleven_big/test/*/{SUB}")):
    seqs = [seq_of(p) for p in sorted(glob.glob(d + "/*.wav"))]
    seqs = [s for s in seqs if s is not None and len(s)]
    if not seqs: continue
    acc = d.split("/")[-2]
    row = f"  {acc:11s} n={len(seqs):3d} "
    for r in RUNS:
        det = [1 if count_events(s, THR, r) >= 1 else 0 for s in seqs]
        tot[r].extend(det)
        row += f"  mr{r}={np.mean(det)*100:3.0f}%"
    print(row)
print(f"  {'ALL':11s} n={len(tot[1]):3d} " + "  ".join(f"  mr{r}={np.mean(tot[r])*100:3.0f}%" for r in RUNS))
