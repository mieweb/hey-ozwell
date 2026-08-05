#!/usr/bin/env python3
"""
Real false-positives-per-HOUR eval (the actual product metric — not per-clip).

Slides the wake model continuously over held-out real speech and counts FIRE EVENTS
(a maximal run of windows >= threshold = ONE event, so a sustained fire isn't
double-counted), then divides by total audio hours. Optional --min-run K requires the
run to last K windows (a debounce, like requiring K consecutive frames before firing).

Input audio should be REAL speech the model never trained on (e.g. LibriSpeech test-clean
or held-out People's Speech). Audio is peak-normalized per clip (matches the trained models).

NOTE: prod additionally VAD-gates (only runs the wake model on speech). This v1 has no VAD,
so on speech-dense audio it's a slightly CONSERVATIVE (upper-bound) FP/hour estimate.
"""
import argparse, glob, os, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM


def window_seq(ev, audio):
    mel = ev.mel.run(None, {"input": audio[None, :]})[0]
    mf = (mel.reshape(-1, 32) / 10.0 + 2.0).astype("float32")
    if mf.shape[0] < WIN:
        return np.array([])
    n = mf.shape[0]; n_trunc = n - (n - WIN) % STRIDE
    starts = range(0, n_trunc - WIN + 1, STRIDE)
    w = np.stack([mf[s:s + WIN] for s in starts])[..., None].astype("float32")
    emb = ev.emb.run(None, {"input_1": w})[0].reshape(-1, EMB_DIM).astype("float32")
    if emb.shape[0] < EMB_FRAMES:
        return np.array([])
    return np.array([
        float(ev.wake.run(None, {"input": emb[s:s + EMB_FRAMES][None].astype("float32")})[0].reshape(-1)[0])
        for s in range(0, emb.shape[0] - EMB_FRAMES + 1)
    ])


def count_events(seq, thr, min_run):
    events = run = 0
    for v in seq:
        if v >= thr:
            run += 1
        else:
            if run >= min_run:
                events += 1
            run = 0
    if run >= min_run:
        events += 1
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio-dir", required=True, help="folder of held-out real-speech wavs")
    ap.add_argument("--pretrained-dir", default="pretrained")
    ap.add_argument("--label", default="model")
    ap.add_argument("--thresholds", default="0.5,0.7,0.9")
    ap.add_argument("--browser", action="store_true",
                    help="Score via the BROWSER-FAITHFUL pipeline (per-1.08s-buffer peak-norm + rolling "
                         "4-buffer assembly, like prod/js) instead of whole-clip norm. Matches what the "
                         "browser actually does -> truthful (higher) live FP estimate.")
    args = ap.parse_args()
    THRS = [float(x) for x in args.thresholds.split(",")]

    ev = WakeWordEvaluator(args.model, args.pretrained_dir)
    if args.browser:
        from browser_embed import stream_scores
    files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    total_sec = 0.0
    seqs = []
    for p in files:
        info = sf.info(p)
        total_sec += info.frames / info.samplerate
        s = stream_scores(load_16k_mono(p), ev.wake) if args.browser else window_seq(ev, load_16k_mono(p))
        if s.size:
            seqs.append(s)
    hours = total_sec / 3600.0
    print(f"\n=== {args.label}: FP/hour over {hours:.2f}h of held-out real speech ({len(files)} clips) ===")
    print("  thr    min-run=1 (prod single-frame)   min-run=3 (debounce)")
    for thr in THRS:
        e1 = sum(count_events(s, thr, 1) for s in seqs)
        e3 = sum(count_events(s, thr, 3) for s in seqs)
        print(f"  {thr:.1f}   {e1:5d} fires = {e1/hours:7.1f}/hr        {e3:5d} fires = {e3/hours:7.1f}/hr")
    print("  (no VAD gating yet — conservative upper bound; target is <1/hr)")


if __name__ == "__main__":
    main()
