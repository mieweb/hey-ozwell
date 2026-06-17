#!/usr/bin/env python3
"""Score a prod model's recall on the big held-out ElevenLabs accent test set."""
import sys, glob, numpy as np
from evaluate_wakeword import WakeWordEvaluator
MODEL, SUB, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
ev = WakeWordEvaluator(MODEL, "pretrained")
print(f"### {LABEL} on big held-out test ({SUB}) ###")
allp = []
for d in sorted(glob.glob(f"/tmp/eleven_big/test/*/{SUB}")):
    w = glob.glob(d + "/*.wav")
    if not w: continue
    p = ev.score_folder(d); allp.append(p); acc = d.split("/")[-2]
    print(f"  {acc:11s} n={len(p):3d}  @0.5={(p>=0.5).mean()*100:3.0f}%  @0.8={(p>=0.8).mean()*100:3.0f}%  @0.9={(p>=0.9).mean()*100:3.0f}%")
p = np.concatenate(allp); name = "ALL"
print(f"  {name:11s} n={len(p):3d}  @0.5={(p>=0.5).mean()*100:3.0f}%  @0.8={(p>=0.8).mean()*100:3.0f}%  @0.9={(p>=0.9).mean()*100:3.0f}%")
