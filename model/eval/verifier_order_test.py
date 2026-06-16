# Test: does a reordering of the 1536-vector drive a REAL window from P=1.0 to ~0? If a transpose
# ([16,96]->[96,16] flatten) gives ~0, the browser is feeding the verifier in the wrong order = the bug.
import sys, glob, numpy as np
sys.path.insert(0,".")
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM
import onnxruntime as ort
ev=WakeWordEvaluator("../checkpoints/scratch-onnx/ozwelldone_surgical.onnx","pretrained")
vs=ort.InferenceSession("../../prod/js/models/ozwell-i'm-done-verifier.onnx",providers=["CPUExecutionProvider"])
def winsc(a):
    mel=ev.mel.run(None,{"input":a[None,:]})[0]; mf=(mel.reshape(-1,32)/10+2).astype("float32")
    n=mf.shape[0]; nt=n-(n-WIN)%STRIDE; st=range(0,nt-WIN+1,STRIDE)
    w=np.stack([mf[s:s+WIN] for s in st])[...,None].astype("float32")
    emb=ev.emb.run(None,{"input_1":w})[0].reshape(-1,EMB_DIM).astype("float32")
    ws=np.stack([emb[s:s+EMB_FRAMES] for s in range(0,emb.shape[0]-EMB_FRAMES+1)])
    sc=np.array([float(ev.wake.run(None,{"input":ws[i][None]})[0].reshape(-1)[0]) for i in range(len(ws))])
    return ws,sc
def P(vec): return float(vs.run(None,{"input":vec.reshape(1,-1).astype("float32")})[1][0,1])
W,S=winsc(load_16k_mono("../../real_audio/Oz-done.wav"))
win=W[S.argmax()]  # [16,96], a real wake window
print(f"window shape {win.shape}, pass-1 score {S.max():.2f}\n")
print(f"  correct order  [16,96]->flat (frame-major) : P={P(win.reshape(-1)):.3f}   (expect ~1.0)")
print(f"  TRANSPOSE      [96,16]->flat (dim-major)    : P={P(win.T.reshape(-1)):.3f}")
print(f"  frames reversed                             : P={P(win[::-1].reshape(-1)):.3f}")
print(f"  dims reversed                               : P={P(win[:,::-1].reshape(-1)):.3f}")
# stats are permutation-invariant -> identical for all of the above:
f=win.reshape(-1); print(f"\n  (all orderings share min={f.min():.2f} max={f.max():.2f} mean={f.mean():.2f} -- why browser stats matched)")
print("ORDER_DONE")
