# Reference input fingerprint: what the verifier's input SHOULD look like (offline embeddings) for a real
# wake, a synthetic wake, and a false-fire. Compare these stats to the browser's [acoustic-verifier] INPUT log.
import sys, glob, numpy as np
sys.path.insert(0,".")
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM
import onnxruntime as ort
ev=WakeWordEvaluator("../checkpoints/scratch-onnx/ozwelldone_surgical.onnx","pretrained")
vs=ort.InferenceSession("../../prod/js/models/ozwell-i'm-done-verifier.onnx",providers=["CPUExecutionProvider"])
def winsc(a):
    mel=ev.mel.run(None,{"input":a[None,:]})[0]; mf=(mel.reshape(-1,32)/10+2).astype("float32")
    if mf.shape[0]<WIN: return None,None
    n=mf.shape[0]; nt=n-(n-WIN)%STRIDE; st=range(0,nt-WIN+1,STRIDE)
    w=np.stack([mf[s:s+WIN] for s in st])[...,None].astype("float32")
    emb=ev.emb.run(None,{"input_1":w})[0].reshape(-1,EMB_DIM).astype("float32")
    if emb.shape[0]<EMB_FRAMES: return None,None
    ws=np.stack([emb[s:s+EMB_FRAMES] for s in range(0,emb.shape[0]-EMB_FRAMES+1)])
    sc=np.array([float(ev.wake.run(None,{"input":ws[i][None]})[0].reshape(-1)[0]) for i in range(len(ws))])
    return ws,sc
def stat(name,win):
    f=win.reshape(-1).astype("float32")
    p=vs.run(None,{"input":f[None]})[1][0,1]
    print(f"{name:28s} dims={list(win.shape)} len={len(f)} min={f.min():.3f} max={f.max():.3f} mean={f.mean():.3f} first5=[{','.join(f'{v:.3f}' for v in f[:5])}]  verifierP={p:.3f}")
W,S=winsc(load_16k_mono("../../real_audio/Oz-done.wav"))
fire=W[S>=0.5]
print("# OFFLINE reference (what the browser SHOULD be feeding the verifier):")
if len(fire): stat("REAL ozwell-done (best win)", fire[S[S>=0.5].argmax()] if False else W[S.argmax()])
# a synthetic positive for contrast
el=sorted(glob.glob("/tmp/eleven_big/test/american/ozwell_done/*.wav"))
if el:
    W2,S2=winsc(load_16k_mono(el[0])); stat("SYNTH ozwell-done (eleven)", W2[S2.argmax()])
print("\n# Compare these to the browser's '[acoustic-verifier] INPUT ...' line.")
print("# If browser min/max/mean/first5 differ a lot -> embedding/scaling mismatch (the bug).")
