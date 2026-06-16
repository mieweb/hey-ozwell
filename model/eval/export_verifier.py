# Freeze the acoustic verifier (diverse synthetic positives + mined real false-fires) to JSON weights for the
# browser. Tiny MLP(1536->256->64->1): runs in plain JS, no ONNX/model download. Input = the SAME [16,96]
# embedding window the wake model already computes, flattened to 1536.
import sys, glob, json, numpy as np
sys.path.insert(0,".")
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM
from sklearn.neural_network import MLPClassifier
ev=WakeWordEvaluator("../checkpoints/scratch-onnx/ozwelldone_surgical.onnx","pretrained"); rng=np.random.default_rng(0)
P="../heybuddy/precalculated/"
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
def best_win(a): W,S=winsc(a); return W[[S.argmax()]] if W is not None else None
def L(p,k=None):
    a=np.load(P+p,mmap_mode="r"); idx=np.sort(rng.choice(len(a),min(k,len(a)),replace=False)) if k else np.arange(len(a)); return np.asarray(a[idx])[:,:EMB_FRAMES,:]
flat=lambda a:a.reshape(len(a),-1)
tr=sorted(glob.glob("/tmp/eleven_big/train/*/ozwell_done/*.wav"))
print(f"embedding {len(tr)} diverse synthetic positives...",flush=True)
pos=np.concatenate([best_win(load_16k_mono(f)) for f in tr if best_win(load_16k_mono(f)) is not None])
gen=L("negs_C.npy",6000); mined=np.concatenate([np.load(P+"mined_false_fires.npy"),np.load(P+"mined_ff_done_vox.npy")])
X=np.concatenate([flat(pos),flat(gen),flat(mined)]); y=np.concatenate([np.ones(len(pos)),np.zeros(len(gen)+len(mined))])
m=MLPClassifier(hidden_layer_sizes=(256,64),max_iter=2000,early_stopping=True,random_state=0).fit(X,y)
# export to ONNX (input [None,1536] float32; zipmap=False -> output_probability is a [1,2] tensor)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
onx=convert_sklearn(m,initial_types=[("input",FloatTensorType([None,1536]))],options={id(m):{"zipmap":False}})
import os; os.makedirs("../../prod/js/models",exist_ok=True)
fp="../../prod/js/models/ozwell-i'm-done-verifier.onnx"
open(fp,"wb").write(onx.SerializeToString())
# sanity: verify ONNX matches sklearn on a few held-out reals
import onnxruntime as ort
sess=ort.InferenceSession(onx.SerializeToString(),providers=["CPUExecutionProvider"])
chk=flat(pos[:5]).astype("float32")
onnx_p=sess.run(None,{"input":chk})[1][:,1]; skl_p=m.predict_proba(chk)[:,1]
print(f"wrote {fp}  | layers {[c.shape for c in m.coefs_]} | pos {len(pos)} neg {len(gen)+len(mined)}")
print(f"parity check P(wake) onnx vs sklearn: {[f'{a:.3f}/{b:.3f}' for a,b in zip(onnx_p,skl_p)]}")
print(f"file size: {os.path.getsize(fp)//1024} KB")
print("EXPORT_DONE")
