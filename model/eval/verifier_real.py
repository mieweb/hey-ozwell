import sys, glob, numpy as np
sys.path.insert(0, ".")
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
ev=WakeWordEvaluator("../checkpoints/scratch-onnx/ozwelldone_surgical.onnx","pretrained"); rng=np.random.default_rng(0)
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
def mine_one(f):
    W,S=winsc(load_16k_mono(f))
    import numpy as _np
    return W[S>=0.5] if W is not None else _np.zeros((0,EMB_FRAMES,EMB_DIM),'float32')
import soundfile as _sf
_NOISE=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))
def _add_noise(y):
    if not _NOISE: return y
    nz,_=_sf.read(_NOISE[rng.integers(len(_NOISE))],dtype="float32"); nz=nz if nz.ndim==1 else nz.mean(1)
    if len(nz)<len(y): nz=np.tile(nz,len(y)//len(nz)+1)
    nz=nz[:len(y)]; ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
    return (y+nz*np.sqrt(ps/pn/(10**(rng.uniform(5,18)/10)))).astype("float32")
def mine(d):  # noise-augment too, so the independent test is large (matches training conditions)
    out=[]
    for f in sorted(glob.glob(f"{d}/*.wav")):
        y=load_16k_mono(f)
        for clip in (y, _add_noise(y), _add_noise(y)):
            W,S=winsc(clip)
            if W is not None: out.append(W[S>=0.5])
    return np.concatenate(out) if out else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32")
import os as _os
_srcs=[f for f in ["../heybuddy/precalculated/mined_false_fires.npy","../heybuddy/precalculated/mined_ff_done_vox.npy"] if _os.path.exists(f)]
mined=np.concatenate([np.load(f) for f in _srcs])  # TRAIN hard-negs: People's + VoxPopuli (diverse)
print("train sources:", _srcs)
ff_ami=mine("/tmp/fphour/thirdparty_ami")                                  # TEST hard-negs (independent corpus)
ff_you=__import__("numpy").concatenate([mine_one(f) for f in __import__("glob").glob("../../real_audio/*.wav") if "Oz-done" not in f and "Hey-oz" not in f])  # YOUR 12-min false-fires (most representative)
rp,_=winsc(load_16k_mono("../../real_audio/Oz-done.wav"))
rs=np.array([float(ev.wake.run(None,{"input":rp[i][None]})[0].reshape(-1)[0]) for i in range(len(rp))]); rp=rp[rs>=0.5]
tts=np.asarray(np.load("../heybuddy/precalculated/ozwell_i_m_done.npy",mmap_mode="r")[rng.choice(127000,5000,replace=False)])
gen=np.asarray(np.load("../heybuddy/precalculated/negs_C.npy",mmap_mode="r")[rng.choice(160000,5000,replace=False)])[:,:EMB_FRAMES,:]
flat=lambda a:a.reshape(len(a),-1)
i=rng.permutation(len(rp)); rp_tr,rp_te=rp[i[len(rp)//2:]],rp[i[:len(rp)//2]]
X=np.concatenate([flat(tts),flat(rp_tr),flat(gen),flat(mined)]); y=np.concatenate([np.ones(len(tts)+len(rp_tr)),np.zeros(len(gen)+len(mined))])
print(f"train hard-negs (mined): {len(mined)} | AMI held-out false-fires: {len(ff_ami)} | real-wake test: {len(rp_te)}")
for name,m in [("logistic",LogisticRegression(max_iter=3000,class_weight="balanced")),
               ("MLP(256,64)",MLPClassifier(hidden_layer_sizes=(256,64),max_iter=2000,early_stopping=True))]:
    m.fit(X,y)
    print(f"  {name:12s}: real wakes kept {m.predict(flat(rp_te)).mean()*100:3.0f}% | YOUR-recording false-fires REJECTED {100-m.predict(flat(ff_you)).mean()*100:3.0f}% (n={len(ff_you)}) | AMI REJECTED {100-m.predict(flat(ff_ami)).mean()*100:3.0f}%")

# threshold sweep: reject if P(wake) < t. Find the point keeping real wakes high + rejecting false-fires.
from sklearn.neural_network import MLPClassifier as _M
mlp=_M(hidden_layer_sizes=(256,64),max_iter=2000,early_stopping=True).fit(X,y)
pr=lambda a: mlp.predict_proba(flat(a))[:,1] if len(a) else __import__("numpy").array([])
Pr_real, Pr_you, Pr_ami = pr(rp_te), pr(ff_you), pr(ff_ami)
print("\n=== MLP threshold sweep (reject if P<t) ===")
print("   t    real-wakes-kept   YOUR-FF-rejected   AMI-FF-rejected")
for t in [0.3,0.4,0.5,0.6,0.7,0.8]:
    rk=(Pr_real>=t).mean()*100; yr=(Pr_you<t).mean()*100; ar=(Pr_ami<t).mean()*100
    print(f"  {t:.1f}     {rk:4.0f}%            {yr:4.0f}%              {ar:4.0f}%")
print("SWEEP_DONE")
print("VERIFIER_REAL_DONE")
