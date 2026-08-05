# Resolve the 75%-vs-0% contradiction. Embed eleven_big WAVs FRESH (the method that gave 75%) AND load the
# CACHED eleven embeddings, train the verifier with 5 FIXED seeds each, report real-kept mean+-spread.
# Tells us: (a) was 75% real or seed-noise, (b) do fresh vs cached embeddings genuinely differ.
import sys, glob, numpy as np
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
def add_noise(y):
    import soundfile as sf; NZ=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))
    nz,_=sf.read(NZ[rng.integers(len(NZ))],dtype="float32"); nz=nz if nz.ndim==1 else nz.mean(1)
    if len(nz)<len(y): nz=np.tile(nz,len(y)//len(nz)+1)
    nz=nz[:len(y)]; ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
    return (y+nz*np.sqrt(ps/pn/(10**(rng.uniform(5,15)/10)))).astype("float32")
# held-out test sets
rp,_=winsc(load_16k_mono("../../real_audio/Oz-done.wav")); rs=np.array([float(ev.wake.run(None,{"input":rp[i][None]})[0].reshape(-1)[0]) for i in range(len(rp))]); real=rp[rs>=0.5]
ami=[]
for f in sorted(glob.glob("/tmp/fphour/thirdparty_ami/*.wav")):
    y=load_16k_mono(f)
    for c in (y,add_noise(y),add_noise(y)):
        W,S=winsc(c)
        if W is not None: ami.append(W[S>=0.5])
ami=np.concatenate(ami)
gen=L("negs_C.npy",6000); mined=np.concatenate([np.load(P+"mined_false_fires.npy"),np.load(P+"mined_ff_done_vox.npy")])
# fresh-embedded eleven positives
el=sorted(glob.glob("/tmp/eleven_big/train/*/ozwell_done/*.wav"))[:700]
print(f"embedding {len(el)} eleven wavs fresh...",flush=True)
fresh=np.concatenate([best_win(load_16k_mono(f)) for f in el if best_win(load_16k_mono(f)) is not None])
cached=L("eleven_accent_pos_done_big.npy",len(fresh))   # match count
piper=L("ozwell_i_m_done.npy",len(fresh))
print(f"real wakes {len(real)} | AMI {len(ami)} | fresh-eleven {len(fresh)} | cached-eleven {len(cached)}\n",flush=True)
def evalpos(name,pos):
    rks,ars=[],[]
    for s in range(5):
        X=np.concatenate([flat(pos),flat(gen),flat(mined)]); y=np.concatenate([np.ones(len(pos)),np.zeros(len(gen)+len(mined))])
        m=MLPClassifier(hidden_layer_sizes=(256,64),max_iter=2000,early_stopping=True,random_state=s).fit(X,y)
        pr=m.predict_proba(flat(real))[:,1]; pa=m.predict_proba(flat(ami))[:,1]
        rks.append((pr>=.5).mean()*100); ars.append((pa<.5).mean()*100)
    print(f"{name:28s} real-kept@.5 = {np.mean(rks):3.0f}% (range {min(rks):.0f}-{max(rks):.0f}) | AMI-rej {np.mean(ars):3.0f}%")
evalpos("FRESH eleven (gave 75%)",fresh)
evalpos("CACHED eleven (gave 0%)",cached)
evalpos("CACHED Piper-Am",piper)
print("REPLICATE_DONE")
