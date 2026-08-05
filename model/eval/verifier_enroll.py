# Simulate ENROLLMENT: add a few of YOUR real utterances to the verifier's positives, test on your HELD-OUT
# utterances. (Legit, not a leak: enrollment trains on the user's voice to recognize the user's voice.)
# Question: do the confidently-rejected real wakes (0.00) flip to accepted, while FP-rejection holds?
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
# synthetic positives + negatives (fixed)
tr=sorted(glob.glob("/tmp/eleven_big/train/*/ozwell_done/*.wav"))
print(f"embedding {len(tr)} synthetic train voices...",flush=True)
syn=np.concatenate([best_win(load_16k_mono(f)) for f in tr if best_win(load_16k_mono(f)) is not None])
gen=L("negs_C.npy",6000); mined=np.concatenate([np.load(P+"mined_false_fires.npy"),np.load(P+"mined_ff_done_vox.npy")])
ami=[]
for f in sorted(glob.glob("/tmp/fphour/thirdparty_ami/*.wav")):
    y=load_16k_mono(f)
    for c in (y,add_noise(y),add_noise(y)):
        W,S=winsc(c)
        if W is not None: ami.append(W[S>=0.5])
ami=np.concatenate(ami)
# your recording -> utterances
W,S=winsc(load_16k_mono("../../real_audio/Oz-done.wav")); fire=S>=0.5
utts=[]; cur=[]
for i,f in enumerate(fire):
    if f: cur.append(i)
    elif cur and (i-cur[-1])>5: utts.append(cur); cur=[]
if cur: utts.append(cur)
uttW=[W[u] for u in utts]; n=len(utts)
def train_eval(enroll_idx):
    test_idx=[i for i in range(n) if i not in enroll_idx]
    enroll_pos=np.concatenate([uttW[i] for i in enroll_idx]) if enroll_idx else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32")
    pos=np.concatenate([syn,enroll_pos]); X=np.concatenate([flat(pos),flat(gen),flat(mined)]); y=np.concatenate([np.ones(len(pos)),np.zeros(len(gen)+len(mined))])
    m=MLPClassifier(hidden_layer_sizes=(256,64),max_iter=2000,early_stopping=True,random_state=0).fit(X,y)
    kept=np.mean([ (m.predict_proba(flat(uttW[i]))[:,1]>=0.5).any() for i in test_idx ])*100
    amir=(m.predict_proba(flat(ami))[:,1]<0.5).mean()*100
    return kept,amir,len(test_idx)
print(f"\nyour utterances: {n}. Baseline (NO enrollment) per-utterance kept = 62% (8/13)\n")
print("=== ENROLLMENT SIMULATION: add K of your reps as positives, test on held-out reps ===")
print(" K-enrolled   held-out real-utt kept   AMI FP killed   (avg of random splits)")
for K in [0,3,5,8]:
    if K==0:
        kept,amir,nt=train_eval([]); print(f"   {K:2d}          {kept:4.0f}%  (n={nt})            {amir:3.0f}%")
    else:
        res=[train_eval(list(rng.choice(n,K,replace=False))) for _ in range(5)]
        kept=np.mean([r[0] for r in res]); amir=np.mean([r[1] for r in res]); nt=res[0][2]
        print(f"   {K:2d}          {kept:4.0f}%  (test n={nt})         {amir:3.0f}%")
print("ENROLL_DONE")
