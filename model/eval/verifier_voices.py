# PROPER eval: train verifier positives on eleven_big/TRAIN voices, test real-kept on eleven_big/TEST voices
# (DISJOINT voices, hundreds, stable) + your-recording as a single real-world anchor. Negatives = mined REAL
# FF + generic; test negatives = AMI (independent). 5 seeds, report mean+range. Resolves the 53-window noise
# AND the per-speaker overfitting concern.
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
def embed_dir(wavs):
    out=[]
    for f in wavs:
        w=best_win(load_16k_mono(f))
        if w is not None: out.append(w)
    return np.concatenate(out) if out else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32")
def L(p,k=None):
    a=np.load(P+p,mmap_mode="r"); idx=np.sort(rng.choice(len(a),min(k,len(a)),replace=False)) if k else np.arange(len(a)); return np.asarray(a[idx])[:,:EMB_FRAMES,:]
flat=lambda a:a.reshape(len(a),-1)
def add_noise(y):
    import soundfile as sf; NZ=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))
    nz,_=sf.read(NZ[rng.integers(len(NZ))],dtype="float32"); nz=nz if nz.ndim==1 else nz.mean(1)
    if len(nz)<len(y): nz=np.tile(nz,len(y)//len(nz)+1)
    nz=nz[:len(y)]; ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
    return (y+nz*np.sqrt(ps/pn/(10**(rng.uniform(5,15)/10)))).astype("float32")

tr_wavs=sorted(glob.glob("/tmp/eleven_big/train/*/ozwell_done/*.wav"))
te_wavs=sorted(glob.glob("/tmp/eleven_big/test/*/ozwell_done/*.wav"))
print(f"embedding train voices ({len(tr_wavs)}) + test voices ({len(te_wavs)}) ...",flush=True)
pos_tr=embed_dir(tr_wavs)            # TRAIN positives (train voices)
pos_te=embed_dir(te_wavs)           # TEST positives (DISJOINT test voices) = many-voice metric
real,_=winsc(load_16k_mono("../../real_audio/Oz-done.wav")); rs=np.array([float(ev.wake.run(None,{"input":real[i][None]})[0].reshape(-1)[0]) for i in range(len(real))]); real=real[rs>=0.5]
ami=[]
for f in sorted(glob.glob("/tmp/fphour/thirdparty_ami/*.wav")):
    y=load_16k_mono(f)
    for c in (y,add_noise(y),add_noise(y)):
        W,S=winsc(c)
        if W is not None: ami.append(W[S>=0.5])
ami=np.concatenate(ami)
gen=L("negs_C.npy",6000); mined=np.concatenate([np.load(P+"mined_false_fires.npy"),np.load(P+"mined_ff_done_vox.npy")])
# per-accent breakdown of test voices
acc_idx={}; i=0
for f in te_wavs:
    w=best_win(load_16k_mono(f))
    if w is not None:
        a=f.split("/test/")[1].split("/")[0]; acc_idx.setdefault(a,[]).append(i); i+=1
print(f"\nTRAIN pos {len(pos_tr)} | TEST pos (disjoint voices) {len(pos_te)} | your-recording {len(real)} | AMI {len(ami)} | neg {len(gen)+len(mined)}\n",flush=True)
X0=np.concatenate([flat(pos_tr),flat(gen),flat(mined)]); y0=np.concatenate([np.ones(len(pos_tr)),np.zeros(len(gen)+len(mined))])
def stat(arr): return f"{np.mean(arr):3.0f}% (range {min(arr):.0f}-{max(arr):.0f})"
tv,rv,av={},[],[]
seedscores=[]
for s in range(5):
    m=MLPClassifier(hidden_layer_sizes=(256,64),max_iter=2000,early_stopping=True,random_state=s).fit(X0,y0)
    Pte=m.predict_proba(flat(pos_te))[:,1]; Pr=m.predict_proba(flat(real))[:,1]; Pa=m.predict_proba(flat(ami))[:,1]
    seedscores.append(((Pte>=.5).mean()*100,(Pr>=.5).mean()*100,(Pa<.5).mean()*100,Pte))
tvs=[s[0] for s in seedscores]; rvs=[s[1] for s in seedscores]; avs=[s[2] for s in seedscores]
print("=== @ threshold 0.5, across 5 seeds ===")
print(f"  TEST-VOICE kept (many disjoint voices): {stat(tvs)}")
print(f"  YOUR-RECORDING kept (1 speaker anchor): {stat(rvs)}")
print(f"  AMI false-fires rejected (independent): {stat(avs)}")
# operating curve: conservative veto thresholds (only suppress when verifier very confident it's junk)
print("\n=== VETO operating curve (avg 5 seeds): suppress fire if verifier P < t ===")
print("   t    test-voice kept   your-rec kept   FALSE-FIRES killed")
allP=[]
for s in range(5):
    m=MLPClassifier(hidden_layer_sizes=(256,64),max_iter=2000,early_stopping=True,random_state=s).fit(X0,y0)
    allP.append((m.predict_proba(flat(pos_te))[:,1],m.predict_proba(flat(real))[:,1],m.predict_proba(flat(ami))[:,1]))
for t in [0.05,0.1,0.15,0.2,0.3,0.5]:
    tv=np.mean([(p[0]>=t).mean() for p in allP])*100
    rv=np.mean([(p[1]>=t).mean() for p in allP])*100
    av=np.mean([(p[2]<t).mean() for p in allP])*100
    print(f"  {t:.2f}      {tv:4.0f}%           {rv:4.0f}%            {av:4.0f}%")
# per-accent on last seed
Pte=seedscores[-1][3]
print("\n  per-accent test-voice kept@0.5:")
for a,idx in sorted(acc_idx.items()):
    print(f"    {a:12s} n={len(idx):3d}  {(Pte[idx]>=.5).mean()*100:3.0f}%")
print("VOICES_DONE")
