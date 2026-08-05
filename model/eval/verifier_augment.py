# AUGMENTATION EXPERIMENT: can making synthetic positives SOUND like real mic recordings (room reverb +
# real background noise) lift the verifier's real-wake recall, WITHOUT collecting real recordings?
# Positives = ElevenLabs phrase wavs (cross-engine synthetic), each embedded clean + reverb+noise augmented.
# Negatives = mined REAL false-fires + generic. TEST = held-out REAL wakes (Oz-done) + AMI false-fires.
# Compare real-recall vs the synthetic-only baseline (~17%).
import sys, glob, numpy as np, soundfile as sf
sys.path.insert(0,".")
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM
from sklearn.neural_network import MLPClassifier
ev=WakeWordEvaluator("../checkpoints/scratch-onnx/ozwelldone_surgical.onnx","pretrained"); rng=np.random.default_rng(0)
P="../heybuddy/precalculated/"; NOISE=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))

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
def best_win(a):  # the highest-scoring window of a positive clip (the phrase)
    W,S=winsc(a); return W[[S.argmax()]] if W is not None else None
def reverb(y):    # cheap synthetic room: exponential-decay impulse
    ir=np.exp(-np.arange(int(0.05*16000))/ (0.02*16000)).astype("float32"); ir[0]=1.0
    return np.convolve(y,ir)[:len(y)].astype("float32")
def add_noise(y):
    if not NOISE: return y
    nz,_=sf.read(NOISE[rng.integers(len(NOISE))],dtype="float32"); nz=nz if nz.ndim==1 else nz.mean(1)
    if len(nz)<len(y): nz=np.tile(nz,len(y)//len(nz)+1)
    nz=nz[:len(y)]; ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
    return (y+nz*np.sqrt(ps/pn/(10**(rng.uniform(5,15)/10)))).astype("float32")

def embed_pos(wavs, augment):
    out=[]
    for f in wavs:
        y=load_16k_mono(f)
        clips=[y] if not augment else [y, add_noise(y), reverb(add_noise(y)), add_noise(reverb(y))]
        for c in clips:
            w=best_win(c)
            if w is not None: out.append(w)
    return np.concatenate(out) if out else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32")
def mine_real(d):  # AMI false-fires (held-out negatives), noise-aug to grow n
    out=[]
    for f in sorted(glob.glob(f"{d}/*.wav")):
        y=load_16k_mono(f)
        for c in (y,add_noise(y),add_noise(y)):
            W,S=winsc(c)
            if W is not None: out.append(W[S>=0.5])
    return np.concatenate(out) if out else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32")
def take(p,k):
    a=np.load(P+p,mmap_mode="r"); idx=np.sort(rng.choice(len(a),min(k,len(a)),replace=False)); return np.asarray(a[idx])[:,:EMB_FRAMES,:]
flat=lambda a:a.reshape(len(a),-1)

# held-out REAL wakes (never in training)
rp,_=winsc(load_16k_mono("../../real_audio/Oz-done.wav")); rs=np.array([float(ev.wake.run(None,{"input":rp[i][None]})[0].reshape(-1)[0]) for i in range(len(rp))]); real_wakes=rp[rs>=0.5]
ami=mine_real("/tmp/fphour/thirdparty_ami")
mined=np.concatenate([np.load(P+"mined_false_fires.npy"),np.load(P+"mined_ff_done_vox.npy")])
gen=take("negs_C.npy",6000)
el=sorted(glob.glob("/tmp/eleven_big/train/*/ozwell_done/*.wav"))[:900]
print(f"eleven positive wavs: {len(el)} | held-out real wakes: {len(real_wakes)} | AMI test FF: {len(ami)}",flush=True)

def run(tag, pos):
    X=np.concatenate([flat(pos),flat(gen),flat(mined)]); y=np.concatenate([np.ones(len(pos)),np.zeros(len(gen)+len(mined))])
    m=MLPClassifier(hidden_layer_sizes=(256,64),max_iter=2000,early_stopping=True).fit(X,y)
    pr=m.predict_proba(flat(real_wakes))[:,1]; pa=m.predict_proba(flat(ami))[:,1]
    print(f"\n[{tag}] pos n={len(pos)}")
    print("   t   real-kept  AMI-rejected")
    for t in [0.3,0.5,0.7]:
        print(f"  {t:.1f}    {(pr>=t).mean()*100:4.0f}%      {(pa<t).mean()*100:4.0f}%")
    return m

print("embedding CLEAN eleven positives...",flush=True);  pos_clean=embed_pos(el, augment=False)
run("BASELINE: synthetic positives, NO augmentation", pos_clean)
print("\nembedding AUGMENTED eleven positives (reverb+real-noise)...",flush=True); pos_aug=embed_pos(el, augment=True)
run("AUGMENTED: synthetic positives + reverb/noise -> real-mic", pos_aug)
print("AUGMENT_DONE")
