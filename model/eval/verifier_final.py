# FINAL model: train on ALL diverse sources (People's + VoxPopuli + AMI), hold out only the
# recording (most representative) + a random slice of each source (in-distribution check).
# Run this ONLY after the held-out-AMI probe confirms generalization. Positives = TTS (synthetic is
# fine per Jonathan); ALL real wakes held out for the test (representative positive check).
import sys, glob, numpy as np, soundfile as sf
sys.path.insert(0,".")
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM
from sklearn.neural_network import MLPClassifier
ev=WakeWordEvaluator("../checkpoints/scratch-onnx/ozwelldone_surgical.onnx","pretrained"); rng=np.random.default_rng(0)
NOISE=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))
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
def addn(y):
    if not NOISE: return y
    nz,_=sf.read(NOISE[rng.integers(len(NOISE))],dtype="float32"); nz=nz if nz.ndim==1 else nz.mean(1)
    if len(nz)<len(y): nz=np.tile(nz,len(y)//len(nz)+1)
    nz=nz[:len(y)]; ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
    return (y+nz*np.sqrt(ps/pn/(10**(rng.uniform(5,18)/10)))).astype("float32")
def mine_dir(d):
    out=[]
    for f in sorted(glob.glob(f"{d}/*.wav")):
        y=load_16k_mono(f)
        for clip in (y,addn(y),addn(y)):
            W,S=winsc(clip)
            if W is not None: out.append(W[S>=0.5])
    return np.concatenate(out) if out else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32")
def split(a,f=0.2):
    i=rng.permutation(len(a)); k=int(len(a)*f); return a[i[k:]],a[i[:k]]  # train, test
# ALL sources mined
ppl=np.load("../heybuddy/precalculated/mined_false_fires.npy")
vox=np.load("../heybuddy/precalculated/mined_ff_done_vox.npy") if glob.glob("../heybuddy/precalculated/mined_ff_done_vox.npy") else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32")
ami=mine_dir("/tmp/fphour/thirdparty_ami")
# 80/20 split each -> in-distribution held-out test
ppl_tr,ppl_te=split(ppl); vox_tr,vox_te=split(vox); ami_tr,ami_te=split(ami)
hn_tr=np.concatenate([ppl_tr,vox_tr,ami_tr]); hn_te=np.concatenate([ppl_te,vox_te,ami_te])
# recording: ALL real wakes (test) + recording false-fires (test) — fully held out, representative
rp,_=winsc(load_16k_mono("../../real_audio/Oz-done.wav")); rs=np.array([float(ev.wake.run(None,{"input":rp[i][None]})[0].reshape(-1)[0]) for i in range(len(rp))]); rp=rp[rs>=0.5]
you=np.concatenate([mine_dir(f.rsplit("/",1)[0]) for f in []] or [np.zeros((0,EMB_FRAMES,EMB_DIM),"float32")])
you=np.concatenate([ (lambda W_S:(W_S[0][W_S[1]>=0.5] if W_S[0] is not None else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32")))(winsc(load_16k_mono(f))) for f in glob.glob("../../real_audio/*.wav") if "Oz-done" not in f and "Hey-oz" not in f])
tts=np.asarray(np.load("../heybuddy/precalculated/ozwell_i_m_done.npy",mmap_mode="r")[rng.choice(127000,8000,replace=False)])
gen=np.asarray(np.load("../heybuddy/precalculated/negs_C.npy",mmap_mode="r")[rng.choice(160000,8000,replace=False)])[:,:EMB_FRAMES,:]
flat=lambda a:a.reshape(len(a),-1)
X=np.concatenate([flat(tts),flat(gen),flat(hn_tr)]); y=np.concatenate([np.ones(len(tts)),np.zeros(len(gen)+len(hn_tr))])
print(f"FINAL train hard-negs: People's {len(ppl_tr)} + VoxPopuli {len(vox_tr)} + AMI {len(ami_tr)} = {len(hn_tr)}")
print(f"held-out test: in-dist {len(hn_te)} | recording FF {len(you)} | real wakes {len(rp)}")
m=MLPClassifier(hidden_layer_sizes=(256,64),max_iter=2000,early_stopping=True).fit(X,y)
P=lambda a: m.predict_proba(flat(a))[:,1] if len(a) else np.array([])
Pr,Pin,Pyou=P(rp),P(hn_te),P(you)
print("   t   real-kept  in-dist-rej  recording-rej")
for t in [0.3,0.4,0.5,0.6]:
    print(f"  {t:.1f}    {(Pr>=t).mean()*100:4.0f}%     {(Pin<t).mean()*100:4.0f}%        {(Pyou<t).mean()*100:4.0f}%")
print("FINAL_DONE")
