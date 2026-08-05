import sys, glob, time, numpy as np, soundfile as sf
sys.path.insert(0, "eval")
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM
from datasets import load_dataset
ev = WakeWordEvaluator("checkpoints/scratch-onnx/ozwelldone_surgical.onnx","eval/pretrained")
rng = np.random.default_rng(0)
NOISE = sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav")) or sorted(glob.glob("/tmp/fphour/*/*.wav"))[:50]
def windows_scores(a):
    mel=ev.mel.run(None,{"input":a[None,:]})[0]; mf=(mel.reshape(-1,32)/10+2).astype("float32")
    if mf.shape[0]<WIN: return None,None
    n=mf.shape[0]; nt=n-(n-WIN)%STRIDE; st=range(0,nt-WIN+1,STRIDE)
    w=np.stack([mf[s:s+WIN] for s in st])[...,None].astype("float32")
    emb=ev.emb.run(None,{"input_1":w})[0].reshape(-1,EMB_DIM).astype("float32")
    if emb.shape[0]<EMB_FRAMES: return None,None
    ws=np.stack([emb[s:s+EMB_FRAMES] for s in range(0,emb.shape[0]-EMB_FRAMES+1)])
    sc=np.array([float(ev.wake.run(None,{"input":ws[i][None]})[0].reshape(-1)[0]) for i in range(len(ws))])
    return ws,sc
def add_noise(y,snr_db):
    if not NOISE: return y
    nz,_=sf.read(NOISE[rng.integers(len(NOISE))],dtype="float32"); nz=nz if nz.ndim==1 else nz.mean(1)
    if len(nz)<len(y): nz=np.tile(nz,len(y)//len(nz)+1)
    nz=nz[:len(y)]; ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
    return (y+nz*np.sqrt(ps/pn/(10**(snr_db/10)))).astype("float32")
fires=[]; nears=[]; n=0; t0=time.time()
ds=load_dataset("facebook/voxpopuli","en",split="train",streaming=True,trust_remote_code=True)
for ex in ds:
    a=ex["audio"]; y=np.asarray(a["array"],dtype="float32"); sr=a["sampling_rate"]
    if sr!=16000 or y.size<sr: continue
    for clip in (y, add_noise(y, rng.uniform(5,18))):   # clean + noise-augmented (realistic, fires more)
        W,S=windows_scores(clip)
        if W is None: continue
        fires.append(W[S>=0.5]); nears.append(W[(S>=0.4)&(S<0.5)])
    n+=1
    if n%200==0:
        nf=sum(len(x) for x in fires); nn=sum(len(x) for x in nears)
        print(f"  {n} clips | {nf} false-fires | {nn} near-misses | {time.time()-t0:.0f}s",flush=True)
        np.save("heybuddy/precalculated/mined_ff_done_vox.npy", np.concatenate([x for x in fires if len(x)]) if nf else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32"))
        np.save("heybuddy/precalculated/mined_nm_done_vox.npy", np.concatenate([x for x in nears if len(x)]) if nn else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32"))
    if n>=20000: break
print("MINE_DONE", n, "clips",flush=True)
