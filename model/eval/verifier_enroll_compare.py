# HEAD-TO-HEAD: enrollment via SIMILARITY vs TRAINING, on real captured data.
# Enroll on K=5 of Jonathan's reps; test on held-out reps (want kept HIGH) + junk (want rejected HIGH).
#  - SIMILARITY: accept if cosine to nearest enrolled rep >= threshold. No training, no negatives.
#  - TRAINING:   tiny MLP on K enrolled reps + BUNDLED synthetic negatives (what we'd ship).
# Two junk test sets: captured near-misses (HARD) + held-out synthetic conversation (EASY).
# Fair comparison: for each method, pick its threshold to keep ~90% of held-out real wakes, then compare rejection.
import sys, glob, json, gzip, numpy as np
sys.path.insert(0,".")
from evaluate_wakeword import load_16k_mono
from browser_embed import stream_embeddingbuffers
import onnxruntime as ort, soundfile as sf
from sklearn.neural_network import MLPClassifier
rng=np.random.default_rng(0)
wake=ort.InferenceSession("../checkpoints/scratch-onnx/ozwelldone_surgical.onnx",providers=["CPUExecutionProvider"])
def ws(eb): return float(wake.run(None,{"input":eb[None].astype("float32")})[0].reshape(-1)[0])

# captured real data (browser format)
cap=json.load(gzip.open("../captures/verifier-capture.json.gz"))
capd=cap["ozwell-i'm-done"] if "ozwell-i'm-done" in cap else cap
POS=np.array(capd["pos"],dtype="float32"); JUNK_HARD=np.array(capd["neg"],dtype="float32")  # near-misses
print(f"captured: {len(POS)} real wakes | {len(JUNK_HARD)} near-miss junk",flush=True)

# bundled synthetic negatives (browser-faithful, noise-augmented) = what we'd ship + an easy-junk test split
NOISE=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))
def addn(y):
    nz,_=sf.read(NOISE[rng.integers(len(NOISE))],dtype="float32"); nz=nz if nz.ndim==1 else nz.mean(1)
    if len(nz)<len(y): nz=np.tile(nz,len(y)//len(nz)+1)
    nz=nz[:len(y)]; ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
    return (y+nz*np.sqrt(ps/pn/(10**(rng.uniform(0,12)/10)))).astype("float32")
import os
CACHE="/tmp/enroll_conv_negs.npy"
if os.path.exists(CACHE):
    SYN=np.load(CACHE); print(f"loaded cached conversational negs: {len(SYN)}",flush=True)
else:
    print("generating CONVERSATIONAL negatives (People's + AMI, browser-faithful, noise-aug)...",flush=True)
    SYN=[]
    clips=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))[:160]+sorted(glob.glob("/tmp/fphour/thirdparty_ami/*.wav"))[:60]
    for w in clips:
        y=load_16k_mono(w)
        for c in (y,addn(y),addn(y)):
            SYN+=[e for e in stream_embeddingbuffers(c) if ws(e)>=0.5]
        if len(SYN)>=900: break
    SYN=np.array(SYN[:900],dtype="float32"); np.save(CACHE,SYN)
i=rng.permutation(len(SYN)); SYN_TR=SYN[i[150:]]; JUNK_EASY=SYN[i[:150]]  # bundled-train vs a real-sized conversation test
print(f"conversational negs: {len(SYN_TR)} bundled-train | {len(JUNK_EASY)} conversation test\n",flush=True)

flat=lambda a:a.reshape(len(a),-1)
def norm(a): a=flat(a); return a/ (np.linalg.norm(a,axis=1,keepdims=True)+1e-9)
def kept_at_thr(scores,thr): return (scores>=thr).mean()*100
def rej_at_thr(scores,thr): return (scores<thr).mean()*100

def run_split(K):
    idx=rng.permutation(len(POS)); enroll=POS[idx[:K]]; test_pos=POS[idx[K:]]
    out={}
    # ---- SIMILARITY ----
    en=norm(enroll);
    sim=lambda X: (norm(X)@en.T).max(axis=1)   # max cosine to any enrolled rep
    s_pos,s_he,s_hard=sim(test_pos),sim(JUNK_EASY),sim(JUNK_HARD)
    thr=np.quantile(s_pos,0.10)                # threshold that keeps ~90% of held-out real wakes
    out["sim"]=(kept_at_thr(s_pos,thr),rej_at_thr(s_he,thr),rej_at_thr(s_hard,thr))
    # ---- TRAINING (enrolled reps + bundled synthetic negs) ----
    X=np.concatenate([flat(enroll),flat(SYN_TR)]); y=np.concatenate([np.ones(K),np.zeros(len(SYN_TR))])
    m=MLPClassifier(hidden_layer_sizes=(32,),alpha=1.0,max_iter=3000,random_state=0).fit(X,y)
    p=lambda X: m.predict_proba(flat(X))[:,1]
    p_pos,p_he,p_hard=p(test_pos),p(JUNK_EASY),p(JUNK_HARD)
    thr=np.quantile(p_pos,0.10)
    out["train"]=(kept_at_thr(p_pos,thr),rej_at_thr(p_he,thr),rej_at_thr(p_hard,thr))
    return out

print("K = enrolled embedding WINDOWS (each spoken rep ~= 6-7 windows, so 3 spoken reps ~= 20 windows)\n")
for K in (3,5,10,20):
    res=[run_split(K) for _ in range(8)]
    for meth in ("sim","train"):
        a=np.array([r[meth] for r in res]).mean(axis=0)
        label="SIMILARITY" if meth=="sim" else "TRAINING  "
        print(f"K={K:2d} win {label}: real-wake kept {a[0]:3.0f}% | conversation rejected {a[1]:3.0f}% | near-miss rejected {a[2]:3.0f}%")
    print()
print("(both tuned to ~90% real-wake retention; 3 spoken reps ~= K=20 windows)")
print("COMPARE_DONE")
