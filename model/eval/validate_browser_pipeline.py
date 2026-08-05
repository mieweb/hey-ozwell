# VALIDATE the browser-faithful embedder: train a verifier on browser-faithful SYNTHETIC embeddings, then
# test it on the user's CAPTURED real-browser embeddings. If captured real wakes score HIGH and captured
# junk scores LOW, the offline pipeline matches the browser -> we can mass-generate diverse data and ship.
import sys, glob, json, gzip, numpy as np, onnxruntime as ort
sys.path.insert(0,".")
from evaluate_wakeword import load_16k_mono
from browser_embed import stream_embeddingbuffers
from sklearn.neural_network import MLPClassifier
rng=np.random.default_rng(0)
wake=ort.InferenceSession("../checkpoints/scratch-onnx/ozwelldone_surgical.onnx",providers=["CPUExecutionProvider"])
def wscore(eb): return float(wake.run(None,{"input":eb[None].astype("float32")})[0].reshape(-1)[0])

def clip_pos(wav):
    """Browser-faithful POSITIVE: the highest-firing embeddingBuffer of a phrase clip."""
    ebs=list(stream_embeddingbuffers(load_16k_mono(wav)))
    if not ebs: return None
    sc=[wscore(e) for e in ebs]; return ebs[int(np.argmax(sc))]
import soundfile as sf
_NOISE=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))
def _add_noise(y):
    nz,_=sf.read(_NOISE[rng.integers(len(_NOISE))],dtype="float32"); nz=nz if nz.ndim==1 else nz.mean(1)
    if len(nz)<len(y): nz=np.tile(nz,len(y)//len(nz)+1)
    nz=nz[:len(y)]; ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
    return (y+nz*np.sqrt(ps/pn/(10**(rng.uniform(0,12)/10)))).astype("float32")
def clip_negs(wav,maxn=12):
    """Browser-faithful NEGATIVES: firing windows from non-phrase speech, NOISE-AUGMENTED so the wake model
    false-fires (clean speech barely fires; noise makes it fire ~40% -> harvest hard negatives, like the miner)."""
    y=load_16k_mono(wav); out=[]
    for clip in (y,_add_noise(y),_add_noise(y)):
        out+=[e for e in stream_embeddingbuffers(clip) if wscore(e)>=0.5]
    return out[:maxn]

print("generating browser-faithful SYNTHETIC training data...",flush=True)
pos_wavs=sorted(glob.glob("/tmp/eleven_big/train/*/ozwell_done/*.wav"))[:300]
POS=np.stack([p for p in (clip_pos(w) for w in pos_wavs) if p is not None])
neg_wavs=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))[:120]
NEG=[]
for w in neg_wavs:
    NEG+=clip_negs(w)
    if len(NEG)>=1500: break
NEG=np.stack(NEG) if NEG else np.zeros((0,16,96),"float32")
print(f"browser-faithful synthetic: {len(POS)} pos / {len(NEG)} neg",flush=True)

# CAPTURED real-browser embeddings = held-out TEST
cap=json.load(gzip.open("../captures/verifier-capture.json.gz")) if glob.glob("../captures/verifier-capture.json.gz") else json.load(open("../captures/verifier-capture.json"))
# capture is either {pos,neg} (old) or {word:{pos,neg}} (new)
capd = cap["ozwell-i'm-done"] if "ozwell-i'm-done" in cap else cap
cap_pos=np.array(capd["pos"],dtype="float32").reshape(-1,16,96)
cap_neg=np.array(capd["neg"],dtype="float32").reshape(-1,16,96)
print(f"captured real-browser TEST: {len(cap_pos)} pos / {len(cap_neg)} neg\n",flush=True)

flat=lambda a:a.reshape(len(a),-1)
X=np.concatenate([flat(POS),flat(NEG)]); y=np.concatenate([np.ones(len(POS)),np.zeros(len(NEG))])
m=MLPClassifier(hidden_layer_sizes=(32,),alpha=1.0,max_iter=4000,random_state=0).fit(X,y)
Pp=m.predict_proba(flat(cap_pos))[:,1]; Pn=m.predict_proba(flat(cap_neg))[:,1]
print("=== verifier trained on browser-faithful SYNTHETIC, tested on CAPTURED real-browser ===")
print("   t    captured real-wake kept   captured junk rejected")
for t in (0.3,0.5,0.65):
    print(f"  {t:.2f}      {(Pp>=t).mean()*100:4.0f}%                  {(Pn<t).mean()*100:4.0f}%")
print(f"\n  mean P: captured real wakes {Pp.mean():.3f}  | captured junk {Pn.mean():.3f}")
print("  MATCH if real-wake kept is HIGH -> offline pipeline reproduces the browser; can mass-generate.")
print("VALIDATE_DONE")
