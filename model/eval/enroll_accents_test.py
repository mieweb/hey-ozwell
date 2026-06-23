# Multi-accent enrollment generalization (synthetic voices stand in for "other people").
# Each ElevenLabs voice has 2 utterances (s1, s2). Per voice: ENROLL on s1's windows, then test whether
# the held-out s2 utterance matches (cosine) — that's "enroll on one saying, recognize another saying by the
# same person" — and whether conversation junk is rejected. Aggregated per accent across many voices.
# Tests the accent-generalization MECHANISM of similarity enrollment, which we can't test with real people.
import sys, glob, os, re, numpy as np
sys.path.insert(0,".")
from evaluate_wakeword import load_16k_mono
from browser_embed import stream_embeddingbuffers
import onnxruntime as ort
wake=ort.InferenceSession("../checkpoints/scratch-onnx/ozwelldone_surgical.onnx",providers=["CPUExecutionProvider"])
def ws(eb): return float(wake.run(None,{"input":eb[None].astype("float32")})[0].reshape(-1)[0])
def l2(a): a=a.reshape(len(a),-1); return a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-9)
def fired(wav):
    ebs=list(stream_embeddingbuffers(load_16k_mono(wav)))
    if not ebs: return None,None
    sc=[ws(e) for e in ebs]; W=np.array([ebs[i].reshape(-1) for i in range(len(ebs)) if sc[i]>=0.5],dtype="float32")
    peak=ebs[int(np.argmax(sc))].reshape(-1).astype("float32")
    return (W if len(W) else peak[None]), peak

CONV=np.load("/tmp/enroll_conv_negs.npy") if os.path.exists("/tmp/enroll_conv_negs.npy") else None
CONV=l2(CONV.reshape(len(CONV),-1)) if CONV is not None else None
NPER=15  # voices per accent (cap for speed)
accents=sorted(glob.glob("/tmp/eleven_big/test/*/ozwell_done"))
print(f"conversation junk windows: {0 if CONV is None else len(CONV)} | up to {NPER} voices/accent\n",flush=True)

# build per-voice (s1 enroll windows, s2 peak query), grouped by voice_id from filename "<id>_s{1,2}.wav"
def by_voice(d):
    g={}
    for f in sorted(glob.glob(f"{d}/*.wav")):
        m=re.match(r"(.+)_s\d+\.wav$", os.path.basename(f))
        if m: g.setdefault(m.group(1),[]).append(f)
    return {k:v for k,v in g.items() if len(v)>=2}

# pick a global threshold that keeps ~90% retention across all voices, then report per accent
rows=[]; allpos=[]
data={}
for d in accents:
    acc=d.split("/test/")[1].split("/")[0]
    voices=list(by_voice(d).items())[:NPER]
    ent=[]
    for vid,fs in voices:
        Wenr,_=fired(fs[0]); _,q=fired(fs[1])
        if Wenr is None or q is None: continue
        T=l2(Wenr); qn=q/ (np.linalg.norm(q)+1e-9)
        pos=float((T@qn).max())                     # does held-out utterance match the enrolled one?
        ent.append((T,pos))
    data[acc]=ent; allpos+=[p for _,p in ent]
    print(f"  {acc:12s}: {len(ent)} voices embedded",flush=True)

thr=np.quantile(allpos,0.10) if allpos else 0.5     # keep ~90% real retention
print(f"\nglobal threshold (90% retention) = {thr:.3f}\n")
print(f"{'accent':12s}  real-utt kept   conversation rejected")
for acc,ent in data.items():
    if not ent: continue
    kept=np.mean([p>=thr for _,p in ent])*100
    if CONV is not None:
        rej=np.mean([ (T@CONV.T).max(axis=0).mean()<thr for T,_ in ent])  # rough: conv junk vs each enrolled voice
        # better: fraction of conv windows below thr, averaged over voices
        rejs=[ ( (CONV@T.T).max(axis=1) < thr ).mean() for T,_ in ent ]
        rej=np.mean(rejs)*100
    else: rej=float("nan")
    print(f"{acc:12s}   {kept:4.0f}%            {rej:4.0f}%")
print("\n(enroll on ONE synthetic utterance/voice, recognize a HELD-OUT utterance of the same voice; junk=real conversation)")
print("ENROLL_ACCENTS_DONE")
