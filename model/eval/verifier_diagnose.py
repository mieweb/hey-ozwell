# DIAGNOSTIC: what axis did the verifier learn — "synthetic vs real" (domain), or "phrase vs not" (content)?
# Train the SAME verifier as verifier_final (synthetic TTS positives + generic negs vs mined REAL false-fires),
# then score 6 probe groups. If SYNTHETIC-non-phrase scores HIGH and REAL-phrase scores LOW => domain confound
# (augmentation is the right lever). If synthetic-non-phrase scores LOW => it IS content-aware on synthetic but
# fails to transfer to real voices (augmentation alone won't save it; need real positives).
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
def fired(f):  # phrase window(s) where pass-1 fired
    W,S=winsc(load_16k_mono(f)); return W[S>=0.5] if W is not None and (S>=0.5).any() else (W[[S.argmax()]] if W is not None else np.zeros((0,EMB_FRAMES,EMB_DIM),"float32"))
def take(p,k):
    a=np.load(P+p,mmap_mode="r"); idx=rng.choice(len(a),min(k,len(a)),replace=False); return np.asarray(a[np.sort(idx)])[:,:EMB_FRAMES,:]
flat=lambda a:a.reshape(len(a),-1)

# ---- TRAIN the verifier exactly as the failed final one: synthetic TTS pos + generic neg vs mined REAL FF ----
tts=take("ozwell_i_m_done.npy",8000)            # synthetic Piper phrase  (POSITIVE)
gen=take("negs_C.npy",8000)                     # real-human generic      (NEGATIVE)
mined=np.concatenate([np.load(P+"mined_false_fires.npy"),np.load(P+"mined_ff_done_vox.npy")])  # mined REAL false-fires (NEG)
X=np.concatenate([flat(tts),flat(gen),flat(mined)]); y=np.concatenate([np.ones(len(tts)),np.zeros(len(gen)+len(mined))])
m=MLPClassifier(hidden_layer_sizes=(256,64),max_iter=2000,early_stopping=True).fit(X,y)
sc=lambda a: m.predict_proba(flat(a))[:,1] if len(a) else np.array([np.nan])

# ---- PROBE GROUPS ----
real_phrase=fired("../../real_audio/Oz-done.wav")                                   # REAL voice, IS phrase  [want HIGH]
syn_phrase_piper=take("diverse_pos.npy",2000)                                       # SYNTH, IS phrase
syn_phrase_eleven=take("eleven_surgical_pos_done.npy",2000)                         # SYNTH (other engine), IS phrase
syn_nonphrase=take("confusable_negs.npy",2000)                                      # SYNTH, NOT phrase (confusables) [KEY]
real_nonphrase=np.load(P+"mined_false_fires.npy")[:2000]                            # REAL, NOT phrase (false-fires)
groups=[("REAL  phrase  (your voice)   [want HIGH]",real_phrase),
        ("SYNTH phrase  (Piper, in-dist)",          syn_phrase_piper),
        ("SYNTH phrase  (ElevenLabs, x-engine)",    syn_phrase_eleven),
        ("SYNTH NON-phrase (confusables) [KEY]",    syn_nonphrase),
        ("REAL  NON-phrase (false-fires)",          real_nonphrase)]
print(f"\n### DIAGNOSTIC: verifier mean P(accept) per group  (train: {len(tts)} synth-pos / {len(gen)+len(mined)} neg) ###")
for name,g in groups:
    s=sc(g); print(f"  {name:42s} n={len(g):4d}  meanP={np.nanmean(s):.3f}  accept@0.5={np.nanmean(s>=0.5)*100:4.0f}%")
print("\nINTERPRET: SYNTH-NON-phrase HIGH + REAL-phrase LOW => domain confound (augment to fix).")
print("           SYNTH-NON-phrase LOW  + REAL-phrase LOW => content-aware on synth, no real transfer (need real positives).")

# ---- bootstrap CI on the real-phrase accept rate (how unreliable is the headline number?) ----
rp=sc(real_phrase)
if len(rp)>1:
    bs=[ (rng.choice(rp,len(rp),replace=True)>=0.5).mean() for _ in range(2000) ]
    lo,hi=np.percentile(bs,[2.5,97.5])
    print(f"\nReal-phrase accept@0.5 = {(rp>=0.5).mean()*100:.0f}%  (95% bootstrap CI {lo*100:.0f}-{hi*100:.0f}%, n={len(rp)} windows from ONE recording)")
print("DIAGNOSE_DONE")
