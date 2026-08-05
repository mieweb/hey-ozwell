#!/usr/bin/env python3
"""Targeted confusable NEGATIVES with REALISTIC voices (ElevenLabs, cross-vendor) — not Piper.
Diagnosed from Jonathan's real recording: false fires cluster on "as well"/"well"/"done" and
coincidental cadence ("labs from last week"). The old confusable_negs were Piper-synthetic, so the
"as well" didn't match REAL speech -> didn't transfer. Regenerate in real voices + clinical contexts.
Generate -> light aug -> embed -> precalculated/confusable_targeted_negs.npy  ([n,16,96]).
"""
import os, sys, glob
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elevenlabs import ElevenLabs
from heybuddy.embeddings import get_speech_embeddings

SR=16000; SEG=int(SR*1.44); AUG=2
NOISE=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav")); rng=np.random.default_rng(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"
OUTDIR="/tmp/conf_targeted"; os.makedirs(OUTDIR, exist_ok=True)

# the actual collisions, in varied + clinical sentence contexts
SENTENCES=[
 "that worked as well","she did really well on the exam","we finished the labs last week",
 "that went well overall","I'm done with this part","we're all done here","the results came back as well",
 "he performed well in the study","oh well, let's move on","as well as the other findings",
 "the patient is done with treatment","I reviewed the labs from last week","it healed well",
 "the swelling went down as well","that's done as well","we'll do the rest as well",
 "well, the vitals look stable","the wound is healing well","I'm almost done with the charts",
 "farewell for now","the nozzle was clogged","everything else is done","as well as expected",
 "the cells were well differentiated","that medication works well","are we done with the exam",
 "the scan from last week was clear","not as well as we hoped","all done with the paperwork",
 "his labs from last week are back",
]
ACCENTS=["american","indian","british","latin","spanish","chinese","australian"]

def key(): return open(os.path.expanduser("~/.eleven_api_key")).read().strip()
def fit(a):
    a=a[:SEG] if len(a)>=SEG else np.concatenate([np.zeros((SEG-len(a))//2,"float32"),a,np.zeros(SEG-len(a)-(SEG-len(a))//2,"float32")])
    return a.astype("float32")
def augment(a):
    x=torch.tensor(a,device=DEV); n=int(rng.integers(-2,3))
    if n: x=torchaudio.functional.pitch_shift(x,SR,n)
    y=x.detach().cpu().numpy().astype("float32")
    if NOISE and rng.random()<0.5:
        nz,_=sf.read(NOISE[rng.integers(len(NOISE))],dtype="float32"); nz=nz if nz.ndim==1 else nz.mean(1)
        if len(nz)<len(y): nz=np.tile(nz,len(y)//len(nz)+1)
        nz=nz[:len(y)]; snr=rng.uniform(8,22); ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
        y=y+nz*np.sqrt(ps/pn/(10**(snr/10)))
    return fit(y)

c=ElevenLabs(api_key=key())
# pull ~5 voices per accent for realistic diversity
voices=[]
for acc in ACCENTS:
    try: voices += [(acc,v.voice_id) for v in c.voices.get_shared(page_size=5,accent=acc,language="en").voices]
    except Exception as e: print(f"[{acc}] list fail {e}",flush=True)
print(f"{len(voices)} voices x {len(SENTENCES)} sentences",flush=True)

made=0
for acc,vid in voices:
    for s in SENTENCES:
        p=f"{OUTDIR}/{vid}_{abs(hash(s))%10000}.wav"
        if os.path.exists(p): continue
        try:
            au=b"".join(c.text_to_speech.convert(vid,text=s,model_id="eleven_multilingual_v2",output_format="wav_16000"))
            open(p,"wb").write(au); made+=1
        except Exception as e:
            if "quota_exceeded" in str(getattr(e,"body",e)) or getattr(e,"status_code",None)==401:
                print(f"\nQUOTA EXHAUSTED after {made}. embedding what we have.",flush=True); break
            print(f"  {vid[:8]} fail {str(e)[:50]}",flush=True)
    else: continue
    break
print(f"generated {made} clips; embedding...",flush=True)

emb_model=get_speech_embeddings(device_id=0); auds=[]
for w in sorted(glob.glob(f"{OUTDIR}/*.wav")):
    a,_=sf.read(w,dtype="float32"); a=a if a.ndim==1 else a.mean(1)
    auds.append(fit(a))
    for _ in range(AUG): auds.append(augment(a))
out=[]
for i in range(0,len(auds),256):
    e=emb_model(auds[i:i+256],spectrogram_batch_size=32,embedding_batch_size=32,remove_nan=False)
    out.append(np.asarray(e,dtype="float32"))
emb=np.concatenate(out); emb=emb[~np.isnan(emb).any(axis=(1,2))]
np.save("heybuddy/precalculated/confusable_targeted_negs.npy",emb)
print(f"confusable_targeted_negs.npy {emb.shape} (from {made} real clips x{AUG+1})",flush=True)
print("CONF_TARGETED_DONE",flush=True)
