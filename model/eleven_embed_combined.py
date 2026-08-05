#!/usr/bin/env python3
"""Combined DIVERSE ozwell-done positives: pool ElevenLabs accents + cross-lingual native + Voice Design,
mapped into accent groups, balanced to an EQUAL per-accent target. GPU pitch-shift augmentation.
Now the weak accents have 50-109 real voices (not 2-15), so the multiplier is modest -> less overfitting.
Output: precalculated/eleven_combined_pos_done.npy   Prints a BALANCE AUDIT (sources/voices/xAug/positives).
"""
import os, sys, glob
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heybuddy.embeddings import get_speech_embeddings

SR=16000; SEG=int(SR*1.44); TARGET=3000; MAXAUG=45
NOISE=sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav")); rng=np.random.default_rng(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"; print(f"aug device: {DEV}", flush=True)

# accent group -> list of glob patterns across all 3 sources (cross-lingual locales mapped here)
G = lambda *p: [x for pat in p for x in glob.glob(pat)]
ACCENT_WAVS = {
 "japanese": lambda: G("/tmp/eleven_big/train/japanese/ozwell_done/*.wav","/tmp/vd_accents/japanese/ozwell_done/*.wav","/tmp/crosslingual/ja-JP/ozwell_done/*.wav"),
 "korean":   lambda: G("/tmp/eleven_big/train/korean/ozwell_done/*.wav","/tmp/vd_accents/korean/ozwell_done/*.wav","/tmp/crosslingual/ko-KR/ozwell_done/*.wav"),
 "chinese":  lambda: G("/tmp/eleven_big/train/chinese/ozwell_done/*.wav","/tmp/vd_accents/chinese/ozwell_done/*.wav","/tmp/crosslingual/zh-CN/ozwell_done/*.wav"),
 "spanish":  lambda: G("/tmp/eleven_big/train/spanish/ozwell_done/*.wav","/tmp/vd_accents/spanish/ozwell_done/*.wav","/tmp/crosslingual/es-ES/ozwell_done/*.wav"),
 "latin":    lambda: G("/tmp/eleven_big/train/latin/ozwell_done/*.wav","/tmp/vd_accents/latin/ozwell_done/*.wav",
                       "/tmp/crosslingual/es-MX/ozwell_done/*.wav","/tmp/crosslingual/es-AR/ozwell_done/*.wav","/tmp/crosslingual/es-CL/ozwell_done/*.wav",
                       "/tmp/crosslingual/es-CO/ozwell_done/*.wav","/tmp/crosslingual/es-PE/ozwell_done/*.wav","/tmp/crosslingual/es-US/ozwell_done/*.wav","/tmp/crosslingual/es-VE/ozwell_done/*.wav"),
 "filipino": lambda: G("/tmp/eleven_big/train/filipino/ozwell_done/*.wav"),
 "indian":   lambda: G("/tmp/eleven_big/train/indian/ozwell_done/*.wav"),
 "british":  lambda: G("/tmp/eleven_big/train/british/ozwell_done/*.wav"),
 "australian":lambda:G("/tmp/eleven_big/train/australian/ozwell_done/*.wav"),
 "american": lambda: G("/tmp/eleven_big/train/american/ozwell_done/*.wav","/tmp/vd_accents/american/ozwell_done/*.wav"),
}

def fit(a):
    a=a[:SEG] if len(a)>=SEG else np.concatenate([np.zeros((SEG-len(a))//2,"float32"),a,np.zeros(SEG-len(a)-(SEG-len(a))//2,"float32")])
    return a.astype("float32")
def augment(a):
    x=torch.tensor(a,device=DEV); n=int(rng.integers(-3,4))
    if n: x=torchaudio.functional.pitch_shift(x,SR,n)
    rate=float(rng.uniform(0.9,1.12))
    if abs(rate-1)>0.01:
        L=int(len(x)/rate); x=torch.nn.functional.interpolate(x[None,None],size=L,mode="linear",align_corners=False)[0,0]
    y=x.detach().cpu().numpy().astype("float32")
    if NOISE and rng.random()<0.8:
        nz,_=sf.read(NOISE[rng.integers(len(NOISE))],dtype="float32"); nz=nz if nz.ndim==1 else nz.mean(1)
        if len(nz)<len(y): nz=np.tile(nz,len(y)//len(nz)+1)
        nz=nz[:len(y)]; snr=rng.uniform(5,20); ps,pn=(y**2).mean()+1e-9,(nz**2).mean()+1e-9
        y=y+nz*np.sqrt(ps/pn/(10**(snr/10)))
    return fit(y)

emb_model=get_speech_embeddings(device_id=0); auds=[]; audit=[]
for acc,fn in ACCENT_WAVS.items():
    wavs=fn()
    if not wavs: continue
    M=int(np.clip(round(TARGET/len(wavs)),1,MAXAUG)); cnt=0
    for w in wavs:
        a,_=sf.read(w,dtype="float32"); a=a if a.ndim==1 else a.mean(1)
        auds.append(fit(a)); cnt+=1
        for _ in range(M-1): auds.append(augment(a)); cnt+=1
    audit.append((acc,len(wavs),M,cnt)); print(f"  {acc:11s} voices={len(wavs):3d} x{M:2d} -> {cnt}", flush=True)

out=[]
for i in range(0,len(auds),256):
    e=emb_model(auds[i:i+256],spectrogram_batch_size=32,embedding_batch_size=32,remove_nan=False)
    out.append(np.asarray(e,dtype="float32"))
    if i%4096==0: print(f"  embed {i}/{len(auds)}", flush=True)
emb=np.concatenate(out); emb=emb[~np.isnan(emb).any(axis=(1,2))]
np.save("heybuddy/precalculated/eleven_combined_pos_done.npy",emb)
print("\n=== BALANCE AUDIT (accent | voices | xAug | positives) ===", flush=True)
for acc,v,m,c in audit: print(f"  {acc:11s} {v:3d} x{m:2d} {c}", flush=True)
print(f"TOTAL combined accent positives: {emb.shape}", flush=True)
print("EMBED_COMBINED_DONE", flush=True)
