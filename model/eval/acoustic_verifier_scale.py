#!/usr/bin/env python3
"""Acoustic verifier — scaled, cross-corpus honest test.
Mine false-fire embeddings (windows where pass-1 fires) from 3 real conversational corpora.
TRAIN the verifier on false-fires from VoxPopuli + People's (+ TTS positives + generic negs).
TEST on AMI false-fires it has NEVER seen (independent corpus) + held-out positives.
A small classifier on the existing embedding features; learns the phrase's SOUND, no transcription.
"""
import sys, glob, numpy as np
sys.path.insert(0, ".")
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM
from sklearn.linear_model import LogisticRegression

PASS1 = "../checkpoints/scratch-onnx/ozwelldone_surgical.onnx"
ev = WakeWordEvaluator(PASS1, "pretrained"); rng = np.random.default_rng(0)

def emb_windows(audio):
    mel = ev.mel.run(None, {"input": audio[None, :]})[0]
    mf = (mel.reshape(-1, 32) / 10.0 + 2.0).astype("float32")
    if mf.shape[0] < WIN: return None, None
    n = mf.shape[0]; nt = n - (n - WIN) % STRIDE; st = range(0, nt - WIN + 1, STRIDE)
    w = np.stack([mf[s:s + WIN] for s in st])[..., None].astype("float32")
    emb = ev.emb.run(None, {"input_1": w})[0].reshape(-1, EMB_DIM).astype("float32")
    if emb.shape[0] < EMB_FRAMES: return None, None
    wins = np.stack([emb[s:s + EMB_FRAMES] for s in range(0, emb.shape[0] - EMB_FRAMES + 1)])
    sc = np.array([float(ev.wake.run(None, {"input": wins[i][None]})[0].reshape(-1)[0]) for i in range(len(wins))])
    return wins, sc

def mine_dir(d, thr=0.5):
    out = []
    for f in sorted(glob.glob(f"{d}/*.wav")):
        W, S = emb_windows(load_16k_mono(f))
        if W is not None: out.append(W[S >= thr])
    return np.concatenate(out) if out else np.zeros((0, EMB_FRAMES, EMB_DIM), "float32")

print("mining false-fires from real conversation (pass-1 = surgical @0.5)...", flush=True)
ff_vox = mine_dir("/tmp/fphour/voxpopuli_test"); print(f"  VoxPopuli false-fires: {len(ff_vox)}", flush=True)
ff_peo = mine_dir("/tmp/fphour/peoples_test");   print(f"  People's  false-fires: {len(ff_peo)}", flush=True)
ff_ami = mine_dir("/tmp/fphour/thirdparty_ami"); print(f"  AMI (held-out) false-fires: {len(ff_ami)}", flush=True)
real_pos, _ = emb_windows(load_16k_mono("../../real_audio/Oz-done.wav"))
real_pos = real_pos[(np.array([float(ev.wake.run(None,{"input":real_pos[i][None]})[0].reshape(-1)[0]) for i in range(len(real_pos))])>=0.5)]

tts = np.load("../heybuddy/precalculated/ozwell_i_m_done.npy", mmap_mode="r")
tts = np.asarray(tts[rng.choice(len(tts), 5000, replace=False)])
gen = np.asarray(np.load("../heybuddy/precalculated/negs_C.npy", mmap_mode="r")[rng.choice(160000, 5000, replace=False)])[:, :EMB_FRAMES, :]
flat = lambda a: a.reshape(len(a), -1)
def split(a, f=0.5):
    i = rng.permutation(len(a)); k = max(1, int(len(a)*f)); return a[i[k:]], a[i[:k]]
tts_tr, tts_te = split(tts); rp_tr, rp_te = split(real_pos)

# TRAIN on VoxPopuli + People's false-fires (NOT AMI); TEST on AMI (independent corpus)
Xneg = np.concatenate([flat(gen), flat(ff_vox), flat(ff_peo)])
X = np.concatenate([flat(tts_tr), flat(rp_tr), Xneg])
y = np.concatenate([np.ones(len(tts_tr)+len(rp_tr)), np.zeros(len(Xneg))])
clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X, y)
acc = lambda a: clf.predict(flat(a)).mean() if len(a) else float("nan")
P = lambda a: clf.predict_proba(flat(a))[:,1].mean() if len(a) else float("nan")

print("\n=== ACOUSTIC VERIFIER — cross-corpus held-out (trained on VoxPopuli+People's, tested on AMI) ===")
print(f"  ACCEPT real wakes (your voice, held-out): {acc(rp_te)*100:3.0f}%  (n={len(rp_te)})   P={P(rp_te):.2f}  [want HIGH]")
print(f"  ACCEPT TTS positives (held-out):          {acc(tts_te)*100:3.0f}%  (n={len(tts_te)})  [want HIGH]")
print(f"  ACCEPT AMI false-fires (NEVER trained on):{acc(ff_ami)*100:3.0f}%  (n={len(ff_ami)})   P={P(ff_ami):.2f}  [want LOW]")
print(f"  -> on a corpus it never saw, it rejects {100-acc(ff_ami)*100:.0f}% of false-fires while keeping {acc(rp_te)*100:.0f}% of real wakes")
print("VERIFIER_SCALE_DONE")

# ---- model comparison (answer: does more power help, or is it data-limited?) ----
from sklearn.neural_network import MLPClassifier
print("\n=== does a more powerful model help? (same data, same AMI held-out) ===")
for name, m in [("logistic", LogisticRegression(max_iter=3000, class_weight="balanced")),
                ("MLP (256,64)", MLPClassifier(hidden_layer_sizes=(256,64), max_iter=2000, early_stopping=True))]:
    m.fit(X, y)
    a_real = m.predict(flat(rp_te)).mean()*100
    a_ff   = m.predict(flat(ff_ami)).mean()*100
    print(f"  {name:14s}: real wakes kept {a_real:3.0f}% | AMI false-fires rejected {100-a_ff:3.0f}%  (train hard-negs: {len(ff_vox)+len(ff_peo)})")
print("MODEL_COMPARE_DONE")
