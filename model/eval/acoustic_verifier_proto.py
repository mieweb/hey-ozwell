#!/usr/bin/env python3
"""Acoustic verifier prototype (Apple-style pass-2) — learns the SOUND of the phrase, not its text.
Replaces the failed Whisper gate. Operates on the SAME speech-embedding features as pass-1.

Key question: among clips PASS-1 already fired on, can a small classifier separate REAL wakes from
the CONVERSATIONAL false-fires? We mine firing-window embeddings:
  - from Oz-done.wav (all real "ozwell i'm done")        -> real-wake POSITIVES
  - from the World Cup + clinical conversation           -> false-fire HARD NEGATIVES
plus bulk TTS positives + generic negatives. Train logistic regression; eval on HELD-OUT
real-wakes vs HELD-OUT false-fires (the honest separation test).
"""
import sys, glob, numpy as np
sys.path.insert(0, ".")
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM
from sklearn.linear_model import LogisticRegression

PASS1 = "../checkpoints/scratch-onnx/ozwelldone_surgical.onnx"  # the 92% config that raised FP
ev = WakeWordEvaluator(PASS1, "pretrained")
rng = np.random.default_rng(0)

def emb_windows(audio):
    mel = ev.mel.run(None, {"input": audio[None, :]})[0]
    mf = (mel.reshape(-1, 32) / 10.0 + 2.0).astype("float32")
    if mf.shape[0] < WIN: return np.zeros((0, EMB_FRAMES, EMB_DIM), "float32"), np.array([])
    n = mf.shape[0]; nt = n - (n - WIN) % STRIDE; st = range(0, nt - WIN + 1, STRIDE)
    w = np.stack([mf[s:s + WIN] for s in st])[..., None].astype("float32")
    emb = ev.emb.run(None, {"input_1": w})[0].reshape(-1, EMB_DIM).astype("float32")
    if emb.shape[0] < EMB_FRAMES: return np.zeros((0, EMB_FRAMES, EMB_DIM), "float32"), np.array([])
    wins = np.stack([emb[s:s + EMB_FRAMES] for s in range(0, emb.shape[0] - EMB_FRAMES + 1)])
    scores = np.array([float(ev.wake.run(None, {"input": wins[i][None]})[0].reshape(-1)[0]) for i in range(len(wins))])
    return wins, scores

def mine_fired(wav, thr=0.5):
    """Embeddings of windows where PASS-1 fired (>=thr)."""
    a = load_16k_mono(wav); W, S = emb_windows(a)
    return W[S >= thr] if len(S) else W

# --- mine real-wake positives (Oz-done.wav) and false-fire hard negatives (conversation) ---
real_pos = mine_fired("../../real_audio/Oz-done.wav")
hard_neg = np.concatenate([mine_fired(f) for f in glob.glob("../../real_audio/*.wav")
                           if "Oz-done" not in f and "Hey-oz" not in f] or [np.zeros((0,EMB_FRAMES,EMB_DIM),"float32")])
print(f"mined: real-wake positives={len(real_pos)} | conversational false-fire hard-negs={len(hard_neg)}")
if len(real_pos) < 4 or len(hard_neg) < 4:
    print("not enough mined data for a split"); sys.exit()

# --- bulk data from caches (TTS positives, generic negatives) ---
tts_pos = np.load("../heybuddy/precalculated/ozwell_i_m_done.npy", mmap_mode="r")
tts_pos = np.asarray(tts_pos[rng.choice(len(tts_pos), 4000, replace=False)])
gen_neg = np.load("../heybuddy/precalculated/negs_C.npy", mmap_mode="r")
gen_neg = np.asarray(gen_neg[rng.choice(len(gen_neg), 4000, replace=False)])[:, :EMB_FRAMES, :]

def split(a, frac=0.5):
    idx = rng.permutation(len(a)); k = max(1, int(len(a) * frac)); return a[idx[k:]], a[idx[:k]]
rp_tr, rp_te = split(real_pos); hn_tr, hn_te = split(hard_neg)
flat = lambda a: a.reshape(len(a), -1)

# train: TTS+real positives vs generic+hard negatives
X = np.concatenate([flat(tts_pos), flat(rp_tr), flat(gen_neg), flat(hn_tr)])
y = np.concatenate([np.ones(len(tts_pos)+len(rp_tr)), np.zeros(len(gen_neg)+len(hn_tr))])
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, y)

def acc(a): return clf.predict(flat(a)).mean()  # fraction classified positive
print("\n=== ACOUSTIC VERIFIER — held-out separation (the real test) ===")
print(f"  held-out REAL wakes      accepted: {acc(rp_te)*100:3.0f}%  (n={len(rp_te)})  [want HIGH]")
print(f"  held-out FALSE-FIRES     accepted: {acc(hn_te)*100:3.0f}%  (n={len(hn_te)})  [want LOW]")
print(f"  (sanity) TTS positives   accepted: {acc(tts_pos)*100:3.0f}%   generic negs accepted: {acc(gen_neg)*100:3.0f}%")
# probability margin
pr = clf.predict_proba(flat(rp_te))[:,1].mean() if len(rp_te) else 0
pn = clf.predict_proba(flat(hn_te))[:,1].mean() if len(hn_te) else 0
print(f"  mean P(wake): real held-out {pr:.2f}  vs  false-fire held-out {pn:.2f}")
print("VERIFIER_PROTO_DONE")
