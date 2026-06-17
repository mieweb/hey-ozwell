# Build + export the SHIPPABLE pass-2 acoustic verifier for ozwell-i'm-done.
#
# KEY FINDING (06-16): TTS-only positives do NOT generalize to real human wakes (verifier_final.py
# kept only 17% of real wakes). Real wakes MUST be in the positive set (the probe's 85% came from
# training on half the real wakes). So:
#   - Honest eval (PROBE): AMI held out ENTIRELY (independent corpus) + half the real wakes held out.
#   - Shipped model: train on EVERYTHING (TTS + ALL real wakes [+noise-aug] + People's+Vox+AMI negs).
# Recall lever tested here: noise-augment the real wakes to expand real-voice positive mass.
#
# Usage: python eval/verifier_export.py   (run from model/)
import sys, glob, json, numpy as np, soundfile as sf, os
sys.path.insert(0, ".")
from evaluate_wakeword import WakeWordEvaluator, load_16k_mono, WIN, STRIDE, EMB_FRAMES, EMB_DIM
from sklearn.neural_network import MLPClassifier

PASS1 = "../checkpoints/scratch-onnx/ozwelldone_surgical.onnx"   # the 92%-recall config used for mining
OUT_ONNX = "../prod/js/models/ozwell-i'm-done-verifier.onnx"
AMI_CACHE = "heybuddy/precalculated/mined_ff_ami.npy"
REAL_WAKE = "../real_audio/Oz-done.wav"
ev = WakeWordEvaluator(PASS1, "pretrained"); rng = np.random.default_rng(0)
NOISE = sorted(glob.glob("/tmp/fphour/peoples_1h/*.wav"))

def winsc(a):
    mel = ev.mel.run(None, {"input": a[None, :]})[0]; mf = (mel.reshape(-1, 32)/10+2).astype("float32")
    if mf.shape[0] < WIN: return None, None
    n = mf.shape[0]; nt = n-(n-WIN) % STRIDE; st = range(0, nt-WIN+1, STRIDE)
    w = np.stack([mf[s:s+WIN] for s in st])[..., None].astype("float32")
    emb = ev.emb.run(None, {"input_1": w})[0].reshape(-1, EMB_DIM).astype("float32")
    if emb.shape[0] < EMB_FRAMES: return None, None
    ws = np.stack([emb[s:s+EMB_FRAMES] for s in range(0, emb.shape[0]-EMB_FRAMES+1)])
    sc = np.array([float(ev.wake.run(None, {"input": ws[i][None]})[0].reshape(-1)[0]) for i in range(len(ws))])
    return ws, sc

def addn(y):
    if not NOISE: return y
    nz, _ = sf.read(NOISE[rng.integers(len(NOISE))], dtype="float32"); nz = nz if nz.ndim == 1 else nz.mean(1)
    if len(nz) < len(y): nz = np.tile(nz, len(y)//len(nz)+1)
    nz = nz[:len(y)]; ps, pn = (y**2).mean()+1e-9, (nz**2).mean()+1e-9
    return (y+nz*np.sqrt(ps/pn/(10**(rng.uniform(5, 18)/10)))).astype("float32")

def mine_dir(d, aug=2):  # firing windows (score>=0.5) over a dir, with `aug` noisy copies each
    out = []
    for f in sorted(glob.glob(f"{d}/*.wav")):
        y = load_16k_mono(f)
        for clip in [y]+[addn(y) for _ in range(aug)]:
            W, S = winsc(clip)
            if W is not None: out.append(W[S >= 0.5])
    return np.concatenate(out) if out else np.zeros((0, EMB_FRAMES, EMB_DIM), "float32")

def mine_wakes(f, aug):  # real-wake firing windows from a single recording, clean + `aug` noisy copies
    out = []
    y = load_16k_mono(f)
    for clip in [y]+[addn(y) for _ in range(aug)]:
        W, S = winsc(clip)
        if W is not None: out.append(W[S >= 0.5])
    return np.concatenate(out) if out else np.zeros((0, EMB_FRAMES, EMB_DIM), "float32")

flat = lambda a: a.reshape(len(a), -1)

# --- negatives (hard) ---
ppl = np.load("heybuddy/precalculated/mined_false_fires.npy")
vox = np.load("heybuddy/precalculated/mined_ff_done_vox.npy")
if os.path.exists(AMI_CACHE):
    ami = np.load(AMI_CACHE); print(f"AMI cache: {ami.shape}")
else:
    print("mining AMI (cached after first run)..."); ami = mine_dir("/tmp/fphour/thirdparty_ami", aug=2)
    np.save(AMI_CACHE, ami); print(f"AMI mined+cached: {ami.shape}")
rec_ff = mine_dir("../real_audio/false_fires", aug=0)   # independent recording false-fires (clean)

# --- positives ---
tts = np.asarray(np.load("heybuddy/precalculated/ozwell_i_m_done.npy", mmap_mode="r")[rng.choice(127000, 8000, replace=False)])
gen = np.asarray(np.load("heybuddy/precalculated/negs_C.npy", mmap_mode="r")[rng.choice(160000, 8000, replace=False)])[:, :EMB_FRAMES, :]
wakes_clean = mine_wakes(REAL_WAKE, aug=0)              # real wakes, clean (the honest test material)
print(f"negs: People's {len(ppl)} + Vox {len(vox)} + AMI {len(ami)} | TTS pos {len(tts)} | gen negs {len(gen)} | real wakes(clean) {len(wakes_clean)}")

def train(pos, neg):
    X = np.concatenate([flat(pos), flat(neg)]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=2000, early_stopping=True).fit(X, y)

def sweep(m, real_test, ami_test, rec_test, tag):
    P = lambda a: m.predict_proba(flat(a))[:, 1] if len(a) else np.array([])
    Pr, Pa, Pf = P(real_test), P(ami_test), P(rec_test)
    print(f"\n=== {tag} (reject if P<t) ===\n   t   real-kept  AMI-rej(indep)  recFF-rej")
    for t in [0.2, 0.3, 0.4, 0.5]:
        print(f"  {t:.1f}    {(Pr>=t).mean()*100:4.0f}%      {(Pa<t).mean()*100:4.0f}%        {(Pf<t).mean()*100:4.0f}%")

# --- PROBE: AMI fully held out; half the real wakes held out. Compare NO-aug vs noise-aug positives. ---
i = rng.permutation(len(wakes_clean)); k = len(wakes_clean)//2
te_idx, tr_idx = i[:k], i[k:]
wake_te = wakes_clean[te_idx]                                   # held-out CLEAN real wakes (honest recall test)
negs_probe = np.concatenate([ppl, vox])                        # AMI NOT in training -> independent
for aug in [0, 4]:
    wakes_aug = mine_wakes(REAL_WAKE, aug=aug)                  # re-mine with aug; align to train indices by re-split is impossible -> use clean-tr + extra aug windows
    # train positives = TTS + (clean training-half real wakes) + (all aug windows of the recording)
    pos = np.concatenate([tts, wakes_clean[tr_idx]] + ([wakes_aug] if aug else []))
    m = train(pos, negs_probe)
    sweep(m, wake_te, ami, rec_ff, f"PROBE aug={aug}  (real-wake train {len(pos)-len(tts)}, AMI held out)")

# --- FINAL shippable model: ALL data (TTS + ALL real wakes + noise-aug + People's+Vox+AMI negs) ---
wakes_all = mine_wakes(REAL_WAKE, aug=6)
pos_final = np.concatenate([tts, wakes_all])
neg_final = np.concatenate([ppl, vox, ami, gen])
mfin = train(pos_final, neg_final)
print(f"\nFINAL train: pos {len(pos_final)} (TTS {len(tts)} + real+aug {len(wakes_all)}) | neg {len(neg_final)}")

# --- export to ONNX (input name 'input' [None,1536]; zipmap off -> output_probability [N,2]) ---
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
onx = convert_sklearn(mfin, initial_types=[("input", FloatTensorType([None, EMB_FRAMES*EMB_DIM]))],
                      options={id(mfin): {"zipmap": False}})
with open(OUT_ONNX, "wb") as f: f.write(onx.SerializeToString())
print(f"\nEXPORTED -> {OUT_ONNX} ({os.path.getsize(OUT_ONNX)} bytes)")
# sanity: run the exported ONNX on a held-out real wake and a false-fire
import onnxruntime as rt
sess = rt.InferenceSession(OUT_ONNX, providers=["CPUExecutionProvider"])
def onnx_p(a): return sess.run(None, {"input": flat(a).astype("float32")})[1][:, 1] if len(a) else np.array([])
print(f"ONNX check: real-wake mean P={onnx_p(wake_te).mean():.3f} | AMI-FF mean P={onnx_p(ami[:200]).mean():.3f} | recFF mean P={onnx_p(rec_ff).mean():.3f}")
print("EXPORT_DONE")
