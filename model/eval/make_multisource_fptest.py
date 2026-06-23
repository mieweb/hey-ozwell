#!/usr/bin/env python3
"""Build a MULTI-SOURCE held-out FP test so no single distribution biases the result.
Sources (each ~30min, 16kHz mono wavs in /tmp/fphour/<name>/):
  - peoples_test : People's Speech TEST split (open). In-distro-ish for fpx (trained on peoples train).
  - voxpopuli_test : already built (home turf for fpvox). Reused by the runner.
  - thirdparty_* : a corpus NEITHER model trained on (tries Common Voice -> TED-LIUM -> AMI).
The runner scores fpx vs fpvox PER SOURCE + COMBINED, so each model's home advantage is visible
and the COMBINED number approximates a mixed real setting."""
import os, glob, wave, numpy as np
from datasets import load_dataset

TARGET_SEC = 1800  # 30 min/source

def dump(name, make_ds):
    out = f"/tmp/fphour/{name}"; os.makedirs(out, exist_ok=True)
    if len(glob.glob(out + "/*.wav")) > 50:
        print(f"[{name}] already populated, skip", flush=True); return name
    try:
        ds = make_ds()
    except Exception as e:
        print(f"[{name}] LOAD FAIL {type(e).__name__}: {str(e)[:140]}", flush=True); return None
    tot = 0.0; n = 0
    try:
        for ex in ds:
            a = ex.get("audio")
            if not a or "array" not in a: continue
            arr = np.asarray(a["array"], dtype=np.float32); sr = a["sampling_rate"]
            if arr.size < sr * 0.5: continue
            if sr != 16000:
                idx = np.linspace(0, arr.size - 1, int(arr.size * 16000 / sr))
                arr = np.interp(idx, np.arange(arr.size), arr).astype(np.float32)
            pcm = (np.clip(arr, -1, 1) * 32767).astype("<i2")
            with wave.open(os.path.join(out, f"{n:04d}.wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
            tot += arr.size / (sr if sr == 16000 else sr); n += 1
            if n % 100 == 0: print(f"[{name}] {n} clips {tot/60:.1f}min", flush=True)
            if tot >= TARGET_SEC: break
    except Exception as e:
        print(f"[{name}] STREAM ERR after {n} clips: {type(e).__name__}: {str(e)[:140]}", flush=True)
    print(f"[{name}] DONE {n} clips {tot/60:.1f}min", flush=True)
    return name if n > 0 else None

# 1) People's Speech test split
dump("peoples_test", lambda: load_dataset("MLCommons/peoples_speech", "clean", split="test",
                                          streaming=True, trust_remote_code=True))

# 2) a true third party neither model trained on
CANDS = [
    ("common_voice", lambda: load_dataset("mozilla-foundation/common_voice_17_0", "en", split="test",
                                          streaming=True, trust_remote_code=True)),
    ("tedlium",      lambda: load_dataset("LIUM/tedlium", "release3", split="test",
                                          streaming=True, trust_remote_code=True)),
    ("ami",          lambda: load_dataset("edinburghcstr/ami", "ihm", split="test",
                                          streaming=True, trust_remote_code=True)),
]
for desc, mk in CANDS:
    print(f"[thirdparty] trying {desc}...", flush=True)
    try:
        next(iter(mk()))  # verify it loads + yields
    except Exception as e:
        print(f"[thirdparty] {desc} unavailable: {type(e).__name__}: {str(e)[:100]}", flush=True); continue
    dump(f"thirdparty_{desc}", mk); break
print("MULTISOURCE_BUILD_DONE", flush=True)
