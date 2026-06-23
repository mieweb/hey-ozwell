#!/usr/bin/env python3
"""Dump VoxPopuli `test` split as 16kHz mono wavs for an HONEST held-out FP test.
test split is DISJOINT from the en/train shards fpvox trained on -> no leakage. ~1.5h target.
Output: /tmp/fphour/voxpopuli_test/*.wav  (same format as peoples_1h)."""
import os, wave, numpy as np
from datasets import load_dataset

OUT = "/tmp/fphour/voxpopuli_test"
TARGET_SEC = 1.5 * 3600
os.makedirs(OUT, exist_ok=True)

ds = load_dataset("facebook/voxpopuli", "en", split="test", streaming=True, trust_remote_code=True)
total = 0.0; n = 0
for ex in ds:
    a = ex["audio"]; arr = np.asarray(a["array"], dtype=np.float32); sr = a["sampling_rate"]
    if sr != 16000 or arr.size < sr * 0.5:   # skip <0.5s; VoxPopuli is natively 16k
        continue
    pcm = np.clip(arr, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(os.path.join(OUT, f"{n:04d}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
    total += arr.size / sr; n += 1
    if n % 100 == 0:
        print(f"  {n} clips, {total/60:.1f} min", flush=True)
    if total >= TARGET_SEC:
        break
print(f"DONE: {n} clips, {total/3600:.2f}h -> {OUT}")
