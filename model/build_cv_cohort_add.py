#!/usr/bin/env python3
"""Add Common Voice spontaneous-English speakers to the AS-norm cohort (broader, more diverse crowd).
One 1.2s segment per DISTINCT client_id, embedded via the same TitaNet the runtime uses, merged with the
existing cohort. Output -> combined cohort for the product."""
import sherpa_onnx, soundfile as sf, numpy as np, json, csv, os, random

MODEL = "/home/jlocala/sherpa-onnx/wasm/speaker-diarization/assets/nemo_en_titanet_small.onnx"
BASE = "/home/jlocala/hey-ozwell/cv-drop/sps-corpus-4.0-2026-06-12-en"
TSV = f"{BASE}/ss-corpus-en.tsv"; AUD = f"{BASE}/audios"
EXIST = "/tmp/sv-cohort.json"
OUT = "/tmp/sv-cohort-combined.json"
N_ADD = 250

ext = sherpa_onnx.SpeakerEmbeddingExtractor(
    sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=MODEL, num_threads=2, debug=False, provider="cpu"))

def embed(x, sr):
    s = ext.create_stream(); s.accept_waveform(sr, x.astype("float32")); s.input_finished()
    e = np.array(ext.compute(s), "float64"); n = np.linalg.norm(e); return e / n if n > 0 else e

byspk = {}
for r in csv.DictReader(open(TSV), delimiter="\t"):
    byspk.setdefault(r["client_id"], []).append(r["audio_file"])
spk = list(byspk); random.Random(0).shuffle(spk)
print(f"{len(spk)} distinct speakers; sampling {N_ADD}")

add = []
for s in spk:
    if len(add) >= N_ADD:
        break
    f = os.path.join(AUD, byspk[s][0])
    try:
        a, sr = sf.read(f)
    except Exception:
        continue
    a = (a.mean(1) if a.ndim > 1 else a).astype("float32")
    seg, win = None, int(1.2 * sr)
    for st in range(0, max(0, len(a) - win), int(0.6 * sr)):
        s2 = a[st:st + win]
        if np.sqrt(np.mean(s2 ** 2)) > 0.02:
            seg = s2; break
    if seg is None:
        continue
    add.append([round(float(x), 5) for x in embed(seg, sr)])

exist = json.load(open(EXIST))
combined = exist + add
json.dump(combined, open(OUT, "w"))
print(f"existing {len(exist)} + CV {len(add)} = {len(combined)} -> {OUT}")
arr = np.array(combined)
import itertools
sims = [float(np.dot(arr[i], arr[j])) for i, j in itertools.islice(itertools.combinations(range(len(arr)), 2), 3000)]
print(f"combined pairwise cosine mean {np.mean(sims):.2f} (distinct voices)")
