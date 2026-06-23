#!/usr/bin/env python3
"""Generate ElevenLabs ACCENT POSITIVES for both phrases, with DISJOINT train/test voice pools.
Targets the cross-vendor weak spots (JP/ZH/ES/Latin) + breadth (Indian/Filipino/Korean/British/AU/US).
- Voices per accent are split: ~75% TRAIN, ~25% TEST (>=1 test if possible) by voice_id -> no leakage.
- Output 16kHz mono wavs (our training format):
    /tmp/eleven_pos/{train,test}/<accent>/<phrase_slug>/<voiceid>[_seedN].wav
- TRAIN voices get 2 seeds (light variation); TEST voices 1. Real multiplication happens at embed-time aug.
- Meters character usage (subscription delta). Phrases are 10-15 chars -> cost is trivial.
Run:  python gen_eleven_accents.py
"""
import os, glob, wave, numpy as np
from elevenlabs import ElevenLabs

ACCENTS = ["japanese","chinese","spanish","latin","indian","filipino","korean","british","australian","american"]
PHRASES = {"hey_ozwell": "hey ozwell", "ozwell_done": "ozwell i'm done"}
MODEL = "eleven_multilingual_v2"
TRAIN_SEEDS = [1, 2]
TEST_SEEDS  = [1]
OUT = "/tmp/eleven_pos"

def key(): return open(os.path.expanduser("~/.eleven_api_key")).read().strip()
def chars(c): s=c.user.subscription.get(); return s.character_count, s.character_limit

def save(c, vid, text, path, seed):
    audio = b"".join(c.text_to_speech.convert(vid, text=text, model_id=MODEL,
                                              output_format="wav_16000", seed=seed))
    with open(path, "wb") as f: f.write(audio)

def main():
    c = ElevenLabs(api_key=key())
    before, lim = chars(c)
    print(f"chars before: {before:,}/{lim:,} (remaining {lim-before:,})", flush=True)
    made = {"train":0, "test":0}; errs = 0
    for acc in ACCENTS:
        try:
            voices = c.voices.get_shared(page_size=40, accent=acc, language="en").voices
        except Exception as e:
            print(f"[{acc}] list FAIL {type(e).__name__}: {e}", flush=True); continue
        vids = [v.voice_id for v in voices]
        if not vids: print(f"[{acc}] no voices", flush=True); continue
        n_test = max(1, len(vids)//4)
        test_v, train_v = vids[:n_test], vids[n_test:]   # disjoint split
        print(f"[{acc}] {len(vids)} voices -> {len(train_v)} train / {len(test_v)} test", flush=True)
        for split, vlist, seeds in [("train", train_v, TRAIN_SEEDS), ("test", test_v, TEST_SEEDS)]:
            for slug, text in PHRASES.items():
                d = f"{OUT}/{split}/{acc}/{slug}"; os.makedirs(d, exist_ok=True)
                for vid in vlist:
                    for sd in seeds:
                        suffix = "" if len(seeds)==1 else f"_s{sd}"
                        p = f"{d}/{vid}{suffix}.wav"
                        if os.path.exists(p): continue
                        try:
                            save(c, vid, text, p, sd); made[split]+=1
                        except Exception as e:
                            body = str(getattr(e, "body", e))
                            if "quota_exceeded" in body or getattr(e, "status_code", None) == 401:
                                print(f"\n⛔ API-KEY QUOTA EXHAUSTED (status 401 quota_exceeded). "
                                      f"Doug must raise the key's credit limit. Re-run resumes (skips existing). "
                                      f"Made so far: train {made['train']}, test {made['test']}.", flush=True)
                                for sp in ("train","test"):
                                    for ac in ACCENTS:
                                        for sl in PHRASES:
                                            w = glob.glob(f"{OUT}/{sp}/{ac}/{sl}/*.wav")
                                            if w: print(f"  {sp:5s} {ac:11s} {sl:12s} {len(w)}", flush=True)
                                print("GEN_ELEVEN_ACCENTS_DONE", flush=True); return
                            nonlocal_err(e, acc, vid)
    after, _ = chars(c)
    print(f"\n=== DONE === train clips {made['train']}, test clips {made['test']}", flush=True)
    print(f"chars spent: {after-before:,}  (remaining {lim-after:,})", flush=True)
    for split in ("train","test"):
        for acc in ACCENTS:
            for slug in PHRASES:
                w = glob.glob(f"{OUT}/{split}/{acc}/{slug}/*.wav")
                if w: print(f"  {split:5s} {acc:11s} {slug:12s} {len(w)}", flush=True)
    print("GEN_ELEVEN_ACCENTS_DONE", flush=True)

def nonlocal_err(e, acc, vid):
    print(f"  [{acc}] {vid[:10]} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)

if __name__ == "__main__":
    main()
