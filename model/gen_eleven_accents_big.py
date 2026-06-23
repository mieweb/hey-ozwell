#!/usr/bin/env python3
"""BIG ElevenLabs accent set — many distinct voices, generous disjoint train/test split.
Fixes the "test on n=10" problem: pull up to MAXV voices/accent, reserve ~40% as a DISTINCT
test pool (the honest way to grow a test set is more voices, not augmenting the same ones).
TEST voices: 2 takes each (different seeds) for a little within-voice variation sampling.
TRAIN voices: 1 take each (heavy augmentation happens later at embed time, on GPU).
Output: /tmp/eleven_big/{train,test}/<accent>/<phrase_slug>/<voiceid>_s<seed>.wav
Resumable (skips existing); aborts cleanly on API-key quota. Japanese/Korean are library-capped.
"""
import os, glob, numpy as np
from elevenlabs import ElevenLabs

ACCENTS = ["japanese","chinese","spanish","latin","indian","filipino","korean","british","australian","american"]
PHRASES = {"hey_ozwell": "hey ozwell", "ozwell_done": "ozwell i'm done"}
MODEL = "eleven_multilingual_v2"
MAXV = 80            # voices to pull per accent
TEST_FRAC = 0.40     # reserve this fraction of voices as a DISJOINT test pool
TEST_TAKES = [1, 2]  # seeds per test voice
TRAIN_TAKES = [1]    # seeds per train voice (augmented heavily at embed time)
OUT = "/tmp/eleven_big"

def key(): return open(os.path.expanduser("~/.eleven_api_key")).read().strip()
def chars(c): s=c.user.subscription.get(); return s.character_count, s.character_limit

def main():
    c = ElevenLabs(api_key=key())
    before, lim = chars(c); print(f"chars before {before:,}/{lim:,}", flush=True)
    made = {"train":0,"test":0}
    for acc in ACCENTS:
        try:
            vids = [v.voice_id for v in c.voices.get_shared(page_size=MAXV, accent=acc, language="en").voices]
        except Exception as e:
            print(f"[{acc}] list FAIL {e}", flush=True); continue
        if not vids: continue
        n_test = max(1, round(len(vids)*TEST_FRAC))
        test_v, train_v = vids[:n_test], vids[n_test:]
        print(f"[{acc}] {len(vids)} voices -> {len(train_v)} train / {len(test_v)} test", flush=True)
        for split, vlist, seeds in [("train",train_v,TRAIN_TAKES), ("test",test_v,TEST_TAKES)]:
            for slug, text in PHRASES.items():
                d = f"{OUT}/{split}/{acc}/{slug}"; os.makedirs(d, exist_ok=True)
                for vid in vlist:
                    for sd in seeds:
                        p = f"{d}/{vid}_s{sd}.wav"
                        if os.path.exists(p): continue
                        try:
                            audio = b"".join(c.text_to_speech.convert(vid, text=text, model_id=MODEL,
                                                                      output_format="wav_16000", seed=sd))
                            open(p,"wb").write(audio); made[split]+=1
                        except Exception as e:
                            body = str(getattr(e,"body",e))
                            if "quota_exceeded" in body or getattr(e,"status_code",None)==401:
                                print(f"\n⛔ QUOTA EXHAUSTED. made train {made['train']} test {made['test']}. Re-run resumes.", flush=True)
                                _inv(); return
                            print(f"  [{acc}] {vid[:8]} FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
    after,_ = chars(c)
    print(f"\n=== DONE === train {made['train']} test {made['test']} | chars spent {after-before:,}", flush=True)
    _inv(); print("GEN_BIG_DONE", flush=True)

def _inv():
    for sp in ("train","test"):
        for acc in ACCENTS:
            n = len(glob.glob(f"{OUT}/{sp}/{acc}/*/*.wav"))
            if n: print(f"  {sp:5s} {acc:11s} {n}", flush=True)

if __name__ == "__main__":
    main()
