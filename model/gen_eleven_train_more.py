#!/usr/bin/env python3
"""Pull MORE distinct training voices for ozwell-done (disjoint from the n=390 test set).
More real voices > over-augmenting a few. Excludes any voice_id already used in test (or train).
Generates 'ozwell i'm done' for up to TARGET_TRAIN_VOICES new voices/accent.
Output appends to /tmp/eleven_big/train/<accent>/ozwell_done/.  Resumable; aborts on quota.
"""
import os, glob, re
from elevenlabs import ElevenLabs

ACCENTS = ["japanese","chinese","spanish","latin","indian","filipino","korean","british","australian","american"]
PHRASE = ("ozwell_done", "ozwell i'm done")
MODEL = "eleven_multilingual_v2"
MAXV = 150                 # how deep to page into the library
TARGET_TRAIN_VOICES = 100  # cap distinct train voices/accent (library-limited for JP/KO)
OUT = "/tmp/eleven_big/train"

def key(): return open(os.path.expanduser("~/.eleven_api_key")).read().strip()
def vids_in(d):  # voice_ids already present (test or existing train) -> exclude
    return {re.sub(r"_s\d+\.wav$","",os.path.basename(p)) for p in glob.glob(d+"/*.wav")}

def main():
    c = ElevenLabs(api_key=key())
    slug, text = PHRASE
    made = 0
    for acc in ACCENTS:
        try:
            lib, page = [], 0
            while len(lib) < MAXV:
                r = c.voices.get_shared(page_size=100, accent=acc, language="en", page=page)
                lib += [v.voice_id for v in r.voices]
                if not getattr(r, "has_more", False): break
                page += 1
        except Exception as e:
            print(f"[{acc}] list FAIL {e}", flush=True); continue
        test_ids  = vids_in(f"/tmp/eleven_big/test/{acc}/{slug}")
        train_ids = vids_in(f"{OUT}/{acc}/{slug}")
        already = test_ids | train_ids
        new = [v for v in lib if v not in already][:TARGET_TRAIN_VOICES]
        d = f"{OUT}/{acc}/{slug}"; os.makedirs(d, exist_ok=True)
        print(f"[{acc}] lib={len(lib)} test={len(test_ids)} existing-train={len(train_ids)} -> +{len(new)} new", flush=True)
        for vid in new:
            p = f"{d}/{vid}_s1.wav"
            if os.path.exists(p): continue
            try:
                audio = b"".join(c.text_to_speech.convert(vid, text=text, model_id=MODEL, output_format="wav_16000", seed=1))
                open(p,"wb").write(audio); made += 1
            except Exception as e:
                body = str(getattr(e,"body",e))
                if "quota_exceeded" in body or getattr(e,"status_code",None)==401:
                    print(f"\n⛔ QUOTA EXHAUSTED after {made} new clips. Re-run resumes.", flush=True); _inv(); return
                print(f"  [{acc}] {vid[:8]} FAIL {str(e)[:60]}", flush=True)
    print(f"\n=== DONE === {made} new train clips", flush=True); _inv(); print("GEN_TRAINMORE_DONE", flush=True)

def _inv():
    for acc in ACCENTS:
        n = len(glob.glob(f"{OUT}/{acc}/ozwell_done/*.wav"))
        print(f"  {acc:11s} train voices: {n}", flush=True)

if __name__ == "__main__":
    main()
