#!/usr/bin/env python3
"""Break the shared-library voice-count ceiling with ElevenLabs Voice Design.
Generates MANY unique synthetic voices per weak accent (+ American for the diversity lift) by
DESCRIPTION (varying gender/age/tone), far beyond the library's 8 JP / 3 KO / 25 ZH voices.
Flow per voice: design(desc, filler) -> create -> convert both phrases (wav_16000) -> delete (free slot).
Output: /tmp/vd_accents/<accent>/<slug>/*.wav   Resumable-ish; aborts on quota.
These are a NEW generator -> disjoint from the shared-library n=390 test (kept as held-out).
"""
import os, glob, itertools
from elevenlabs import ElevenLabs

# weak/needs-help accents + American (voice-rich but weak -> lift). Skip already-strong IN/GB/AU/FIL.
ACCENTS = {
    "japanese": "speaking English with a clear Japanese accent",
    "korean":   "speaking English with a clear Korean accent",
    "chinese":  "speaking English with a Mandarin Chinese accent",
    "spanish":  "speaking English with a European Castilian Spanish accent",
    "latin":    "speaking English with a Latin American Spanish accent",
    "american": "speaking natural neutral American English",
}
GENDERS = ["man", "woman"]
AGES = ["young", "middle-aged", "older"]
TONES = ["calm conversational", "warm friendly", "energetic", "professional clear"]
PHRASES = {"hey_ozwell": "hey ozwell", "ozwell_done": "ozwell I'm done"}
FILLER = ("Hello, this is a short sample of my natural speaking voice so you can hear how I sound "
          "when I talk clearly and calmly in everyday conversation throughout the day.")
TTV = "eleven_multilingual_ttv_v2"; TTS = "eleven_multilingual_v2"
DESIGNS_PER_ACCENT = 12   # x3 previews ≈ 36 voices/accent
OUT = "/tmp/vd_accents"

def key(): return open(os.path.expanduser("~/.eleven_api_key")).read().strip()

def main():
    c = ElevenLabs(api_key=key())
    combos = list(itertools.product(AGES, GENDERS, TONES))
    made = 0
    for acc, accent_clause in ACCENTS.items():
        done = len(glob.glob(f"{OUT}/{acc}/ozwell_done/*.wav"))
        if done >= DESIGNS_PER_ACCENT*3:
            print(f"[{acc}] already has {done}, skip", flush=True); continue
        print(f"[{acc}] designing ~{DESIGNS_PER_ACCENT*3} voices", flush=True)
        for i in range(DESIGNS_PER_ACCENT):
            age, gender, tone = combos[i % len(combos)]
            desc = f"A {age} {gender} {accent_clause}, {tone} tone"
            try:
                prev = getattr(c.text_to_voice.design(voice_description=desc, text=FILLER, model_id=TTV, seed=i), "previews", []) or []
            except Exception as e:
                if _quota(e): return _done(made)
                print(f"  [{acc}] design FAIL {str(getattr(e,'body',e))[:80]}", flush=True); continue
            for j, p in enumerate(prev):
                gid = getattr(p, "generated_voice_id", None)
                if not gid: continue
                vid = None
                try:
                    v = c.text_to_voice.create(voice_name=f"vd_{acc}_{i}_{j}", voice_description=desc, generated_voice_id=gid)
                    vid = getattr(v, "voice_id", None)
                    for slug, text in PHRASES.items():
                        d = f"{OUT}/{acc}/{slug}"; os.makedirs(d, exist_ok=True)
                        au = b"".join(c.text_to_speech.convert(vid, text=text, model_id=TTS, output_format="wav_16000"))
                        open(f"{d}/vd_{i}_{j}.wav", "wb").write(au)
                    made += 1
                except Exception as e:
                    if _quota(e):
                        if vid:
                            try: c.voices.delete(vid)
                            except: pass
                        return _done(made)
                    print(f"  [{acc}] voice {i}_{j} FAIL {str(getattr(e,'body',e))[:70]}", flush=True)
                finally:
                    if vid:
                        try: c.voices.delete(vid)
                        except: pass
            if i % 4 == 0: print(f"  [{acc}] {i+1}/{DESIGNS_PER_ACCENT} designs, {made} voices total", flush=True)
    _done(made)

def _quota(e):
    b = str(getattr(e, "body", e))
    return "quota_exceeded" in b or getattr(e, "status_code", None) == 401

def _done(made):
    print(f"\n=== DONE === {made} voices generated", flush=True)
    for acc in ACCENTS:
        n = len(glob.glob(f"{OUT}/{acc}/ozwell_done/*.wav"))
        print(f"  {acc:10s} {n} voices", flush=True)
    print("VOICEDESIGN_DONE", flush=True)

if __name__ == "__main__":
    main()
