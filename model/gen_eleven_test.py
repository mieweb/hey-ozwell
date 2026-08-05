#!/usr/bin/env python3
"""Metered ElevenLabs TTS test — generate a SMALL sample first to learn pricing/usage before scaling.
Generates one phrase across a spread of shared-library voices (incl. accents Google/Azure lack:
spanish/latin/chinese/korean/japanese/filipino/indian), saves 16kHz wavs by accent, and reports
EXACT character usage (subscription delta) + per-clip cost + projection for a full positive set.

  python gen_eleven_test.py --phrase "ozwell i'm done" --limit 100 --per-accent 12
Key in ~/.eleven_api_key. Shared voices are used directly by voice_id (no library-add, no voice-slot use).
"""
import os, argparse, wave
from elevenlabs import ElevenLabs

ACCENTS = ["indian","spanish","latin","chinese","korean","japanese","filipino","british","australian","american"]
MODEL = "eleven_multilingual_v2"   # multilingual = best accent fidelity

def key():
    return open(os.path.expanduser("~/.eleven_api_key")).read().strip()

def chars_used(c):
    s = c.user.subscription.get()
    return s.character_count, s.character_limit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", default="ozwell i'm done")
    ap.add_argument("--limit", type=int, default=100, help="max clips to generate (cost guard)")
    ap.add_argument("--per-accent", type=int, default=12, help="voices to pull per accent")
    ap.add_argument("--out", default="/tmp/eval/eleven_test")
    args = ap.parse_args()

    c = ElevenLabs(api_key=key())
    before, lim = chars_used(c)
    print(f"chars before: {before:,} / {lim:,}  (remaining {lim-before:,})")
    print(f"phrase={args.phrase!r}  len={len(args.phrase)} chars  limit={args.limit} clips  model={MODEL}\n")

    # collect (accent, name, voice_id) across accents
    voices = []
    for acc in ACCENTS:
        try:
            items = c.voices.get_shared(page_size=args.per_accent, accent=acc, language="en").voices
            for v in items:
                voices.append((acc, v.name, v.voice_id))
        except Exception as e:
            print(f"  [list {acc}] failed: {type(e).__name__}: {e}")
    print(f"collected {len(voices)} candidate voices across {len(ACCENTS)} accents")

    made = {a: 0 for a in ACCENTS}; total = 0; errors = 0
    for acc, name, vid in voices:
        if total >= args.limit:
            break
        d = os.path.join(args.out, acc); os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{vid}.wav")
        try:
            audio = b"".join(c.text_to_speech.convert(vid, text=args.phrase, model_id=MODEL,
                                                      output_format="wav_16000"))
            with open(path, "wb") as f:
                f.write(audio)
            made[acc] += 1; total += 1
        except Exception as e:
            errors += 1
            print(f"  [{acc}] {name[:30]:30s} FAIL {type(e).__name__}: {str(e)[:90]}")

    after, _ = chars_used(c)
    spent = after - before
    print(f"\n=== RESULTS ===")
    print(f"clips generated: {total}  (errors: {errors})")
    print("by accent:", {a: n for a, n in made.items() if n})
    print(f"characters spent: {spent:,}  (= {spent/total:.1f} chars/clip)" if total else "no clips")
    print(f"chars remaining now: {lim-after:,}")
    if total:
        per_clip = spent / total
        # growing_business pricing reference ~ $0.18 / 1k chars (varies by plan/overage)
        for n in (1000, 5000, 20000):
            cc = int(per_clip * n)
            print(f"  projection: {n:,} clips -> {cc:,} chars  (~${cc/1000*0.18:.2f} @ $0.18/1k)")
    print(f"\nwavs in: {args.out}/<accent>/")

if __name__ == "__main__":
    main()
