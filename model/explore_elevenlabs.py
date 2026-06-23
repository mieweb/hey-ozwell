#!/usr/bin/env python3
"""Explore what the ElevenLabs voice library offers, esp. by ACCENT (incl. Latino/East Asian
that Google/Azure lacked). Run once ELEVEN_API_KEY is set:  ELEVEN_API_KEY=... python explore_elevenlabs.py
Prints accents available + voice counts, so we can pick voices for a cross-vendor test / diverse positives."""
import os, collections
from elevenlabs import ElevenLabs

key = os.getenv("ELEVEN_API_KEY")
if not key:
    p = os.path.expanduser("~/.eleven_api_key")
    if os.path.exists(p):
        key = open(p).read().strip()
assert key, "put the key in ~/.eleven_api_key (chmod 600) or set ELEVEN_API_KEY"
client = ElevenLabs(api_key=key)

def labels_of(v):
    lab = getattr(v, "labels", None) or {}
    return lab if isinstance(lab, dict) else {}

# 1) account voices
print("=== account voices ===")
try:
    voices = client.voices.get_all().voices
    by_accent = collections.Counter()
    for v in voices:
        acc = labels_of(v).get("accent", "?")
        by_accent[acc] += 1
        print(f"  {v.name:24s} accent={acc:12s} lang={labels_of(v).get('language','?')}")
    print("  by accent:", dict(by_accent))
except Exception as e:
    print("  (account voices error:", e, ")")

# 2) shared community library (the big one — accent-filterable). API name varies by SDK version.
print("\n=== shared library by accent (sampling) ===")
for accent in ["indian", "british", "australian", "american", "spanish", "latin", "chinese", "korean", "japanese", "filipino"]:
    try:
        # try the common shared-voices call; adjust to the installed SDK if this errors
        resp = client.voices.get_shared(page_size=20, accent=accent, language="en")
        items = getattr(resp, "voices", resp)
        print(f"  accent='{accent}': {len(items)} voices  e.g. {[getattr(x,'name','?') for x in items[:3]]}")
    except Exception as e:
        print(f"  accent='{accent}': (query failed — {type(e).__name__}: {e})")
print("\nNOTE: if get_shared errors, run `python -c \"from elevenlabs import ElevenLabs; help(ElevenLabs(api_key='x').voices)\"` to see the right method, and we'll adjust.")
