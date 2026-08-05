#!/usr/bin/env python3
"""Cross-lingual DIVERSITY positives: native-language Azure voices reading the English phrases.
NOT for authentic accent (ear test: sounds American-ish) but for DIVERSITY -> generalization,
the lever that took American 64->92 on 06-06. Many distinct native acoustic models per language.
Output: /tmp/crosslingual/<locale>/<slug>/<voice>.wav  (16kHz wav). Both phrases.
"""
import os, json, urllib.request
KEY=open(os.path.expanduser("~/.azure_speech_key")).read().strip(); REGION="eastus2"
EP=f"https://{REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
PHRASES={"hey_ozwell":"hey ozwell","ozwell_done":"ozwell I'm done"}
# native-language locales (Latin Spanish spread + East Asian) -> diversity
LOCALES=["ja-JP","ko-KR","zh-CN","zh-CN-shaanxi","zh-CN-sichuan",
         "es-MX","es-ES","es-AR","es-CO","es-US","es-CL","es-PE","es-VE"]
OUT="/tmp/crosslingual"

def all_voices():
    req=urllib.request.Request(f"https://{REGION}.tts.speech.microsoft.com/cognitiveservices/voices/list",
                               headers={"Ocp-Apim-Subscription-Key":KEY})
    return json.loads(urllib.request.urlopen(req,timeout=30).read())

def synth(voice,text):
    ssml=f"<speak version='1.0' xml:lang='en-US'><voice name='{voice}'>{text}</voice></speak>"
    req=urllib.request.Request(EP,data=ssml.encode(),headers={
        "Ocp-Apim-Subscription-Key":KEY,"Content-Type":"application/ssml+xml",
        "X-Microsoft-OutputFormat":"riff-16khz-16bit-mono-pcm","User-Agent":"xl"})
    return urllib.request.urlopen(req,timeout=30).read()

voices=[v["ShortName"] for v in all_voices() if v.get("Locale") in LOCALES]
print(f"{len(voices)} native voices across {len(LOCALES)} locales", flush=True)
made=0
for v in voices:
    loc=v.split("-")[0]+"-"+v.split("-")[1]
    for slug,text in PHRASES.items():
        d=f"{OUT}/{loc}/{slug}"; os.makedirs(d,exist_ok=True)
        p=f"{d}/{v}.wav"
        if os.path.exists(p): continue
        try:
            open(p,"wb").write(synth(v,text)); made+=1
        except Exception as e:
            print(f"  {v} {slug} FAIL {str(e)[:60]}", flush=True)
print(f"\n=== DONE === {made} clips from {len(voices)} voices", flush=True)
for loc in sorted(set(v.split('-')[0]+'-'+v.split('-')[1] for v in voices)):
    import glob; print(f"  {loc}: {len(glob.glob(f'{OUT}/{loc}/ozwell_done/*.wav'))} voices", flush=True)
print("CROSSLINGUAL_DONE", flush=True)
