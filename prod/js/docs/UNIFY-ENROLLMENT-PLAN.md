# Unify enrollment: one capture → WHO (speaker) + WHAT (phrase) gates

Written 2026-06-17 after auditing both branches. Goal: ONE voice enrollment that gates both
"is it the doctor" (WHO) and "did they say the phrase" (WHAT), with a working precision check
(reject false fires) instead of the dead Whisper transcript gate.

## KEY FINDING — don't duplicate; the product already has most of this
The wakeword-eval (demo) branch grew its own `src/models/enrollment.js` (speech-embedding cosine
similarity, per-phrase templates) + a trained `*-verifier.onnx`. **The product branch already has
the SAME speech-embedding similarity mechanism** as the "content voiceprint":
- `hey-buddy.js`: `voiceprints={}`, `setVoiceprint(name,vectors)`, `voiceprintSimilarity(name,liveVec)`
  (subtracts a running background-mean so cosine reflects content), `voiceprintThreshold` (0.72 in
  index), `voiceprintGate` (0.3). Computed every frame as `returnMap[name].voiceprintSim`.
- `index.js` (~475–636): loads/saves voiceprints to localStorage, an enroll flow that captures reps
  per phrase → `setVoiceprint`, re-applies on load. **Used for RECALL** (in checkWakeWords, if the
  model fires nothing, fall back to a voiceprint match to amplify a weak/accented wake).

So the product has **two enrollments already**:
1. **WHO** — TitaNet speaker voiceprint (`speaker-verify.js` → `SpeakerVerify`, per-phrase centroids,
   `enrollDoctor()` ~line 348, gates action in `runWakeGate` ~line 243).
2. **WHAT (recall)** — content voiceprint (speech-embedding cosine), enrolled separately (~475–636),
   used to BOOST recall, NOT to reject.
3. **WHAT (precision)** — the **dead Whisper stage-2** in `runWakeGate` (~256–268). Proven to fail
   ("ozwell" → ASR hallucinates). This is what should be replaced.

=> Do NOT port `enrollment.js` / a 2nd voiceprint system in. Reuse the product's content voiceprint.

## The 3 surgical changes (each independently testable in the browser)

### 1. One enrollment → both voiceprints
Today `enrollDoctor()` (speaker) and the content-voiceprint enroll (~475–636) are separate capture
flows → user enrolls twice. Combine: in the single enroll loop, each captured rep already gives
(a) the utterance AUDIO (→ `SpeakerVerify.enroll`) and (b) the live `embedding` (exposed in the
`onProcessed` data as `result.embedding`, the flattened [16,96]). Collect both per rep:
- audio clips → `SpeakerVerify.enroll(phrase, clips)`  (WHO, already done)
- embeddings → `heyBuddy.setVoiceprint(phrase, vectors)` + save to the voiceprint localStorage (WHAT)
Result: one "Enroll" button → both gates populated. TEST: enroll once, confirm both
`SpeakerVerify.hasEnrollment(name)` and `heyBuddy.hasVoiceprint(name)` are true.

### 2. Replace dead Whisper stage-2 with a PRECISION gate (reject false fires)
In `runWakeGate`, the WHAT check is Whisper transcript-match (dead). Replace with the content
voiceprint we already compute: a fire is a false-fire if the live voiceprint similarity is LOW.
Two options:
- (a) Reuse `voiceprintSim`: stash the fire-time `returnMap[name].voiceprintSim` (or recompute on
  the wake window) and REJECT if `< rejectThreshold` (calibrate; recall-boost uses ~0.72, the reject
  bar may differ — tune on the live readout, like we did on the demo: real wakes ~0.92, junk/near-miss
  lower).
- (b) OR bring the demo's trained `ozwell-i'm-done-verifier.onnx` / `hey-ozwell-verifier.onnx` as a
  general floor (works pre-enrollment); enrolled users get the voiceprint precision on top.
Recommended: (a) for enrolled users (per-user, strong — demo showed 94–99% conversation rejection at
~90% recall), keep (b) as the un-enrolled floor. TEST: enrolled, real wake → CONFIRM; play
conversation/near-miss → REJECT. Then remove the Whisper block + its `window.Whisper` dependency in
the gate (keep Whisper for the actual dictation transcription, which is separate).

### 3. Detection gate = WHO ∧ WHAT
`runWakeGate` already does WHO (speaker-verify). After change #2 it also does WHAT (voiceprint
precision). Fire only if both pass. Keep the existing recall-fallback (voiceprint amplifies weak
model fires) — that's the RECALL use and composes with the PRECISION reject (different threshold).

## Don't-lose checklist
- Keep `SpeakerVerify` (WHO) intact.
- Keep the content-voiceprint RECALL fallback in `checkWakeWords` (amplify weak wakes).
- Only ADD the voiceprint PRECISION reject + REMOVE the Whisper gate. Whisper stays for dictation.
- The demo's `enrollment.js` + trained verifiers are reference; the trained verifiers are useful as
  the un-enrolled floor (optional). The demo branch (jlocala/wakeword-eval) keeps its standalone
  version working.

## Why this is documented, not yet coded
The product is a working app wired to the live Ozwell chat, with evolved/overlapping voiceprint
machinery, and there's no node/browser in the dev box to test. Each change above touches core
detection/enrollment paths, so they should be made one at a time WITH a browser test, not dropped in
blind. This doc is the safe checkpoint so nothing is lost and the next session executes cleanly.

## Reference (where things live)
- Product (this worktree): `src/index.js` (runWakeGate ~243, Whisper stage-2 ~256, enrollDoctor ~348,
  content-voiceprint enroll ~475–636), `src/hey-buddy.js` (voiceprint methods ~117–185, checkWakeWords
  ~343), `src/speaker-verify.js` (SpeakerVerify), `sv-runtime/` (TitaNet WASM), `models/`.
- Demo (jlocala/wakeword-eval): `src/models/enrollment.js`, `src/models/acoustic-verifier.js`,
  `models/{hey-ozwell,ozwell-i'm-done}-verifier.onnx`, `model/eval/browser_embed.py` (offline embedder).
