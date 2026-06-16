# Stage-2 Acoustic Verifier — live test handoff (2026-06-16)

Hand this to the browser/product chat. Goal: test the new false-positive filter live in the demo.

## What it is
The wake model ("pass-1") fires too often on conversational/random speech. The **acoustic verifier**
("pass-2") is a tiny MLP that re-scores the SAME embedding window pass-1 just used, and suppresses the
fire if it looks like junk. It judges the SOUND of the phrase, not a transcript — so it works on the
made-up word "ozwell" (the earlier Whisper/ASR text-gate failed because ASR hears "as well" and
hallucinates). It runs ONLY on a fire, reuses the embedding pass-1 already computed → no extra model
inference cost, ~1.6MB ONNX, no audio re-processing.

## Offline results (what we know going in)
- Kills **~99%** of false-fires on an INDEPENDENT corpus (AMI, never trained on). Stable.
- Keeps **~99%** of wakes across 390 disjoint synthetic voices, uniform across 10 accents. Stable.
- **The open risk = real voices.** On the one real recording we have (Jonathan, ~13 utterances), it
  dropped **~40%** of real wakes (the synthetic→real gap). That recording was one noisy session and may
  be unrepresentative — that's what this live test checks.
- Adding **5 of the user's own reps** as positives ("enrollment") recovered real-wake retention to
  **~90%** while keeping 100% false-fire kill. Enrollment is the fallback if live confirms the gap.
- Threshold 0.1 is conservative: junk scores ~0, real wakes higher; 0.1 kills ~99% of junk while keeping
  the most real wakes. The FP-kill curve is flat from 0.05–0.5 (junk all clusters near 0).

## What shipped (branch jlocala/wakeword-eval, pushed)
- `prod/js/models/ozwell-i'm-done-verifier.onnx` — frozen MLP (1536→256→64→1), parity-checked vs sklearn.
- `prod/js/src/models/acoustic-verifier.js` — the AcousticVerifier class (loads the ONNX, scores the
  embedding, `verify(audio, name, embeddingBuffer) -> bool`).
- `prod/js/src/hey-buddy.js` — passes `this.embeddingBuffer` to `verify()`; adds `verifierShadow` mode.
- `prod/js/src/index.js` — `VERIFIER_MODE` toggle + constructs the verifier.
- Only **ozwell-i'm-done** is gated (the over-firing stop phrase). "hey ozwell" passes through untouched.

## Run it
```
git pull
git lfs pull        # fetches the verifier .onnx
# then serve prod the usual way (node server) and open the demo
```

## Test (open the browser console)
Each fire logs its verdict regardless of mode:
`[acoustic-verifier] ozwell-i'm-done: P(wake)=0.xxx thr=0.1 -> CONFIRM/reject`
- **Random junk / conversation** near the mic → expect `🛑 stage-2 REJECTED` (suppressed). The win.
- **Say "ozwell i'm done"** → expect `✅ stage-2 confirmed` and it fires. Recall holding.

## The toggle (index.js, ~line 123)
```js
const VERIFIER_MODE = "active"; // "off" | "shadow" | "active"
```
- **active** (default): actually suppresses junk. Use this for the solo live test — you see wins AND
  failures in the console directly.
- **shadow**: runs + logs `👻 WOULD reject` but FIRES ANYWAY (zero recall risk). For safe data-collection
  in front of real users / a demo, NOT for solo debugging.
- **off**: stage-1 only (pre-verifier behavior).

## Start WITHOUT enrollment
Enrollment is not wired into the runtime yet; the shipped verifier is the general one. Test that first.

## Decision after the live test
- If it catches your real "ozwell i'm done" reliably AND suppresses junk → ship candidate (consider a
  short shadow-mode run for real-world stats before full activation).
- If it drops your real wakes (you say it, see `🛑 REJECTED`) → set `VERIFIER_MODE="off"`, report back.
  Next step = per-user enrollment (5 reps at signup → ~90% real-wake retention, 100% FP-kill preserved).
  Enrollment = on-device, one-time at signup, no central dataset needed.

## Notes / caveats
- Real-wake numbers are from ONE speaker's recording — small n. The live test is more representative.
- The verifier fails OPEN (any error → lets the fire through), so a load failure won't block detection.
- Embedding parity: the verifier consumes the exact tensor the wake model consumes, so if the wake model
  fires correctly in the browser, the verifier's input is in the right space. If the verifier rejects
  EVERYTHING or accepts EVERYTHING, suspect an embedding/preprocessing mismatch, not the model.
