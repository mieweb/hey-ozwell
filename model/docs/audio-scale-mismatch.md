# Finding: train/inference audio-scale mismatch (×32767)

**Date:** 2026-06-03 · **Status:** confirmed empirically, fix not yet applied · **Severity:** likely
root cause of poor wake-word recall (incl. `ozwell i'm done`).

## TL;DR
heybuddy **trains** on audio scaled to int16 range (`audio *= 32767`) before the mel-spectrogram
model. The **production JS runtime** and the **eval harness** both feed audio in `[-1, 1]` float
range — ~32767× quieter. So any model trained with current heybuddy sees a completely different
input distribution at inference than at training, and collapses to near-constant output (~0.1,
recall ≈ 0). This is a strong candidate for *why the shipped wake-word models underperform*.

## Evidence (the three pipelines)
| Pipeline | Audio scale into mel model | Source |
|---|---|---|
| heybuddy training | **× 32767** (int16 range) | [`heybuddy/embeddings.py:182`](../heybuddy/embeddings.py#L182) `audio_tensor *= 32767.0` |
| Production runtime | `[-1, 1]` (Web Audio `getChannelData`) | [`prod/js/src/hey-buddy.js:316`](../../prod/js/src/hey-buddy.js#L316) `process()` → `spectrogram.run(audio)`, no scaling |
| Eval harness | `[-1, 1]` (soundfile float) | [`model/eval/evaluate_wakeword.py:48`](../eval/evaluate_wakeword.py#L48) `load_16k_mono`, no scaling |

Mel **post**-scaling (`x/10 + 2`) and the window params (WIN=76, STRIDE=8, EMB_FRAMES=16) match
across all three — the *only* mismatch is the input audio amplitude.

## Empirical confirmation
Scored through the eval harness with the audio scale toggled:

| Model | Test set | scale ×1 (as-is harness/prod) | scale ×32767 (matches current training) |
|---|---|---|---|
| retrained `ozwell i'm done` (5k, stage 1) | ElevenLabs | recall@0.5 = **10%** | recall@0.5 = **34%**, @0.3 = 66% |
| retrained `ozwell i'm done` (5k, stage 1) | in-domain Piper | recall@0.5 = **6.7%** | recall@0.5 = **66.7%**, @0.3 = 93% |
| shipped `ozwell i'm done` baseline | ElevenLabs | recall@0.5 = 20% | recall@0.5 = 16% |
| **shipped `hey ozwell` (works, ~98%)** | hey-ozwell | **recall@0.5 = 97%**, pos mean 0.957 | recall@0.5 = 55%, pos mean 0.514 |

Reproduce: copy `eval/evaluate_wakeword.py`, add `audio = audio * 32767.0` before the return in
`load_16k_mono`, re-run.

### The decisive datapoint: `hey ozwell`
The **working** `hey ozwell` model scores **97% at ×1** and *drops* to 55% at ×32767 — i.e. it was
trained at `[-1, 1]`, and production's `[-1, 1]` is **correct** for it. The freshly retrained
`ozwell i'm done` is the opposite (wants ×32767), because **current heybuddy training applies the
`*= 32767` and `hey ozwell` was trained before/without it**. (The shipped `ozwell i'm done` baseline
is near-chance at both scales — broken regardless.)

**Conclusion:** production `[-1, 1]` is the source of truth. The `*= 32767` in heybuddy training is
the regression. Do **not** scale production up — that would break `hey ozwell` (97% → 55%).

## A second, independent issue (recipe over-suppression)
Separately, heybuddy's default **3-stage** training over-suppressed recall on the small 5k set
(in-training Piper test recall: Stage 1 ≈ 70–80% → Stage 2 ≈ 38% → Stage 3 ≈ 9%, as it drove FP
rate → 0). The saved `_final.pt` is the collapsed Stage-3 state. Use fewer stages / a higher
`--target-false-positive-rate`, or rely on more positive diversity (full 100k) to resist collapse.
heybuddy keeps only one overwritten `*_final.pt` — no per-stage checkpoints.

## UPDATE 2026-06-03 (later): it's loudness consistency, not a fixed ×32767
Deeper investigation refined the above. The embedding scale produced by the mel→embedding pipeline
depends on **audio loudness**, and the real requirement is that **positives, negatives, and inference
audio all sit at a consistent loudness**:
- Piper positives peak ~1.0 (loud) → embeddings max ~73.
- ElevenLabs eval clips peak ~0.7 (loud) → small embeddings.
- freesound negatives peak ~**0.1** (very quiet) → embeddings max ~**16000** (the "int16"/300× thing
  earlier was actually just quiet audio, not a literal int16 scale).

Consequences:
- Training at `[-1,1]` (disabling `*= 32767`) **does** match the loud `[-1,1]` production/eval audio and
  recovers recall (5k fast model: **76% @0.5 at production scale**, vs ~7% for the int16-trained model).
  That part of the fix holds.
- BUT the negative set must be at the **same loudness** as the positives. The downloaded default
  negatives and a naive freesound `extract` are quiet → wrong embedding scale → if trained against the
  loud `[-1,1]` positives, the model separates by *scale*, not content (→ the 58% FPR we saw, or a
  scale-as-label model). **Open task:** peak/loudness-normalize the negative audio before extracting,
  so negative embeddings match the positives. Then retrain `[-1,1]` positives + matched negatives.

Status: `embeddings.py` `*= 32767` is currently **disabled** (provisional — gives the best production
recall so far). The remaining blocker to a *usable* model is a loudness-matched negative set.

## Provenance — where the mismatch comes from (verified against upstream)
heybuddy is really **two separate codebases** that share the mel/embedding ONNX models but were
written independently:
- the **Python trainer** (`model/heybuddy/`, imported from upstream `painebenjamin/hey-buddy`
  `src/python/heybuddy/` in commit `7b820a7`), and
- the **JS browser runtime** (`prod/js/`, forked from the *same* upstream's `src/js/` demo via
  `amandamarg/hey-ozwell-demo`).

I checked upstream's own source: **the mismatch is in upstream itself.** Upstream's
`src/python/heybuddy/embeddings.py` has `audio_tensor *= 32767.0`; upstream's
`src/js/src/models/mel-spectrogram.js` feeds the mic's `[-1,1]` floats to the mel model with no
scaling. So the `×32767` is *not* a local MIE change — it's pristine upstream, and the train/inference
amplitude contract is simply not enforced between upstream's two halves. Anyone combining upstream's
trainer with upstream's runtime inherits it. We only noticed because `hey ozwell` happens to have been
trained on the `[-1,1]` side (matching the runtime) and works, while `ozwell i'm done` was trained with
the `×32767` trainer and silently doesn't.

## The fix
**Train heybuddy at `[-1, 1]` — remove the `audio_tensor *= 32767.0` at
[`heybuddy/embeddings.py:182`](../heybuddy/embeddings.py#L182)** (and audit `__main__.py:563`
`div_(32768.0)` and any other place audio amplitude is set), so training matches the production
runtime and the working `hey ozwell` model. Then retrain `ozwell i'm done` and verify with the
**unmodified** harness (which correctly mirrors production). Do **not** scale production/harness up —
`hey ozwell` proves `[-1, 1]` is correct and ×32767 would break it (97% → 55%).

Caveat: removing the scale changes the embedding distribution the model trains on, so it requires a
retrain to benefit (it does not fix already-shipped checkpoints). The `*= 32767` also feeds the
embedding cache in `heybuddy/precalculated/` — delete that cache after the change so features
regenerate at the correct scale.
