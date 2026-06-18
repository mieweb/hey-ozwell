# Wake-word validation log

Running record of tests for the daily updates. Single enrolled speaker (Jonathan) unless noted.
Two gates: WHO (speaker, thr 0.4) + WHAT (phrase voiceprint cosine, single fire-frame, thr 0.82).

## 2026-06-18 — Conversational false-positive rate (enrolled speaker)
- Method: ~15 min continuous natural speech (~8 min mock clinical dictation + ~7 min World Cup
  commentary), counting false wakes (fires while NOT saying a wake phrase).
- Result: **0 false fires in 15 min.** The base detector fired multiple times on conversation
  (the bars light up), but the phrase gate (WHAT) rejected 100% of them.
- Separation: conversational speech WHAT ~0.67–0.77; real wakes ~0.85–0.92; threshold 0.82 sits in
  the gap.
- Stats: best estimate ~0/hr. 95% upper bound ~12/hr (limited by the 15-min sample — "rule of three":
  0 events in 0.25 hr → 3/0.25 ≈ 12/hr). A 30–60 min session would tighten the bound (~6/hr, ~3/hr).
- Takeaway: the two-gate system vets the base detector's false fires; residual conversational leaks
  by the enrolled speaker are rare (none observed in 15 min).

## 2026-06-18 — Mask robustness (cloth proxy: shirt collar over nose/mouth)
- Recall = # detected out of 10 per phrase. Mask ON during all testing; the variable is whether the
  ENROLLMENT was done masked or clear.
- **Condition A — enrolled CLEAR, tested masked:** hey-ozwell 10/10, ozwell-i'm-done 8/10.
- **Condition B — enrolled MASKED, tested masked:** hey-ozwell 10/10, ozwell-i'm-done 10/10.
- Takeaway: a cloth mask barely hurt even when enrolled clear (only the stop phrase dipped, to 8/10);
  enrolling masked recovered both to 10/10. Confirms enrollment adapts to the user's conditions.
- Caveat: n=10/condition, so 8/10 vs 10/10 is a 2-utterance difference (within noise) — directionally
  consistent, not a precise recall number. Cloth proxy, not a real surgical mask.

## 2026-06-18 — Mixed-condition enrollment (3 masked + 3 clear per phrase, 6 reps total)
- Enrolled each phrase with 3 reps masked + 3 reps clear, then tested all four condition combos:
  - hey-ozwell, no mask: 10/10
  - ozwell-i'm-done, no mask: 10/10
  - hey-ozwell, masked: 10/10
  - ozwell-i'm-done, masked: 10/10
- Result: **40/40 across all conditions.** Mixed-condition enrollment covers both masked and clear
  with no confusion — phraseCosine matches the closest template, so each condition hits its own.
- Takeaway: enroll a few reps in each condition you'll use (masked/clear/noisy) → robust to all.
  Validates "enroll in your real conditions" as the deployment strategy.
- Caveat: n=10/condition (no misses observed, not "100% proven"); cloth proxy for the mask.

## 2026-06-18 — Background noise (recall + false positives)
- Recall with moderate background noise: hey-ozwell 9/10, ozwell-i'm-done 9/10 (vs 10/10 clear — one
  miss each, within n=10 noise). Enrollment was not noise-specific; enrolling in noise would likely
  recover the last point (same adapt mechanism as the mask).
- False positives: background noise / other voices did NOT false-fire — the WHO (speaker) gate rejects
  non-enrolled voices (other voices ~0.03–0.22 vs enrolled ~0.46–0.85).
- WHO threshold note (currently 0.4): could raise to ~0.5 for stricter impostor rejection, BUT impostor
  margin is already large (others ≤0.22 → rejected at either), and the enrolled user dips to ~0.3–0.4
  in noise/mask, so 0.5 risks false-rejecting the real doctor in those conditions. Keep 0.4 for recall;
  revisit toward 0.5 only if security is prioritized (a real mimic could score >0.22, unlike a TV).

## 2026-06-18 — Enroll-in-noise (recall recovery)
- Problem: enrolled CLEAR, tested in background noise → WHO (speaker) dropped into the 0.30s (below the
  0.4 gate) → real wakes REJECTED (recall miss in noise).
- Fix test: enrolled WITH the noise on, tested in noise → hey-ozwell 10/10, ozwell-i'm-done 9/10. Recall
  recovered (WHO back above 0.4).
- Takeaway: enrolling in the noise condition adapts the speaker template and recovers recall WITHOUT
  lowering the 0.4 threshold — same adapt mechanism as the mask. Confirms "enroll in your conditions" as
  the answer; threshold stays 0.4. (Caveat: 9/10 = 1 miss on n=10, within noise.)
