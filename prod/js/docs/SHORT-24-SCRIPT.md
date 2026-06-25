# Daily short — Hands-free voice in MIE's AIChat (06-24)

**Length:** ~75–90 s. **Style:** screen recording + voiceover (no on-camera face). Show source, show
the live demo, less telling. This is a *team* short, so it's fine to show code.

**The one-line story:** I composed three on-device pieces — wake-word, browser Whisper dictation, and
MIE's real AIChat — into a hands-free voice chat, all running in MIE's Storybook, audio never leaving the page.

---

## ✅ Pre-flight (do this BEFORE recording — avoids dead air on camera)

1. On the Mac, in the `ui` repo: `git pull` (branch `feat/ozwell-voice`), then `pnpm storybook`.
2. Open **Product → Feature Modules → AI → Hands-Free Chat → "HandsFree"**. Allow the mic.
3. **Warm the model:** do ONE full run first (say "hey ozwell" → "test" → "ozwell I'm done"). The first
   dictation downloads the Whisper model (slow once); after that it's instant. Don't film the cold load.
4. Have these files open in tabs, ready to show (Cmd+click in the sidebar's "View source on GitHub", or
   just the editor):
   - `src/components/AI/HandsFreeChat.stories.tsx`  (the composition — the `onWake` handler)
   - `src/components/WakeWord/useWakeWord.ts`  (the `getStream()` line — shared mic)
   - `src/components/AI/whisperTranscribe.ts`  (on-device transcription)
5. Visuals to drop in (in `prod/js/docs/`): `short-24-title.png`, `short-24-architecture.png`.

---

## 🎬 Script

### 1 — Cold open: the payoff (0:00–0:14)
**SCREEN:** the live **Hands-Free Chat** demo, full screen.
**DO:** say "hey ozwell" (banner flips red → *Dictating*), say *"Schedule a follow-up in two weeks,"*
then "ozwell I'm done" (→ *Transcribing* → the message posts).
**SAY:**
> "I just talked to MIE's chat without touching it. 'Hey ozwell' to start, talk, 'ozwell I'm done' to send.
> And the transcription ran entirely in my browser — no audio ever left the page."

### 2 — What it is (0:14–0:26)
**SCREEN:** `short-24-title.png`, then the Storybook sidebar showing the three stories (AIChat Voice,
Wake Word, Hands-Free Chat).
**SAY:**
> "This is MIE's *actual* AIChat component. Today I wired three on-device pieces into their library —
> a wake-word listener, browser Whisper dictation, and the chat — each its own reusable component, then composed."

### 3 — How it works, show the source (0:26–0:52)
**SCREEN:** `HandsFreeChat.stories.tsx` — highlight the `onWake` handler.
**SAY:**
> "The composition is tiny. A wake event just starts or stops dictation —"
**SCREEN:** highlight `startDictation` / `stopDictation`, then `useWakeWord.ts` `getStream()` line, then
`short-24-architecture.png`.
**SAY:**
> "— and here's the one hard part. The listener has to hold the mic open the whole time, and dictation
> needs it too. In a browser, two mic streams cancel each other out — they go silent. So everything shares
> ONE stream: the recorder is just a second consumer of the listener's mic." *(let the diagram sit a beat)*

### 4 — Either input (0:52–1:04)
**SCREEN:** the demo — this time **tap the mic button** in the composer to dictate, then tap to stop.
**SAY:**
> "And it's not only voice. The chat's own mic button does the exact same thing, driven by that same shared
> stream. Say the word, or tap the button — either works."

### 5 — Reproducible + next (1:04–1:18)
**SCREEN:** GitHub — the `jlocala1/ui` fork, `feat/ozwell-voice` branch; then back to the running Storybook.
**SAY:**
> "It's all on a branch in MIE's Storybook — anyone on the team can pull it and run it. Next up: making it
> doctor-only, with on-device speaker verification."

---

## Notes / reminders
- **Emphasize on-device / "audio never leaves the page"** — that's the clinical/HIPAA win, say it at least once.
- **Show, don't tell:** linger on the demo and the actual code, not slides.
- Keep cuts tight; the cold-open payoff is the hook — lead with it.
- If you want a shorter cut for Doug: scenes 1, 4, 5 only (~40 s) — the *what* and the *either-works*, less code.
