# Meeting Recorder — Plan

Local, bot-free meeting transcription: capture system audio + mic to one recording,
then echo-cancel, transcribe, diarize and summarize it in a single offline pass with
local models. No cloud, no meeting bots, audio is deleted when the pipeline finishes.

## Architecture

```
                    ┌─ default input mic ──┐
  (two input        │                      ├─→ recording.wav (16 kHz stereo,
   streams,         └─ system audio tap ───┘      mic = L, system = R)
   opened in code)                                      │
                                       stop (Ctrl-C or silence watchdog)
                                                        ▼
                                    ┌───────── offline pipeline ─────────┐
                                    │ AEC: subtract system bleed from mic│
                                    │ transcribe both channels (whisper) │
                                    │ hallucination filter               │
                                    │ diarize system → Speaker 1/2/...   │
                                    │ interleave by timestamp            │
                                    └────────────────┬───────────────────┘
                                                     ▼
                                    summarizer (mlx_lm) → summary prepended to MD
                                                     │
                                       delete recording.wav (unless --keep-audio)
```

Capture writes; everything else is an offline pass over the finished file. There
is no live transcript: the pipeline runs once, at the end. That trade buys a
continuously-adapting echo canceller, globally-clustered diarization, and whisper
seeing whole-meeting context instead of 30s windows -- and it deletes the
chunker, the transcription worker thread, and the overlap-dedup logic outright.

## Components

### 1. Capture
- Python + `sounddevice`. Two independent input streams:
  - **System default input** — follows whichever mic is active (built-in vs webcam),
    no aggregate device needed.
  - **System audio** — the bundled Core Audio process tap on macOS (see "Pending
    work"), a PulseAudio/PipeWire `.monitor` source on Linux.
- Streams drain on independent queues and deliver unequal sample counts per poll,
  so `recorder.py` buffers per channel and writes only the frames both can fill;
  `close` flushes the ragged tail. This keeps the two channels sample-aligned,
  which the echo canceller depends on.
- Captured at the device rate (48 kHz), decimated to **16 kHz** on the way to disk
  (`resample.py`): AEC, whisper and the speaker embedder all want 16 kHz, and it
  cuts a 90-minute meeting from ~2 GB to ~170 MB. Written incrementally, so a
  crash leaves a usable recording rather than nothing.

### 2. Echo cancellation (`aec.py`)
- System audio leaves the speakers and re-enters the mic, so the far end's words
  land on both channels. Subtracting the predicted echo **before** transcription
  beats deleting duplicate text after, because it also recovers **double-talk** --
  moments where both sides speak, which text dedup throws away wholesale.
- **Delay alignment comes first, and is not optional.** The mic and the system
  tap are separate CoreAudio streams on independent clocks, and `recorder.py`
  interleaves them by arrival order with no timestamps, so the mic trails the
  system channel by a per-recording amount — 127 ms and 110 ms on the two
  meetings measured, drifting ~6 ppm and stepping 37 ms mid-session on a buffer
  resync. `measure_echo_delay` finds it by band-limited GCC-PHAT; `pipeline`
  shifts the mic, cancels, and shifts back so downstream timestamps are
  unaffected. Unaligned these recordings cancel 0.0–2.5 dB; aligned, 3.5–6.4 dB.
- Returns the **minimum** delay across the session, never the mean. The filter
  is causal-only, so it cannot model an echo arriving before its reference:
  undershooting costs a little, overshooting costs nearly everything (aligning
  53 ms past the true delay dropped one recording from 6.7 dB to 0.7 dB).
- **Band-limit to 300–3400 Hz, and floor the PHAT weight** (`PHAT_FLOOR`). Both
  channels pass through the same decimator, which stamps an identical spectral
  fingerprint on each at zero lag; pure full-band PHAT locks onto that instead
  of the echo. It scored 30 on a control pairing a mic against its own system
  channel played backwards, and mis-read a true 127 ms delay as 0.1 ms. Band
  limiting plus flooring drops the controls to ~5.5 and raises the true peaks.
- **Gate on correlation sharpness, not coherence.** Peak height over background:
  impossible pairings score ~5.5–6, real echo paths 27–130, so
  `aec_sharpness_threshold = 15` sits in a wide gap. The previous coherence gate
  scored 0.0006 against a 0.10 threshold on a recording with a real, cancellable
  echo — it skipped cancellation entirely on both meetings.
- **Partitioned block frequency-domain NLMS**: `aec_reach` of filter (2048 @
  16 kHz ≈ 128 ms, matching the ~120 ms measured tail) split into `aec_block`
  partitions (512 ≈ 32 ms). Decoupling the two is what makes it robust. Welded
  together, as they were, one number had to satisfy contradictory demands: too
  short and the filter cannot cover the tail, so its predictions are wrong, the
  divergence guard rejects them and beats the filter back to zero (16119 of
  40630 blocks rejected, 731 pullbacks, 0.1 dB); too long and each block spans
  so much time that almost none is echo-only, so adaptation starves (110
  updates for 8.4M parameters, 0.0 dB). The usable window was ~1 octave wide and
  sat in a different place per recording. One continuous pass over the
  recording, so the filter converges once rather than per chunk.
- **Double-talk detector**: adapts only on blocks where mic power is below
  `aec_dtd_ratio` × system power; otherwise it holds the filter and keeps
  subtracting. **Caveat with teeth**: if the speakers are loud enough that the
  echo alone exceeds that ratio, no block ever qualifies, the filter never
  converges, and cancellation silently does nothing. The pipeline logs ERLE and
  warns at 0.0 dB; the fix is a larger `--aec-dtd-ratio`.

### 3. Transcription
- **whisper** (`large-v3-turbo`), mlx-whisper on macOS / faster-whisper on Linux.
  Backends take samples directly, so no temp WAVs.
- Each channel transcribed separately over the whole recording; segments tagged
  `[me]` (mic) / `[them]` or a diarized speaker (system) and interleaved by
  timestamp into `transcript.md`.
- **Cross-channel bleed dedup** (`transcript.py`): system audio leaves the
  speakers and re-enters the mic, so the same words land on both channels. Bleed
  is one-directional (the tap captures system output only), so a `me` segment
  overlapping a similar-text `them` segment is the echo and is dropped. Match =
  time overlap + normalized-text similarity; genuine talk-over (different words)
  is preserved. Retained as a net for residual echo the AEC misses; with
  cancellation upstream it should fire rarely.
- **Hallucination filter** (`filters.py`): Whisper emits high-prior junk on
  non-speech. Dropped using its own per-window decode signals — `no_speech_prob`
  (silence fillers) and `compression_ratio` (repetition loops) — plus an
  exact-whole-segment text blocklist for confident fillers ("Thank you.") that
  share a decode window with real speech and so escape the signal thresholds.
  All thresholds are `Config` knobs.

### 4. Diarization (`diarize.py`)
- Only the system channel needs it; the mic is one voice by construction.
- **sherpa-onnx**, not pyannote directly: it runs pyannote's segmentation-3.0
  network converted to ONNX plus a WeSpeaker CAM++ embedding model through
  onnxruntime. Same segmentation model, but **no HuggingFace gate, no access
  token, and no torch** -- which is what makes a bundled DMG tractable (~36 MB of
  weights and one wheel, vs. torch's several hundred MB).
- Both models are MIT licensed. The HF gate on `pyannote/*` is a download-time
  access control, not a redistribution restriction, so a packager accepts once
  and end users of a bundled build accept nothing.
- Fetched on first use into `~/.cache/transcripter/models` and checksum-verified.
  Note the upstream release tag really is spelled `speaker-recongition-models`.
- Whisper segments and diarizer turns have independent boundaries; each segment
  takes the speaker it **overlaps most**. A segment spanning a speaker change
  gets one label -- splitting it needs word-level timestamps (later).
- Failure is non-fatal: a diarization error costs speaker labels, not the meeting.

### 5. Silence watchdog
- Per-poll RMS tracking on both live streams. Kill condition, sustained **45s**:
  - system channel ≈ zero (no system audio), AND
  - mic channel below a speech threshold **calibrated at startup** from ambient level
    (Zoom mute is software-only; the OS-level stream always carries room noise).
- Watchdog arms only after speech has been detected at least once (avoids killing
  during a slow meeting start; avoids false-positives from pre-meeting music being
  the only thing keeping it alive).
- Trigger runs the normal graceful stop path: close the recording → offline
  pipeline → summarize → exit.

### 6. Summarization
- Post-meeting, shell out to `mlx_lm.generate` (already installed) with a local model
  (Qwen2.5-7B or Llama-3.1-8B, 4-bit).
- High-level summary prepended to `transcript.md` (or sibling `summary.md` — decide
  at implementation).

## Explicitly deferred
- **Live transcript.** Dropped deliberately when the pipeline went offline-only:
  keeping it would mean either a second transcription pass over the meeting or
  keeping the chunker alive purely for a disposable draft. Everything now lands
  at once, after stop.
- **Word-level speaker splitting** for segments that straddle a speaker change.
- Any UI beyond reading the MD file.

## Manual setup (one-time)
- **macOS 14.4+**: none. System capture uses the bundled Core Audio process-tap
  helper (read-only observer of the current output device). Original design used
  BlackHole + a Multi-Output Device; that killed the hardware volume keys, which
  is what the tap replaces. See "Pending work" above.
- **Linux**: none. A PulseAudio/PipeWire `.monitor` source is found automatically.

## Pending work

### Core Audio process tap (macOS system capture without BlackHole)
- macOS 14.4+ exposes non-invasive **process taps** (`AudioHardwareCreateProcessTap`
  + private aggregate device). Observes the real output device read-only, so the
  user keeps their normal output **and working hardware volume keys** — no BlackHole,
  no Multi-Output.
- **TCC attribution is the crux.** The helper must be its own *responsible*
  process, or macOS attributes the system-audio-recording request to the spawning
  terminal and the tap silently delivers **zeroed** audio (no error, no prompt).
  - The `responsibility_spawnattr_setdisclaim` SPI (the usual disclaim trick) is
    **gone on macOS 26** — not exported by any dylib. Dead end.
  - Working approach: launch the helper via **LaunchServices (`open -n`)**, which
    makes it its own responsible process. It also needs a `.app` bundle with an
    `NSAudioCaptureUsageDescription` (`helper/Info.plist`) and `LSUIElement`
    (a pure `LSBackgroundOnly` app can't present the prompt).
- Since `open` detaches stdio, the helper connects back to a **unix-domain socket**
  the Python side listens on: one JSON header line (`float32 / 48 kHz / stereo`),
  then interleaved PCM. `helper/SystemAudioTap.swift`, bundled at
  `src/transcripter/_bin/SystemAudioTap.app`, built/signed by `helper/build.sh`.
- Python side: `_TapStream` in `capture.py` binds a socket, `open`s the app,
  accepts the connection, downmixes stereo→mono, feeds the SYSTEM queue → chunker.
  Mic path unchanged. Linux keeps the PulseAudio/PipeWire monitor path.
- Signing: a free self-signed cert gives a stable TCC identity across rebuilds
  (ad-hoc `-` works but re-prompts each rebuild). Paid Apple account only needed
  for notarization / distribution to other Macs.

### Summarizer: long-meeting context overflow (implemented)
- **Problem**: single-shot summarization stuffed the whole transcript into one
  prompt. ~90 min ≈ 13–18k tokens. macOS Gemma (~32k ctx) usually fit but quality
  sagged near the top; **Linux llama-cpp was pinned to `n_ctx=8192`** and silently
  truncated from the front → long meetings lost everything but the tail.
- **Fix: map-reduce with a recursive reduce** (`summarizer.py`, `Summarizer`
  base class over backend primitives):
  - Backends expose `complete(text)`, `count_tokens(text)` /
    `count_tokens_batch(texts)`, and `context_tokens`. No new dependency — both
    backends already carry a tokenizer, so counts are exact. `count_tokens_batch`
    tokenizes a whole window/level in one call (MLX); llama loops.
  - **Context detection**: llama = `n_ctx`. MLX = `max_position_embeddings`, which
    multimodal gemma exports **nest under `text_config`** (top-level is absent →
    the naive read silently gave 8192 instead of 131072). We read the nested key
    and clamp to `MLX_CONTEXT_CAP = 32k` — feeding 128k tokens to a 4-bit local
    model is slow and low quality.
  - **Budget**: `usable = context − max_output − prompt_overhead`; windows/batches
    fill to **`FILL = 0.75`** of usable. Output budget is reserved *first*, so
    packing input never eats into generation room. `prompt_overhead` is the
    chat-template wrapper (BOS + role/turn markers + generation prompt) that raw
    token counts miss — **measured once at load** on MLX, a small conservative
    constant on llama (whose template we can't cheaply render to count).
  - **Fits in one call** → single-shot with the structured prompt (unchanged path).
  - **Otherwise map**: window the transcript greedily by segment line (a line is
    atomic → aligns on chunk boundaries) to the budget, with a **2-line overlap**
    carried into the next window so a point split across a seam appears whole in
    one window (`NOTES_PROMPT` merges the duplicate). Condense each window to
    terse notes (no length cap).
  - **Recursive reduce**: if the notes fit one call → final structured pass. If
    not → merge them in token-bounded batches (`NOTES_PROMPT`) and repeat. Levels
    are ~`log_fanin(#windows)`, emergent from the budget, not a fixed count.
  - **Termination** is guaranteed: every map/merge output ≤ `max_output` < budget,
    so each level strictly shrinks the note count toward one. A progress guard
    forces pairwise merges if a pathologically small context ever fails to shrink.
- **Knobs**: `--n-ctx` (record + note) sets the llama-cpp context window for
  bigger-GPU users; no-op on macOS (MLX context is auto-detected then capped).
  Note mode stays single-shot (dictated notes are short).

## Phases
1. ~~**Capture + chunking**: two streams, WAV chunks, silence watchdog.~~
2. ~~**Transcription loop**: streaming worker, tagging, interleave, chunk cleanup.~~
3. **Summarizer + lifecycle polish**: MLX summary, graceful stop paths, retain-audio
   flag.
4. **Offline pipeline** (supersedes 1–2): record to a single WAV, then AEC +
   diarization + transcription as one post-meeting pass. The chunker and the
   transcription worker were deleted here.

## To-do
- **macOS GUI (feasibility)** — assess a native Mac front end over the CLI core
  (menu-bar app / SwiftUI). Minimum scope: start/stop, live transcript view,
  session list, and driving the TCC audio-recording prompt. Open question:
  process model — embed the Python pipeline vs. shell out to `transcripter`.
- **CI/CD (GitHub Actions)** — lint + test (`ruff`, `pytest`) on push/PR to start.
  Later: build/sign the Core Audio tap helper and package releases. Note the macOS
  runner constraints for the tap (no real audio device, TCC prompts can't be
  granted headless) — CI likely covers the pure-Python layers only.

## Known caveats
- Anything said while Zoom-muted still lands in `[me]` (OS-level capture).
- Music playing after a meeting defers the watchdog (mitigated by arm-after-speech,
  but not eliminated).
- Loud speakers can starve the echo canceller's adaptation (see AEC above); the run
  log reports ERLE so this is visible rather than silent.
- **Near-end speech damage is argued, not measured.** On real audio there is no
  ground truth separating "removed echo" from "removed my voice". What is known:
  the output is `near - W*far`, so the filter can only ever subtract a filtered
  copy of the system channel; far-silent frames come back bit-identical; a
  mic paired with an unrelated meeting's system channel comes back untouched
  (ERLE 0.00 dB, −0.01 dB RMS); and the synthetic double-talk test holds
  corr(cleaned, my voice) > 0.95. But on recording A the loud-both frames
  keep only 20% of their power, which is consistent with a strong echo path and
  *also* with over-subtraction. The honest acceptance test is transcription
  quality — `scripts/aec_offline.py --transcribe` does that A/B and has not been
  run on these recordings yet.
- Echo cancellation is modest, not decisive: 3.5–6.4 dB on the two meetings
  measured. `transcript.py`'s bleed dedup stays on as the net. The remaining
  ceiling is the double-talk detector — the filter learns from only 4–18% of
  blocks — so a residual-echo suppressor or a smarter DTD is the next lever, not
  more taps.
- The full recording sits on disk for the duration of the meeting. It is deleted
  after processing unless `--keep-audio`, but a crash now leaves audio behind
  (which is also what makes a re-run possible).
- Recording calls may require participant consent depending on jurisdiction.
