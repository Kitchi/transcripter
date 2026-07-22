# Meeting Recorder — Plan

Local, bot-free meeting transcription: capture system audio + mic, transcribe live-ish
with whisperX, summarize with a local MLX model. No cloud, no meeting bots, audio is
ephemeral.

## Architecture

```
                    ┌─ default input mic ──┐
  (two input        │                      ├─→ ring buffer ─→ 30s WAV chunks (2s overlap)
   streams,         └─ BlackHole 2ch ──────┘        │
   opened in code)                                  ▼
                                        transcribe worker (whisperX, large-v3)
                                          - mic channel   → [me] segments
                                          - system channel → [them] segments
                                          - interleave by timestamp → transcript.md
                                          - delete chunk WAV after transcription
                                                    │
                                       stop (Ctrl-C or silence watchdog)
                                                    ▼
                                    summarizer (mlx_lm) → summary prepended to MD
```

## Components

### 1. Capture
- Python + `sounddevice`. Two independent input streams:
  - **System default input** — follows whichever mic is active (built-in vs webcam),
    no aggregate device needed.
  - **BlackHole 2ch** — receives a copy of system output via a user-created
    Multi-Output Device (speakers/headphones + BlackHole). One-time manual setup;
    tool warns at startup if BlackHole is absent.
- Streams aligned in software by wall-clock start time (30s chunk granularity makes
  precision a non-issue).
- Rolling 30s chunks with 2s overlap (avoids splitting words at boundaries),
  written to a scratch dir.

### 2. Transcription
- **whisperX** with **large-v3** (accuracy over latency, per requirements). Batch mode
  fed rolling chunks → pseudo-live.
- Each channel transcribed separately; segments tagged `[me]` (mic) / `[them]`
  (system audio) and interleaved by timestamp into `transcript.md`.
- Chunk WAV deleted immediately after successful transcription — only ~1–2 chunks
  of audio exist on disk at any time. (Optional flag: retain WAVs until summary
  succeeds, for re-runs.)
- Overlap region deduped by dropping segments whose midpoint falls in the previous
  chunk's window.

### 3. Silence watchdog
- Per-chunk RMS tracking on both channels. Kill condition, sustained **45s**:
  - BlackHole channel ≈ zero (no system audio), AND
  - mic channel below a speech threshold **calibrated at startup** from ambient level
    (Zoom mute is software-only; the OS-level stream always carries room noise).
- Watchdog arms only after speech has been detected at least once (avoids killing
  during a slow meeting start; avoids false-positives from pre-meeting music being
  the only thing keeping it alive).
- Trigger runs the normal graceful stop path: flush partial chunk → transcribe →
  summarize → exit.

### 4. Summarization
- Post-meeting, shell out to `mlx_lm.generate` (already installed) with a local model
  (Qwen2.5-7B or Llama-3.1-8B, 4-bit).
- High-level summary prepended to `transcript.md` (or sibling `summary.md` — decide
  at implementation).

## Explicitly deferred
- **Diarization** of the `[them]` channel (whisperX supports it via pyannote;
  sherpa-onnx as the no-HF-account alternative). Channel split already gives
  me-vs-them in v1.
- Live speaker labels; any UI beyond tailing the MD file.

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

### Summarizer: long-meeting context overflow
- **Current design is single-shot**: the whole transcript is stuffed into one prompt
  (`summarizer.py`). No chunking.
- ~90 min of talk ≈ 13–18k tokens. macOS Gemma 3n E4B (~32k ctx) usually fits but
  quality sags near the top; **Linux llama-cpp is pinned to `n_ctx=8192`** and
  silently truncates from the front → long meetings lose everything but the tail.
- Fix: **map-reduce / hierarchical summarization** — token-bounded windows (aligned
  on existing chunk boundaries, light overlap), summarize each, then summarize the
  summaries. Bounds context regardless of meeting length. Own change, after the tap.

## Phases
1. **Capture + chunking**: two streams, WAV chunks, silence watchdog. Verify with a
   test call.
2. **Transcription loop**: whisperX worker, tagging, interleave, chunk cleanup.
3. **Summarizer + lifecycle polish**: MLX summary, graceful stop paths, retain-audio
   flag.
4. *(later)* Diarization pass.

## Known caveats
- Anything said while Zoom-muted still lands in `[me]` (OS-level capture).
- Music playing through BlackHole after a meeting defers the watchdog (mitigated by
  arm-after-speech, but not eliminated).
- Recording calls may require participant consent depending on jurisdiction.
