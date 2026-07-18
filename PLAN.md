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
1. Install BlackHole (2ch).
2. Audio MIDI Setup → Multi-Output Device: physical output + BlackHole. Select it as
   system output during meetings. (Volume: set on the physical device; Multi-Output
   has no master volume.)

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
