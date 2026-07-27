# transcripter

Local, bot-free meeting transcription for macOS and Linux. Captures your mic and
system audio to one recording, then echo-cancels, transcribes, diarizes and
summarizes it in a single pass when the meeting ends. **No cloud, no meeting
bots** — the recording is deleted once the transcript is written.

The transcription and summary backends are selected automatically by platform:
mlx-whisper + mlx-lm on macOS (Metal), faster-whisper + llama-cpp-python on
Linux (CUDA or CPU).

```
 default mic ──────┐
                   ├─→ recording.wav ─→ on stop: echo cancel ─→ whisper ─→ transcript.md
 system audio  ────┘   (16 kHz stereo)            + diarize        │
 (CoreAudio tap /                                                  ▼
  .monitor)                                       local LLM summary prepended
                                                  recording.wav deleted
```

## Features

- **Two-channel attribution** — your mic becomes `me`, system audio (the other
  participants) becomes `them`, interleaved by timestamp.
- **Speaker diarization** — the far end is split into `Speaker 1`, `Speaker 2`…
  via pyannote's segmentation model run through sherpa-onnx (no HuggingFace
  account, no torch). Tune with `--diarize-threshold`, or pin the count with
  `--speakers N`. Disable with `--no-diarize`.
- **Echo cancellation** — an adaptive filter subtracts the speakers' output from
  your mic before transcription, so the far end doesn't appear twice and
  talk-over survives. Disable with `--no-aec`.
- **Transcript cleanup** — one-directional cross-channel dedup drops the mic's
  pickup of the speakers echoing `them`, and a hallucination filter removes
  Whisper's non-speech junk (silence fillers like "Thank you.", repetition
  loops) using its own decode signals plus a small text blocklist.
- **Silence watchdog** — stops automatically after sustained silence, then runs
  the pipeline and summarizes.
- **Local summary** — post-meeting Summary / Key points / Action items from a
  local LLM, prepended to the transcript.
- **Audio deleted on completion** — the session recording is removed once the
  transcript is written (keep it with `--keep-audio`). It is written
  incrementally at 16 kHz, so a 90-minute meeting costs ~170 MB while recording
  and a crash still leaves a re-runnable file.

## Install

Both platforms need [uv](https://docs.astral.sh/uv/).

`uv sync` (below) sets up a project-local env you drive with `uv run transcripter`.
To instead install `transcripter` as a global command — runnable from anywhere,
no `uv run` prefix:

```sh
uv tool install .          # from a checkout; or `uv tool install transcripter`
transcripter record
```

The platform backends resolve automatically from the environment markers, and the
bundled macOS tap helper ships with the package. On Linux, GPU still needs the
build-time steps below (`uv tool install` won't set `CMAKE_ARGS` for you), so a
CPU-only `uv tool install` is the clean path there.

### macOS

Apple Silicon, **macOS 14.4+**. The MLX backends ship Metal-enabled wheels —
nothing to compile.

```sh
uv sync
```

**No audio setup needed.** System capture uses a non-invasive Core Audio
*process tap* (a bundled, code-signed helper app) that observes your current
output device read-only. Your normal output and hardware volume keys keep
working — no BlackHole, no Multi-Output Device.

On first run, macOS shows a **"…would like to record this computer's audio"**
prompt — approve it (or enable *Transcripter System Audio Tap* under **System
Settings → Privacy & Security → Screen & System Audio Recording**). Until it's
granted, the system channel records silence.

> Requires macOS 14.4+ (the process-tap API). The bundled helper is built from
> [`helper/SystemAudioTap.swift`](helper/SystemAudioTap.swift); rebuild/sign it
> with [`helper/build.sh`](helper/build.sh) (see that script's header for the
> free self-signed-certificate setup).

### Linux

System capture uses a PulseAudio/PipeWire `.monitor` source automatically — no
setup needed. CPU works out of the box:

```sh
uv sync
```

**GPU** needs two independent things (transcription and summary use different
engines):

- **Transcription** (faster-whisper): add the `gpu` extra to pull the CUDA libs.
  ```sh
  uv sync --extra gpu
  ```
- **Summary** (llama-cpp-python): must be *compiled* with CUDA. The flag is a
  build-time env var and **must be exported before install** — a `[gpu]` extra
  cannot set it:
  ```sh
  CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra gpu --reinstall-package llama-cpp-python
  ```

Both fall back to CPU automatically when no GPU is present: faster-whisper via
`device="auto"`, and a CUDA-built llama-cpp simply runs on CPU. So GPU builds
are safe to ship to CPU-only machines.

## Usage

```sh
uv run transcripter record                # records into ./<timestamp>/
```

Stop with `Ctrl-C`, or let the silence watchdog stop it for you. Useful flags:

| Flag | Default | |
|---|---|---|
| `--out DIR` | | Full session path (overrides the two below) |
| `--sessions-dir DIR` | current dir | Base directory for session folders |
| `--name NAME` | | Prefixed to the timestamp leaf, e.g. `standup-<timestamp>` |
| `--model NAME` | platform default | Whisper model (mlx HF repo / faster-whisper name) |
| `--summary-model PATH` | platform default | Summary model (MLX dir / GGUF file) |
| `--no-summary` | | Skip the post-meeting summary |
| `--no-transcribe` | | Capture only, keep the recording |
| `--keep-audio` | | Retain `recording.wav` after processing |
| `--silence-stop-seconds N` | `45` | Silence watchdog (0 disables) |
| `--no-aec` | | Skip echo cancellation |
| `--aec-dtd-ratio N` | `0.05` | Raise if the log reports 0.0 dB ERLE (see Caveats) |
| `--no-diarize` | | Skip diarization; the far end stays one `them` |
| `--diarize-threshold N` | `0.5` | Lower splits into more speakers |
| `--speakers N` | auto | Pin the far-end speaker count |

Everything ends up in one file: `<session-dir>/transcript.md` — summary
on top, timestamped `me` / `Speaker N` transcript below.

The first diarized run downloads two ONNX models (~36 MB) to
`~/.cache/transcripter/models`.

### Voice notes

`note` mode is a mic-only variant for dictating a note to yourself. It captures
**only** your mic (no system audio), transcribes it, and writes a
single summarized note — the raw transcript is used to feed the summarizer and
then discarded.

```sh
uv run transcripter note --name grocery-list   # writes ./<date>-grocery-list.md
```

**Spoken directives.** Whatever you say first is read as an instruction for how
to shape the note, stripped from the body, and applied to the rest — for example:

- *"Make this a to-do list. Pick up milk, call the dentist, renew the car
  registration…"* → a checklist.
- *"This is a technical summary for a GitHub ticket. The auth middleware drops
  the session cookie when…"* → an issue-style write-up.

Say nothing directive-like and it falls back to a titled summary with bullets.
Stops on 45s of mic silence or `Ctrl-C`. Flags are the capture/model subset of
`record`: `--out`, `--sessions-dir`, `--name`, `--silence-stop-seconds`,
`--model`, `--summary-model`. Echo cancellation and diarization do not apply:
there is no far-end reference to cancel against and only one voice to label.

## Caveats

- Anything said while muted in the meeting app still lands in `me` — capture is
  OS-level, software mute doesn't reach it.
- **Loud speakers can disable echo cancellation.** The filter only adapts while
  your mic is quiet relative to the system audio; if the echo itself is louder
  than `--aec-dtd-ratio`, it never converges. The run log prints the ERLE it
  achieved — if that reads `0.0 dB` while the far end was clearly audible, raise
  `--aec-dtd-ratio` (try `0.2`) or turn the speakers down. Headphones sidestep
  the problem entirely.
- **Diarization quality is unverified.** The plumbing works, but the speaker
  clustering has not been validated against a real multi-speaker meeting; if
  everyone comes out as one speaker, lower `--diarize-threshold` or set
  `--speakers N`.
- Recording calls may require participant consent depending on your jurisdiction.

## Development

```sh
uv run pytest
uv run ruff check .
```

The macOS system-audio tap helper is a prebuilt, code-signed `.app` **checked
into the repo** at `src/transcripter/_bin/SystemAudioTap.app` — so users without
Xcode command-line tools can run without compiling. That committed binary is
what actually runs; [`helper/build.sh`](helper/build.sh) regenerates it from the
Swift source and is the source of truth for how it's built. The two can drift —
after editing `helper/SystemAudioTap.swift`, rebuild and commit the result:

```sh
helper/build.sh          # universal binary, code-signed -> src/transcripter/_bin/
```

See [PLAN.md](PLAN.md) for architecture notes and roadmap.
