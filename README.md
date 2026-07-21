# transcripter

Local, bot-free meeting transcription for macOS and Linux. Captures your mic and
system audio, transcribes live, and summarizes with a local LLM. **No cloud, no
meeting bots, audio is ephemeral** — chunks are deleted seconds after they're
transcribed.

The transcription and summary backends are selected automatically by platform:
mlx-whisper + mlx-lm on macOS (Metal), faster-whisper + llama-cpp-python on
Linux (CUDA or CPU).

```
 default mic ──────┐
                   ├─→ 30s WAV chunks (2s overlap) ─→ whisper worker ─→ transcript.md
 system audio  ────┘         (deleted after use)            │
 (BlackHole / .monitor)                                     ▼ on stop
                                                  local LLM summary prepended
```

## Features

- **Two-channel attribution** — your mic becomes `me`, system audio (the other
  participants) becomes `them`, interleaved by timestamp.
- **Live-ish transcript** — `transcript.md` updates every ~30s; tail it during
  the meeting.
- **Silence watchdog** — stops automatically after sustained silence, then
  transcribes the tail and summarizes.
- **Local summary** — post-meeting Summary / Key points / Action items from a
  local LLM, prepended to the transcript.
- **Ephemeral audio** — only a chunk or two of WAV exists on disk at any moment
  (opt out with `--keep-audio`).

## Install

Both platforms need [uv](https://docs.astral.sh/uv/).

### macOS

Apple Silicon. The MLX backends ship Metal-enabled wheels — nothing to compile.

```sh
uv sync
```

One-time audio setup for system capture:

1. Install [BlackHole 2ch](https://existential.audio/blackhole/).
2. In **Audio MIDI Setup**, create a **Multi-Output Device** containing your
   physical output + BlackHole, and select it as system output during meetings.
   (Set volume on the physical device — Multi-Output has no master volume.)

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
| `--no-transcribe` | | Capture only, keep WAVs |
| `--keep-audio` | | Retain chunk WAVs after transcription |
| `--silence-stop-seconds N` | `45` | Silence watchdog (0 disables) |

Everything ends up in one file: `<session-dir>/transcript.md` — summary
on top, timestamped `me`/`them` transcript below.

### Voice notes

`note` mode is a mic-only variant for dictating a note to yourself. It captures
**only** your mic (no BlackHole / system audio), transcribes it, and writes a
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
`record`: `--out`, `--sessions-dir`, `--name`, `--chunk-seconds`,
`--overlap-seconds`, `--silence-stop-seconds`, `--model`, `--summary-model`.

## Caveats

- Anything said while muted in the meeting app still lands in `me` — capture is
  OS-level, software mute doesn't reach it.
- Recording calls may require participant consent depending on your jurisdiction.

## Development

```sh
uv run pytest
uv run ruff check .
```

See [PLAN.md](PLAN.md) for architecture notes and roadmap.
