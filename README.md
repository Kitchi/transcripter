# meeting-recorder

Local, bot-free meeting transcription for macOS. Captures your mic and system
audio, transcribes live with [mlx-whisper](https://github.com/ml-explore/mlx-examples),
and summarizes with a local MLX LLM. **No cloud, no meeting bots, audio is
ephemeral** — chunks are deleted seconds after they're transcribed.

```
 default mic ──┐
               ├─→ 30s WAV chunks (2s overlap) ─→ whisper worker ─→ transcript.md
 BlackHole ────┘         (deleted after use)            │
 (system audio)                                         ▼ on stop
                                            mlx-lm summary prepended
```

## Features

- **Two-channel attribution** — your mic becomes `me`, system audio (the other
  participants) becomes `them`, interleaved by timestamp.
- **Live-ish transcript** — `transcript.md` updates every ~30s; tail it during
  the meeting.
- **Silence watchdog** — stops automatically after sustained silence, then
  transcribes the tail and summarizes.
- **Local summary** — post-meeting Summary / Key points / Action items via
  `mlx-lm`, prepended to the transcript.
- **Ephemeral audio** — only a chunk or two of WAV exists on disk at any moment
  (opt out with `--keep-audio`).

## Requirements

- Apple Silicon Mac (mlx-whisper and mlx-lm are Metal-accelerated)
- [BlackHole 2ch](https://existential.audio/blackhole/) for system-audio capture
- [uv](https://docs.astral.sh/uv/)

### One-time audio setup

1. Install BlackHole (2ch).
2. In **Audio MIDI Setup**, create a **Multi-Output Device** containing your
   physical output + BlackHole, and select it as system output during meetings.
   (Set volume on the physical device — Multi-Output has no master volume.)

## Usage

```sh
uv run meeting-recorder record            # records into sessions/<timestamp>/
```

Stop with `Ctrl-C`, or let the silence watchdog stop it for you. Useful flags:

| Flag | Default | |
|---|---|---|
| `--out DIR` | `sessions/<timestamp>` | Session directory |
| `--model REPO` | `mlx-community/whisper-large-v3-turbo` | Whisper model (HF repo) |
| `--summary-model DIR` | local gemma-4-e4b | Local MLX model dir for the summary |
| `--no-summary` | | Skip the post-meeting summary |
| `--no-transcribe` | | Capture only, keep WAVs |
| `--keep-audio` | | Retain chunk WAVs after transcription |
| `--silence-stop-seconds N` | `45` | Silence watchdog (0 disables) |

Everything ends up in one file: `sessions/<timestamp>/transcript.md` — summary
on top, timestamped `me`/`them` transcript below.

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
