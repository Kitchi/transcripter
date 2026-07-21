import argparse
import logging
import signal
from datetime import datetime
from pathlib import Path

from . import capture, devices
from .config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Local, bot-free meeting recorder.")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Record mic + system audio into WAV chunks.")
    rec.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Full session directory path; overrides --sessions-dir/--name.",
    )
    rec.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory for session folders (default: current directory).",
    )
    rec.add_argument(
        "--name",
        default=None,
        help="Recording name, prefixed to the timestamp (e.g. standup-20260720-143000).",
    )
    rec.add_argument("--chunk-seconds", type=float, default=30.0)
    rec.add_argument("--overlap-seconds", type=float, default=2.0)
    rec.add_argument(
        "--silence-stop-seconds",
        type=float,
        default=45.0,
        help="Stop after this much sustained silence (0 disables).",
    )
    rec.add_argument(
        "--model",
        default=None,
        help="Whisper model (default: platform backend's default -- an mlx-whisper "
        "HF repo on macOS, a faster-whisper model name on Linux).",
    )
    rec.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Capture only; keep chunk WAVs, skip transcription.",
    )
    rec.add_argument(
        "--summary-model",
        type=Path,
        default=None,
        help="Summary model (default: platform backend's default -- an MLX model dir "
        "on macOS, a GGUF file on Linux).",
    )
    rec.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the post-meeting summary.",
    )
    rec.add_argument(
        "--keep-audio",
        action="store_true",
        help="Retain chunk WAVs after transcription (default: delete).",
    )

    note = sub.add_parser(
        "note",
        help="Mic-only voice note: transcribe, then write a directive-steered summary.",
    )
    note.add_argument("--out", type=Path, default=None, help="Full session directory path.")
    note.add_argument("--sessions-dir", type=Path, default=Path.cwd())
    note.add_argument("--name", default=None, help="Note name, prefixed to the date.")
    note.add_argument("--chunk-seconds", type=float, default=30.0)
    note.add_argument("--overlap-seconds", type=float, default=2.0)
    note.add_argument(
        "--silence-stop-seconds",
        type=float,
        default=45.0,
        help="Stop after this much sustained mic silence (0 disables).",
    )
    note.add_argument("--model", default=None, help="Whisper model (platform default if unset).")
    note.add_argument(
        "--summary-model", type=Path, default=None, help="Summary model (platform default if unset)."
    )

    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    if args.command == "note":
        _run_note(args)
        return

    if args.out:
        out = args.out
    else:
        date = datetime.now().strftime("%Y-%m-%d")
        leaf = f"{date}-{args.name}" if args.name else f"{date}-meeting"
        out = args.sessions_dir / leaf
    cfg = Config(
        out_dir=out,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        silence_stop_seconds=args.silence_stop_seconds or float("inf"),
    )

    mic = devices.default_mic()
    system = devices.find_system_capture()
    logging.info("mic: %s", devices.describe(mic))
    logging.info("system: %s", devices.describe(system))
    logging.info("session dir: %s", out)

    # Route SIGTERM through the same graceful-stop path as Ctrl-C; also restore
    # SIGINT in case we were launched as a background job (where it's ignored).
    def _graceful_stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _graceful_stop)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    session = capture.CaptureSession(cfg, devices={capture.MIC: mic, capture.SYSTEM: system})

    worker = None
    if not args.no_transcribe:
        from .transcriber import make_backend
        from .worker import TranscriptionWorker

        worker = TranscriptionWorker(
            backend=make_backend(args.model),
            out_path=out / "transcript.md",
            overlap_seconds=cfg.overlap_seconds,
            keep_audio=args.keep_audio,
            rms_floor=cfg.transcribe_rms_floor,
        )
        worker.start()
        session.on_chunk = worker.enqueue

    reason = session.run()
    logging.info("session ended (%s): %d chunks", reason, len(session.chunk_files))
    if worker is not None:
        logging.info("waiting for transcription to finish...")
        worker.finish()
        transcript_path = out / "transcript.md"
        if not args.no_summary and worker.builder.segments:
            from .summarizer import make_summarizer

            logging.info("summarizing...")
            summarizer = make_summarizer(args.summary_model)
            try:
                summary = summarizer.summarize(transcript_path.read_text())
                transcript_path.write_text(
                    summary + "\n\n---\n\n" + transcript_path.read_text()
                )
            except Exception:
                logging.exception("summarization failed; transcript left as-is")
        final_path = _flatten_session(
            out, transcript_path, session.chunk_files, cfg.sample_rate, args.keep_audio
        )
        logging.info("transcript: %s", final_path)


def _session_dir(args, default_leaf: str) -> Path:
    """Resolve the session directory from --out or --sessions-dir/--name."""
    if args.out:
        return args.out
    date = datetime.now().strftime("%Y-%m-%d")
    leaf = f"{date}-{args.name}" if args.name else f"{date}-{default_leaf}"
    return args.sessions_dir / leaf


def _install_signal_handlers() -> None:
    """Route SIGTERM through the graceful Ctrl-C path; restore SIGINT for bg jobs."""

    def _graceful_stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _graceful_stop)
    signal.signal(signal.SIGINT, signal.default_int_handler)


def _run_note(args) -> None:
    """Mic-only note: capture -> ephemeral transcript -> directive-steered summary."""
    from .summarizer import NOTE_PROMPT, make_summarizer
    from .transcriber import make_backend
    from .worker import TranscriptionWorker

    out = _session_dir(args, "note")
    cfg = Config(
        out_dir=out,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        silence_stop_seconds=args.silence_stop_seconds or float("inf"),
    )

    mic = devices.default_mic()
    logging.info("mic: %s", devices.describe(mic))
    logging.info("session dir: %s", out)

    _install_signal_handlers()

    session = capture.CaptureSession(cfg, devices={capture.MIC: mic})
    transcript_path = out / "transcript.md"
    worker = TranscriptionWorker(
        backend=make_backend(args.model),
        out_path=transcript_path,
        overlap_seconds=cfg.overlap_seconds,
        rms_floor=cfg.transcribe_rms_floor,
        label_speakers=False,
    )
    worker.start()
    session.on_chunk = worker.enqueue

    reason = session.run()
    logging.info("session ended (%s): %d chunks", reason, len(session.chunk_files))
    logging.info("waiting for transcription to finish...")
    worker.finish()

    import shutil

    note_path = _unique_path(out.parent / f"{out.name}.md")
    if not worker.builder.segments:
        logging.warning("no speech transcribed; nothing to summarize")
        shutil.rmtree(out, ignore_errors=True)
        return

    logging.info("summarizing note...")
    summarizer = make_summarizer(args.summary_model)
    summary = summarizer.summarize(transcript_path.read_text(), prompt=NOTE_PROMPT)
    note_path.write_text(summary + "\n")
    shutil.rmtree(out, ignore_errors=True)
    logging.info("note: %s", note_path)


def _unique_path(path: Path) -> Path:
    """Return `path`, or `path` with a -2, -3... suffix if it already exists."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 2
    while (candidate := parent / f"{stem}-{n}{suffix}").exists():
        n += 1
    return candidate


def _flatten_session(out, transcript_path, chunk_files, sample_rate, keep_audio) -> Path:
    """Move transcript.md up to <session-dir>.md.

    Normally the session folder is removed afterwards. With ``keep_audio`` the
    overlapping chunk WAVs are concatenated into a single ``recording.wav`` and
    the folder is kept holding just that file.
    """
    import shutil

    final_path = _unique_path(out.parent / f"{out.name}.md")
    transcript_path.rename(final_path)
    if keep_audio and chunk_files:
        _concat_recording(out / "recording.wav", chunk_files, sample_rate)
        shutil.rmtree(out / "chunks", ignore_errors=True)
    else:
        shutil.rmtree(out)
    return final_path


def _concat_recording(dest, chunk_files, sample_rate) -> None:
    """Reconstruct a stereo WAV from the overlapping per-channel chunks.

    Each chunk is placed at its absolute sample offset (overlaps overwrite with
    identical audio), rebuilding a continuous track per channel. Mic goes to the
    left channel, system to the right -- kept separate so neither can clip.
    """
    import numpy as np

    from .capture import MIC, SYSTEM
    from .wavio import read_wav, write_wav_stereo

    tracks = _reconstruct_tracks(chunk_files, sample_rate, np, read_wav)
    n = max((len(t) for t in tracks.values()), default=0)
    left = tracks.get(MIC, np.zeros(n, dtype=np.float32))
    right = tracks.get(SYSTEM, np.zeros(n, dtype=np.float32))
    write_wav_stereo(dest, left, right, sample_rate)


def _reconstruct_tracks(chunk_files, sample_rate, np, read_wav) -> dict:
    """Place each channel's overlapping chunks into one continuous track."""
    placements: dict[str, list[tuple[int, "object"]]] = {}
    total = 0
    for cf in chunk_files:
        audio = read_wav(cf.path, target_rate=sample_rate)
        start = round(cf.start_seconds * sample_rate)
        placements.setdefault(cf.channel, []).append((start, audio))
        total = max(total, start + len(audio))

    tracks = {}
    for channel, placed in placements.items():
        track = np.zeros(total, dtype=np.float32)
        for start, audio in placed:
            track[start : start + len(audio)] = audio
        tracks[channel] = track
    return tracks


if __name__ == "__main__":
    main()
