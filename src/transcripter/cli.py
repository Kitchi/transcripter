import argparse
import logging
import signal
from datetime import datetime, timedelta
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
        help="Capture only; keep recording.wav, skip all processing.",
    )
    rec.add_argument(
        "--summary-model",
        type=Path,
        default=None,
        help="Summary model (default: platform backend's default -- an MLX model dir "
        "on macOS, a GGUF file on Linux).",
    )
    rec.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help="Summary model context window (llama-cpp/Linux only; ignored on macOS, "
        "where MLX context is fixed by the model).",
    )
    rec.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the post-meeting summary.",
    )
    rec.add_argument(
        "--keep-audio",
        action="store_true",
        help="Retain recording.wav after processing (default: delete).",
    )
    rec.add_argument(
        "--no-aec",
        action="store_true",
        help="Skip echo cancellation on the mic channel.",
    )
    rec.add_argument(
        "--aec-dtd-ratio",
        type=float,
        default=Config.aec_dtd_ratio,
        help="Echo canceller adapts only when mic power is below this fraction of "
        "system power. Raise it if the log reports 0.0 dB ERLE with audible far-end "
        "speech (loud speakers couple more echo into the mic).",
    )
    rec.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip speaker diarization; the far end stays a single 'them'.",
    )
    rec.add_argument(
        "--diarize-threshold",
        type=float,
        default=Config.diarize_threshold,
        help="Speaker clustering threshold; lower splits into more speakers.",
    )
    rec.add_argument(
        "--speakers",
        type=int,
        default=-1,
        help="Known number of far-end speakers (default: detect automatically).",
    )

    note = sub.add_parser(
        "note",
        help="Mic-only voice note: transcribe, then write a directive-steered summary.",
    )
    note.add_argument("--out", type=Path, default=None, help="Full session directory path.")
    note.add_argument("--sessions-dir", type=Path, default=Path.cwd())
    note.add_argument("--name", default=None, help="Note name, prefixed to the date.")
    note.add_argument(
        "--silence-stop-seconds",
        type=float,
        default=45.0,
        help="Stop after this much sustained mic silence (0 disables).",
    )
    note.add_argument("--model", default=None, help="Whisper model (platform default if unset).")
    note.add_argument(
        "--summary-model",
        type=Path,
        default=None,
        help="Summary model (platform default if unset).",
    )
    note.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help="Summary model context window (llama-cpp/Linux only; ignored on macOS).",
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
        silence_stop_seconds=args.silence_stop_seconds or float("inf"),
        aec_dtd_ratio=args.aec_dtd_ratio,
        diarize_threshold=args.diarize_threshold,
        diarize_num_speakers=args.speakers,
    )

    mic = devices.default_mic()
    system = devices.find_system_capture()
    logging.info("mic: %s", devices.describe(mic))
    logging.info("system: %s", devices.describe(system))
    logging.info("session dir: %s", out)

    _install_signal_handlers()

    session = capture.CaptureSession(cfg, devices={capture.MIC: mic, capture.SYSTEM: system})

    reason = session.run()
    logging.info("session ended (%s): %.0fs recorded", reason, session.duration_seconds)

    if args.no_transcribe:
        logging.info("recording: %s", session.recording_path)
        return

    from .pipeline import process_recording
    from .transcriber import make_backend

    transcript = process_recording(
        session.recording_path,
        backend=make_backend(args.model),
        cfg=cfg,
        use_aec=not args.no_aec,
        diarize=not args.no_diarize,
    )

    transcript_path = out / "transcript.md"
    transcript_path.write_text(transcript.render())

    if not args.no_summary and transcript.segments:
        from .summarizer import make_summarizer

        logging.info("summarizing...")
        summarizer = make_summarizer(args.summary_model, n_ctx=args.n_ctx)
        try:
            summary = summarizer.summarize_meeting(transcript_path.read_text())
            transcript_path.write_text(summary + "\n\n---\n\n" + transcript_path.read_text())
        except Exception:
            logging.exception("summarization failed; transcript left as-is")
    if session.started_at and session.ended_at:
        header = _session_header(session.started_at, session.ended_at)
        transcript_path.write_text(header + transcript_path.read_text())
    final_path = _flatten_session(out, transcript_path, args.keep_audio)
    logging.info("transcript: %s", final_path)


def _fmt_duration(delta: timedelta) -> str:
    """Human-readable meeting length, e.g. "48m" or "1h 5m"."""
    total = max(int(delta.total_seconds()), 0)
    h, rem = divmod(total, 3600)
    m = rem // 60
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"


def _session_header(started: datetime, ended: datetime) -> str:
    """Metadata block (start/end/duration) prepended above the notes.

    Hard line breaks (trailing two spaces) keep the three fields on their own
    lines in rendered markdown; a trailing blank line separates it from the
    summary that follows.
    """
    fmt = "%Y-%m-%d %H:%M"
    lines = [
        f"**Started:** {started.strftime(fmt)}",
        f"**Ended:** {ended.strftime(fmt)}",
        f"**Duration:** {_fmt_duration(ended - started)}",
    ]
    return "  \n".join(lines) + "\n\n"


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
    import shutil

    from .pipeline import process_recording
    from .summarizer import NOTE_PROMPT, make_summarizer
    from .transcriber import make_backend

    out = _session_dir(args, "note")
    cfg = Config(
        out_dir=out,
        silence_stop_seconds=args.silence_stop_seconds or float("inf"),
    )

    mic = devices.default_mic()
    logging.info("mic: %s", devices.describe(mic))
    logging.info("session dir: %s", out)

    _install_signal_handlers()

    session = capture.CaptureSession(cfg, devices={capture.MIC: mic})
    reason = session.run()
    logging.info("session ended (%s): %.0fs recorded", reason, session.duration_seconds)

    # Mic-only: no far-end reference to cancel against, and one voice to label.
    transcript = process_recording(
        session.recording_path,
        backend=make_backend(args.model),
        cfg=cfg,
        use_aec=False,
        diarize=False,
        label_speakers=False,
    )

    note_path = _unique_path(out.parent / f"{out.name}.md")
    if not transcript.segments:
        logging.warning("no speech transcribed; nothing to summarize")
        shutil.rmtree(out, ignore_errors=True)
        return

    logging.info("summarizing note...")
    summarizer = make_summarizer(args.summary_model, n_ctx=args.n_ctx)
    summary = summarizer.summarize(transcript.render(), prompt=NOTE_PROMPT)
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


def _flatten_session(out, transcript_path, keep_audio) -> Path:
    """Move transcript.md up to <session-dir>.md.

    Normally the session folder (and the recording inside it) is removed
    afterwards. With ``keep_audio`` the folder is kept, holding just
    ``recording.wav``.
    """
    import shutil

    final_path = _unique_path(out.parent / f"{out.name}.md")
    transcript_path.rename(final_path)
    if keep_audio:
        logging.info("audio retained: %s", out / capture.RECORDING_NAME)
    else:
        shutil.rmtree(out, ignore_errors=True)
    return final_path


if __name__ == "__main__":
    main()
