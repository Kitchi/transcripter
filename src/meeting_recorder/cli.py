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
        help="Session directory (default: sessions/<timestamp>)",
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
        default="mlx-community/whisper-large-v3-turbo",
        help="mlx-whisper model repo (HuggingFace).",
    )
    rec.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Capture only; keep chunk WAVs, skip transcription.",
    )
    rec.add_argument(
        "--keep-audio",
        action="store_true",
        help="Retain chunk WAVs after transcription (default: delete).",
    )

    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    out = args.out or Path("sessions") / datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg = Config(
        out_dir=out,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        silence_stop_seconds=args.silence_stop_seconds or float("inf"),
    )

    mic = devices.default_mic()
    system = devices.find_blackhole()
    logging.info("mic: %s", devices.describe(mic))
    logging.info("system: %s", devices.describe(system))
    logging.info("session dir: %s", out)

    # Route SIGTERM through the same graceful-stop path as Ctrl-C; also restore
    # SIGINT in case we were launched as a background job (where it's ignored).
    def _graceful_stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _graceful_stop)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    session = capture.CaptureSession(cfg, mic_device=mic, system_device=system)

    worker = None
    if not args.no_transcribe:
        from .transcriber import MlxWhisperBackend
        from .worker import TranscriptionWorker

        worker = TranscriptionWorker(
            backend=MlxWhisperBackend(args.model),
            out_path=out / "transcript.md",
            overlap_seconds=cfg.overlap_seconds,
            keep_audio=args.keep_audio,
        )
        worker.start()
        session.on_chunk = worker.enqueue

    reason = session.run()
    logging.info("session ended (%s): %d chunks", reason, len(session.chunk_files))
    if worker is not None:
        logging.info("waiting for transcription to finish...")
        worker.finish()
        logging.info("transcript: %s", out / "transcript.md")


if __name__ == "__main__":
    main()
