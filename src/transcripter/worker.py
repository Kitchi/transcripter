"""Background transcription worker: consumes chunk files, maintains transcript.md."""

import logging
import queue
import threading
import time
from pathlib import Path

from .capture import ChunkFile, rms
from .transcriber import Backend
from .transcript import TranscriptBuilder
from .wavio import read_wav

log = logging.getLogger(__name__)

_SENTINEL = None


class TranscriptionWorker:
    def __init__(
        self,
        backend: Backend,
        out_path: Path,
        overlap_seconds: float,
        keep_audio: bool = False,
        rms_floor: float = 0.0,
        label_speakers: bool = True,
    ):
        self.backend = backend
        self.out_path = out_path
        self.keep_audio = keep_audio
        self.rms_floor = rms_floor
        self.builder = TranscriptBuilder(
            overlap_seconds=overlap_seconds, label_speakers=label_speakers
        )
        self._queue: queue.Queue[ChunkFile | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="transcriber", daemon=True)
        self._errors = 0

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, chunk: ChunkFile) -> None:
        self._queue.put(chunk)

    def finish(self) -> None:
        """Block until all queued chunks are transcribed and the worker exits."""
        self._queue.put(_SENTINEL)
        self._thread.join()

    def _run(self) -> None:
        while True:
            chunk = self._queue.get()
            if chunk is _SENTINEL:
                break
            try:
                self._process(chunk)
            except Exception:
                self._errors += 1
                log.exception("failed to transcribe %s (audio retained)", chunk.path.name)

    def _process(self, chunk: ChunkFile) -> None:
        t0 = time.monotonic()
        if self.rms_floor > 0:
            level = rms(read_wav(chunk.path))
            if level < self.rms_floor:
                log.info(
                    "skipped %s: silent (rms=%.5f < %.5f)",
                    chunk.path.name,
                    level,
                    self.rms_floor,
                )
                if not self.keep_audio:
                    chunk.path.unlink(missing_ok=True)
                return
        segments = self.backend.transcribe(chunk.path)
        self.builder.add_chunk(chunk.channel, chunk.index, chunk.start_seconds, segments)
        self.out_path.write_text(self.builder.render())
        if not self.keep_audio:
            chunk.path.unlink(missing_ok=True)
        log.info(
            "transcribed %s: %d segment(s) in %.1fs",
            chunk.path.name,
            len(segments),
            time.monotonic() - t0,
        )
