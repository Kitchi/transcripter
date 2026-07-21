"""Two-stream capture: default mic + BlackHole -> overlapping WAV chunks per channel.

Each stream has its own callback pushing blocks onto a queue; the session loop
drains both, feeds the chunkers and the silence watchdog, and writes chunk WAVs.
"""

import logging
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

from .chunker import Chunk, OverlappingChunker
from .config import Config
from .watchdog import SilenceWatchdog, State
from .wavio import write_wav

log = logging.getLogger(__name__)

MIC = "mic"
SYSTEM = "system"


@dataclass(frozen=True)
class ChunkFile:
    channel: str  # MIC or SYSTEM
    index: int
    start_seconds: float
    path: Path


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0


class _Stream:
    """One input device: callback -> queue -> chunker."""

    def __init__(self, channel: str, device: int, cfg: Config):
        self.channel = channel
        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self.chunker = OverlappingChunker(cfg.chunk_samples, cfg.hop_samples)
        self.last_rms = 0.0
        self.stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=cfg.sample_rate,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.warning("%s stream status: %s", self.channel, status)
        self.queue.put(indata.copy())

    def drain(self) -> np.ndarray:
        """Collect all queued blocks into one array (may be empty)."""
        blocks = []
        while True:
            try:
                blocks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if not blocks:
            return np.empty(0, dtype=np.float32)
        data = np.concatenate(blocks).ravel()
        self.last_rms = rms(data)
        return data


class CaptureSession:
    def __init__(self, cfg: Config, devices: dict[str, int]):
        """`devices` maps channel name (MIC/SYSTEM) to a sounddevice index.

        Mic-only sessions (note mode) pass just `{MIC: idx}`; the watchdog then
        decides activity from the mic alone.
        """
        self.cfg = cfg
        self.chunk_dir = cfg.out_dir / "chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.streams = {ch: _Stream(ch, dev, cfg) for ch, dev in devices.items()}
        self.watchdog = SilenceWatchdog(
            silence_stop_seconds=cfg.silence_stop_seconds,
            calibration_seconds=cfg.calibration_seconds,
            speech_rms_factor=cfg.speech_rms_factor,
            mic_rms_floor=cfg.mic_rms_floor,
            system_rms_threshold=cfg.system_rms_threshold,
        )
        self.chunk_files: list[ChunkFile] = []
        self.on_chunk = None  # optional callback(ChunkFile), set by later phases

    def _write_chunk(self, channel: str, chunk: Chunk) -> None:
        path = self.chunk_dir / f"{channel}-{chunk.index:04d}.wav"
        write_wav(path, chunk.samples, self.cfg.sample_rate)
        cf = ChunkFile(
            channel=channel,
            index=chunk.index,
            start_seconds=chunk.start_sample / self.cfg.sample_rate,
            path=path,
        )
        self.chunk_files.append(cf)
        log.info("wrote %s", path.name)
        if self.on_chunk:
            self.on_chunk(cf)

    def run(self) -> str:
        """Record until Ctrl-C or silence watchdog fires. Returns the stop reason."""
        cfg = self.cfg
        poll = 0.25
        last_status = time.monotonic()
        stop_reason = "interrupted"
        for s in self.streams.values():
            s.stream.start()
        log.info("recording (Ctrl-C to stop)")
        try:
            while True:
                time.sleep(poll)
                for channel, s in self.streams.items():
                    data = s.drain()
                    for chunk in s.chunker.push(data):
                        self._write_chunk(channel, chunk)
                sys_stream = self.streams.get(SYSTEM)
                sys_rms = sys_stream.last_rms if sys_stream else None
                if self.watchdog.update(self.streams[MIC].last_rms, sys_rms, poll):
                    log.info("silence watchdog fired (%.0fs quiet)", cfg.silence_stop_seconds)
                    stop_reason = "silence"
                    break
                now = time.monotonic()
                if now - last_status >= cfg.status_interval_seconds:
                    last_status = now
                    log.info(
                        "mic_rms=%.4f sys_rms=%s state=%s",
                        self.streams[MIC].last_rms,
                        f"{sys_rms:.4f}" if sys_rms is not None else "-",
                        self.watchdog.state.name,
                    )
                    if self.watchdog.state is State.ARMED and self.watchdog.silence_elapsed > 0:
                        log.info("quiet for %.0fs", self.watchdog.silence_elapsed)
        except KeyboardInterrupt:
            log.info("stopped by user")
        finally:
            for channel, s in self.streams.items():
                s.stream.stop()
                s.stream.close()
                final = s.chunker.flush()
                if final is not None:
                    self._write_chunk(channel, final)
        return stop_reason
