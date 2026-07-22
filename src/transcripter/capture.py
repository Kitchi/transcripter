"""Two-stream capture: default mic + system audio -> overlapping WAV chunks per channel.

Each stream has its own source pushing blocks onto a queue; the session loop
drains both, feeds the chunkers and the silence watchdog, and writes chunk WAVs.

The mic is always a sounddevice input. System audio is a sounddevice input on
Linux (a PulseAudio/PipeWire monitor) and the bundled Core Audio tap helper on
macOS (see `_TapStream`).
"""

import json
import logging
import os
import queue
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

from .chunker import Chunk, OverlappingChunker
from .config import Config
from .devices import SYSTEM_TAP, TAP_APP
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


class _BaseStream:
    """One capture source: pushes mono float32 blocks onto a queue -> chunker."""

    def __init__(self, channel: str, cfg: Config):
        self.channel = channel
        self.cfg = cfg
        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self.chunker = OverlappingChunker(cfg.chunk_samples, cfg.hop_samples)
        self.last_rms = 0.0

    def _push(self, block: np.ndarray) -> None:
        self.queue.put(block)

    def drain(self) -> np.ndarray:
        """Collect all queued blocks into one array (may be empty)."""
        blocks = []
        while True:
            try:
                blocks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if not blocks:
            # No new audio this poll: report silence, not a stale RMS. Otherwise a
            # source that stops emitting during quiet keeps the watchdog armed.
            self.last_rms = 0.0
            return np.empty(0, dtype=np.float32)
        data = np.concatenate(blocks).ravel()
        self.last_rms = rms(data)
        return data

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class _DeviceStream(_BaseStream):
    """A sounddevice input: callback -> queue."""

    def __init__(self, channel: str, device: int, cfg: Config):
        super().__init__(channel, cfg)
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
        self._push(indata.copy().ravel())

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()

    def close(self) -> None:
        self.stream.close()


class _TapStream(_BaseStream):
    """macOS system audio via the Core Audio tap helper app.

    The helper must run as its own TCC-responsible process to hold the
    system-audio-recording permission, so we launch it through LaunchServices
    (`open`) rather than spawning the binary directly -- a direct child inherits
    the terminal's identity and the tap silently yields zeroed audio. Because
    `open` detaches stdio, the helper connects back to a unix-domain socket we
    listen on: one JSON header line, then interleaved float32 PCM. We downmix to
    mono, resample if the tap's rate differs, and push blocks like any source.
    """

    _ACCEPT_TIMEOUT = 20.0  # seconds to wait for the helper to connect
    _HEADER_TIMEOUT = 5.0  # seconds to wait for the format header after connect
    _RECV_BYTES = 1 << 16

    def __init__(self, channel: str, cfg: Config):
        super().__init__(channel, cfg)
        self._sock_dir: str | None = None
        self._sock_path: str | None = None
        self._listener: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._running = False
        self._src_rate = cfg.sample_rate
        self._channels = 2
        # Streaming-resample state (carried across blocks; see _resample).
        self._rs_tail = np.empty(0, dtype=np.float32)
        self._rs_pos = 0.0

    def start(self) -> None:
        # mkdtemp holds a private dir (no name-generation race like mktemp), and
        # /tmp keeps sun_path short (~104 bytes) vs macOS's long default TMPDIR.
        self._sock_dir = tempfile.mkdtemp(prefix="tap-", dir="/tmp")
        self._sock_path = os.path.join(self._sock_dir, "s")
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self._sock_path)
        self._listener.listen(1)
        self._listener.settimeout(self._ACCEPT_TIMEOUT)

        # `open -n`: new instance, launched by LaunchServices as its own
        # responsible process. Returns immediately; the app connects back.
        subprocess.run(
            ["/usr/bin/open", "-n", str(TAP_APP), "--args", self._sock_path],
            check=True,
        )
        try:
            self._conn, _ = self._listener.accept()
        except TimeoutError as e:
            raise RuntimeError(
                "system tap helper did not connect -- launch or permission failure "
                "(grant System Audio Recording in System Settings > Privacy & Security)"
            ) from e

        # Bound the header read so a connected-but-mute helper can't hang us;
        # the read loop below wants a blocking socket, so clear it afterward.
        self._conn.settimeout(self._HEADER_TIMEOUT)
        self._read_header()
        self._conn.settimeout(None)
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _recv_line(self) -> bytes:
        buf = b""
        while b"\n" not in buf:
            b = self._conn.recv(1)
            if not b:
                break
            buf += b
        return buf

    def _read_header(self) -> None:
        try:
            line = self._recv_line().decode("utf-8", "replace").strip()
        except TimeoutError as e:
            raise RuntimeError(
                f"system tap helper connected but sent no format header within "
                f"{self._HEADER_TIMEOUT:.0f}s"
            ) from e
        try:
            hdr = json.loads(line)
            self._src_rate = int(hdr["sampleRate"])
            self._channels = int(hdr["channels"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(
                f"system tap helper did not report a valid format header: {line!r}"
            ) from e
        log.info(
            "system tap: %d Hz, %d ch (resample->%d Hz: %s)",
            self._src_rate,
            self._channels,
            self.cfg.sample_rate,
            self._src_rate != self.cfg.sample_rate,
        )

    def _read_loop(self) -> None:
        frame_bytes = self._channels * 4  # float32
        leftover = b""
        while self._running:
            try:
                data = self._conn.recv(self._RECV_BYTES)
            except OSError:
                break
            if not data:
                break
            buf = leftover + data
            nframes = len(buf) // frame_bytes
            if not nframes:
                leftover = buf
                continue
            usable = buf[: nframes * frame_bytes]
            leftover = buf[nframes * frame_bytes :]
            frames = np.frombuffer(usable, dtype=np.float32).reshape(-1, self._channels)
            mono = frames.mean(axis=1).astype(np.float32)
            if self._src_rate != self.cfg.sample_rate:
                mono = self._resample(mono)
            if len(mono):
                self._push(mono)

    def _resample(self, x: np.ndarray) -> np.ndarray:
        """Linear resample with phase carried across blocks (no per-block reset).

        Only the read-loop thread touches the ``_rs_*`` state, so no locking is
        needed. Output-sample positions advance on one continuous source-time
        axis; retained tail samples let interpolation span block boundaries.
        """
        src, dst = self._src_rate, self.cfg.sample_rate
        buf = np.concatenate([self._rs_tail, x]) if self._rs_tail.size else x
        n = len(buf)
        step = src / dst
        # Emit outputs only where both interpolation neighbours exist (pos <= n-1).
        pos = np.arange(self._rs_pos, n - 1 + 1e-9, step)
        if pos.size == 0:
            self._rs_tail = buf
            return np.empty(0, dtype=np.float32)
        out = np.interp(pos, np.arange(n), buf).astype(np.float32)
        next_pos = pos[-1] + step
        keep = min(int(np.floor(next_pos)), n)
        self._rs_tail = buf[keep:].astype(np.float32)
        self._rs_pos = next_pos - keep
        return out

    def stop(self) -> None:
        self._running = False
        # Closing our end makes the helper's next send() fail, so it exits.
        if self._conn:
            self._conn.close()
        if self._reader:
            self._reader.join(timeout=2)

    def close(self) -> None:
        if self._listener:
            self._listener.close()
        if self._sock_dir and os.path.isdir(self._sock_dir):
            shutil.rmtree(self._sock_dir, ignore_errors=True)


def _make_stream(channel: str, device, cfg: Config) -> _BaseStream:
    if device == SYSTEM_TAP:
        return _TapStream(channel, cfg)
    return _DeviceStream(channel, device, cfg)


class CaptureSession:
    def __init__(self, cfg: Config, devices: dict[str, object]):
        """`devices` maps channel name (MIC/SYSTEM) to a capture source.

        A source is a sounddevice index, or the SYSTEM_TAP sentinel for the macOS
        Core Audio tap. Mic-only sessions (note mode) pass just `{MIC: idx}`; the
        watchdog then decides activity from the mic alone.
        """
        self.cfg = cfg
        self.chunk_dir = cfg.out_dir / "chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.streams = {ch: _make_stream(ch, dev, cfg) for ch, dev in devices.items()}
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
        try:
            # Inside the try so a failed start (e.g. tap permission/launch error)
            # still runs the finally: any already-started stream is stopped and
            # its socket/temp files are cleaned up.
            for s in self.streams.values():
                s.start()
            log.info("recording (Ctrl-C to stop)")
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
                s.stop()
                s.close()
                final = s.chunker.flush()
                if final is not None:
                    self._write_chunk(channel, final)
        return stop_reason
