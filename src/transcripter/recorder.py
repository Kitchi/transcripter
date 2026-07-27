"""Incremental session recorder: per-channel blocks -> one interleaved WAV.

Replaces the old chunk-file pipeline. Everything downstream (AEC, Whisper,
diarization) is an offline pass over the finished file, so capture's only job is
to get both channels onto disk continuously and cheaply.

Written as it records rather than buffered in RAM: a 90-minute meeting is ~170 MB
at 16 kHz stereo, and a crash leaves a usable recording instead of nothing.

Channel layout follows the convention the AEC expects: mic left, system right.
Kept as separate channels rather than summed so neither can clip the other, and
so the system track stays usable as the AEC's far-end reference.
"""

import logging
import wave
from pathlib import Path

import numpy as np

from .resample import Decimator

log = logging.getLogger(__name__)

RECORDING_RATE = 16_000


class Recorder:
    """Interleaves per-channel blocks into a 16-bit PCM WAV as they arrive.

    Streams arrive on independent queues and deliver unequal numbers of samples
    per poll, so each channel gets its own pending buffer; only the frames all
    channels can fill are written, and the remainder waits for the next poll.
    `close` flushes what is left, zero-padding the shorter channels.
    """

    def __init__(
        self,
        path: Path,
        channels: list[str],
        src_rate: int,
        dst_rate: int = RECORDING_RATE,
    ):
        if not channels:
            raise ValueError("recorder needs at least one channel")
        if src_rate % dst_rate:
            raise ValueError(f"cannot decimate {src_rate} Hz to {dst_rate} Hz")
        self.path = path
        self.channels = list(channels)
        self.rate = dst_rate
        self.frames_written = 0
        factor = src_rate // dst_rate
        self._decimators = {ch: Decimator(factor) for ch in self.channels}
        self._pending = {ch: np.empty(0, dtype=np.float32) for ch in self.channels}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately held open for the session's lifetime; `close` finalizes it.
        self._wav = wave.open(str(path), "wb")  # noqa: SIM115
        self._wav.setnchannels(len(self.channels))
        self._wav.setsampwidth(2)
        self._wav.setframerate(dst_rate)

    def write(self, blocks: dict[str, np.ndarray]) -> None:
        """Accept one poll's worth of audio per channel (any may be empty)."""
        for ch in self.channels:
            block = blocks.get(ch)
            if block is None or not len(block):
                continue
            decimated = self._decimators[ch].push(block)
            if len(decimated):
                self._pending[ch] = np.concatenate([self._pending[ch], decimated])
        n = min(len(self._pending[ch]) for ch in self.channels)
        if n:
            self._emit(n)

    def close(self) -> float:
        """Flush the decimator tails and any ragged remainder. Returns duration."""
        for ch in self.channels:
            tail = self._decimators[ch].flush()
            if len(tail):
                self._pending[ch] = np.concatenate([self._pending[ch], tail])
        n = max(len(self._pending[ch]) for ch in self.channels)
        if n:
            # Channels can end unequal (streams stop microseconds apart); pad the
            # short ones so the last fraction of a second is not discarded.
            for ch in self.channels:
                short = n - len(self._pending[ch])
                if short > 0:
                    self._pending[ch] = np.concatenate(
                        [self._pending[ch], np.zeros(short, dtype=np.float32)]
                    )
            self._emit(n)
        self._wav.close()
        seconds = self.frames_written / self.rate
        log.info("recording: %s (%.0fs, %d ch)", self.path.name, seconds, len(self.channels))
        return seconds

    def _emit(self, n: int) -> None:
        frame = np.stack([self._pending[ch][:n] for ch in self.channels], axis=1)
        for ch in self.channels:
            self._pending[ch] = self._pending[ch][n:]
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
        self._wav.writeframes(pcm.tobytes())
        self.frames_written += n
