"""Integer-factor decimation, one-shot and streaming.

Capture runs at the device rate (48 kHz); everything downstream -- AEC, Whisper,
speaker embeddings -- wants 16 kHz, so the recorder decimates on the way to disk.
A naive ``x[::factor]`` aliases, so both paths low-pass with the same
windowed-sinc kernel first. `Decimator` is the streaming form: it carries the
filter tail across blocks so the result is sample-identical to `decimate` run
over the whole signal at once.
"""

import numpy as np


def sinc_kernel(factor: int, taps_per_phase: int = 8) -> np.ndarray:
    """Windowed-sinc low-pass at the decimated Nyquist, normalized to unit gain."""
    taps = 4 * factor * taps_per_phase + 1  # odd: symmetric about a sample
    t = np.arange(taps) - taps // 2
    kernel = np.sinc(t / factor) * np.hamming(taps)
    return (kernel / kernel.sum()).astype(np.float32)


def decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Low-pass and downsample by `factor` in one shot."""
    if factor == 1:
        return x.astype(np.float32)
    kernel = sinc_kernel(factor)
    return np.convolve(x, kernel, mode="same")[::factor].astype(np.float32)


class Decimator:
    """Streaming `decimate`: feed arbitrary blocks, get the same samples out.

    Output sample *n* is centred on input sample ``n * factor``, so it needs
    ``half`` inputs either side of it. We therefore hold back the trailing
    ``half`` samples of each block until the next one arrives, and prime the
    history with zeros so the first outputs match the zero-padded one-shot
    result exactly. `flush` zero-pads the tail the same way to drain the rest.
    """

    def __init__(self, factor: int, taps_per_phase: int = 8):
        if factor < 1:
            raise ValueError("factor must be >= 1")
        self.factor = factor
        self.kernel = sinc_kernel(factor, taps_per_phase)
        self.half = len(self.kernel) // 2
        # Absolute input indices let output positions stay on one continuous
        # axis, so block boundaries never shift the decimation phase.
        self._buf = np.zeros(self.half, dtype=np.float32)
        self._base = -self.half  # absolute input index of _buf[0]
        self._next = 0  # absolute input index of the next output sample

    def push(self, x: np.ndarray) -> np.ndarray:
        if self.factor == 1:
            return x.astype(np.float32)
        return self._emit(np.concatenate([self._buf, x.astype(np.float32).ravel()]))

    def flush(self) -> np.ndarray:
        """Drain the held-back tail, zero-padding as the one-shot path does."""
        if self.factor == 1:
            return np.empty(0, dtype=np.float32)
        pad = np.zeros(self.half, dtype=np.float32)
        return self._emit(np.concatenate([self._buf, pad]))

    def _emit(self, buf: np.ndarray) -> np.ndarray:
        taps = len(self.kernel)
        if len(buf) < taps:
            self._buf = buf
            return np.empty(0, dtype=np.float32)
        # valid[i] is the centred output for buf index i + half.
        valid = np.convolve(buf, self.kernel, mode="valid")
        first = self._base + self.half  # absolute index of valid[0]
        last = self._base + len(buf) - 1 - self.half
        positions = np.arange(self._next, last + 1, self.factor)
        if positions.size:
            out = valid[positions - first].astype(np.float32)
            self._next = int(positions[-1]) + self.factor
        else:
            out = np.empty(0, dtype=np.float32)
        # Retain only what the next output still needs to look back at.
        keep = min(max(self._next - self.half - self._base, 0), len(buf))
        self._buf = buf[keep:]
        self._base += keep
        return out
