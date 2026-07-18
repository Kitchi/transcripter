"""Minimal WAV read/write (stdlib + numpy only, mono int16)."""

import wave
from pathlib import Path

import numpy as np


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 samples in [-1, 1] as a 16-bit PCM WAV."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


WHISPER_RATE = 16_000


def read_wav(path: Path, target_rate: int = WHISPER_RATE) -> np.ndarray:
    """Read a mono 16-bit WAV as float32 in [-1, 1], resampled to `target_rate`.

    Only integer decimation is supported (e.g. 48k -> 16k), which is all the
    capture pipeline produces.
    """
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = samples.astype(np.float32) / 32768.0
    if rate == target_rate:
        return audio
    if rate % target_rate:
        raise ValueError(f"cannot decimate {rate} Hz to {target_rate} Hz")
    factor = rate // target_rate
    # Windowed-sinc low-pass at the new Nyquist before decimating.
    taps = 4 * factor * 8 + 1
    t = np.arange(taps) - taps // 2
    kernel = np.sinc(t / factor) * np.hamming(taps)
    kernel /= kernel.sum()
    return np.convolve(audio, kernel.astype(np.float32), mode="same")[::factor]
