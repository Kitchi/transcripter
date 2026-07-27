"""Minimal WAV read/write (stdlib + numpy only, 16-bit PCM)."""

import wave
from pathlib import Path

import numpy as np

from .resample import decimate


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 samples in [-1, 1] as a 16-bit PCM WAV."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def write_wav_stereo(
    path: Path, left: np.ndarray, right: np.ndarray, sample_rate: int
) -> None:
    """Write two mono float32 tracks in [-1, 1] as a 16-bit stereo PCM WAV.

    Tracks are zero-padded to equal length; left/right stay independent so
    summing is never needed and neither channel can clip the other.
    """
    n = max(len(left), len(right))
    stereo = np.zeros((n, 2), dtype=np.float32)
    stereo[: len(left), 0] = left
    stereo[: len(right), 1] = right
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
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
    return _to_rate(audio, rate, target_rate)


def read_wav_stereo(
    path: Path, target_rate: int | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read a stereo 16-bit WAV as (left, right, rate), float32 in [-1, 1].

    `target_rate` decimates as `read_wav` does; None keeps the native rate.
    Used to load a session recording, where left is the mic (near end) and
    right is the system audio (the AEC's far-end reference).
    """
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 2 or w.getsampwidth() != 2:
            raise ValueError(
                f"{path.name}: expected 16-bit stereo, got {w.getnchannels()}ch/"
                f"{8 * w.getsampwidth()}-bit"
            )
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    interleaved = raw.astype(np.float32) / 32768.0
    left, right = interleaved[0::2].copy(), interleaved[1::2].copy()
    if target_rate is None or target_rate == rate:
        return left, right, rate
    return _to_rate(left, rate, target_rate), _to_rate(right, rate, target_rate), target_rate


def _to_rate(audio: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    """Decimate to `target_rate`. Only integer factors are supported."""
    if rate == target_rate:
        return audio
    if rate % target_rate:
        raise ValueError(f"cannot decimate {rate} Hz to {target_rate} Hz")
    return decimate(audio, rate // target_rate)
