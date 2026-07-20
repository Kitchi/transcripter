"""Transcription backends.

Selected by platform (see `make_backend`):
- macOS: mlx-whisper (Metal via Apple's MLX).
- Linux: faster-whisper (CTranslate2; CUDA if a GPU is present, else CPU).
"""

import sys
from pathlib import Path
from typing import Protocol

from .wavio import read_wav

MLX_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
FASTER_DEFAULT_MODEL = "large-v3-turbo"


class Backend(Protocol):
    def transcribe(self, path: Path) -> list[dict]:
        """Return segments as dicts with 'start', 'end', 'text' (chunk-local seconds)."""
        ...


def make_backend(model: str | None = None) -> Backend:
    """Pick the transcription backend for the current platform."""
    if sys.platform == "darwin":
        return MlxWhisperBackend(model or MLX_DEFAULT_MODEL)
    return FasterWhisperBackend(model or FASTER_DEFAULT_MODEL)


class MlxWhisperBackend:
    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo"):
        self.model = model

    def transcribe(self, path: Path) -> list[dict]:
        import mlx_whisper  # deferred: heavy import, loads Metal

        # Pass samples directly; mlx_whisper's file loader shells out to ffmpeg,
        # which we don't want to depend on.
        samples = read_wav(path)
        result = mlx_whisper.transcribe(samples, path_or_hf_repo=self.model)
        return [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result["segments"]
        ]


class FasterWhisperBackend:
    def __init__(self, model: str = FASTER_DEFAULT_MODEL):
        self.model = model
        self._loaded = None  # lazy: model load is expensive, reuse across chunks

    def _get(self):
        if self._loaded is None:
            from faster_whisper import WhisperModel  # deferred: pulls CTranslate2

            # device/compute "auto": CUDA when a GPU is visible, otherwise CPU.
            self._loaded = WhisperModel(self.model, device="auto", compute_type="auto")
        return self._loaded

    def transcribe(self, path: Path) -> list[dict]:
        # read_wav yields 16 kHz mono float32, which faster-whisper consumes
        # directly (no PyAV/ffmpeg decode needed).
        samples = read_wav(path)
        segments, _info = self._get().transcribe(samples)
        return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
