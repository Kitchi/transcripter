"""Transcription backends. Default: mlx-whisper (Metal-accelerated on Apple Silicon)."""

from pathlib import Path
from typing import Protocol

from .wavio import read_wav


class Backend(Protocol):
    def transcribe(self, path: Path) -> list[dict]:
        """Return segments as dicts with 'start', 'end', 'text' (chunk-local seconds)."""
        ...


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
