"""Transcription backends.

Selected by platform (see `make_backend`):
- macOS: mlx-whisper (Metal via Apple's MLX).
- Linux: faster-whisper (CTranslate2; CUDA if a GPU is present, else CPU).
"""

import sys
from typing import Protocol

import numpy as np

MLX_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
FASTER_DEFAULT_MODEL = "large-v3-turbo"


class Backend(Protocol):
    def transcribe(self, samples: np.ndarray) -> list[dict]:
        """Transcribe mono 16 kHz float32 audio.

        Returns segments as dicts with 'start', 'end', 'text' in seconds from
        the start of `samples`.
        """
        ...


def make_backend(model: str | None = None) -> Backend:
    """Pick the transcription backend for the current platform."""
    if sys.platform == "darwin":
        return MlxWhisperBackend(model or MLX_DEFAULT_MODEL)
    return FasterWhisperBackend(model or FASTER_DEFAULT_MODEL)


class MlxWhisperBackend:
    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo"):
        self.model = model

    def transcribe(self, samples: np.ndarray) -> list[dict]:
        import mlx_whisper  # deferred: heavy import, loads Metal

        # Samples go in directly; mlx_whisper's file loader shells out to
        # ffmpeg, which we don't want to depend on.
        # condition_on_previous_text=False: stop a hallucinated phrase from
        # seeding the rest of the decode.
        result = mlx_whisper.transcribe(
            samples, path_or_hf_repo=self.model, condition_on_previous_text=False
        )
        return [
            {
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "no_speech_prob": s.get("no_speech_prob", 0.0),
                "compression_ratio": s.get("compression_ratio", 0.0),
            }
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

    def transcribe(self, samples: np.ndarray) -> list[dict]:
        # faster-whisper consumes 16 kHz mono float32 directly (no PyAV/ffmpeg
        # decode needed).
        # vad_filter drops non-speech; condition_on_previous_text=False keeps a
        # hallucination from seeding the rest of the decode.
        segments, _info = self._get().transcribe(
            samples, vad_filter=True, condition_on_previous_text=False
        )
        return [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "no_speech_prob": getattr(s, "no_speech_prob", 0.0),
                "compression_ratio": getattr(s, "compression_ratio", 0.0),
            }
            for s in segments
        ]
