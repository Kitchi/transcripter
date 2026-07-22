from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a recording session."""

    out_dir: Path
    sample_rate: int = 48_000
    chunk_seconds: float = 30.0
    overlap_seconds: float = 2.0

    # Silence watchdog
    silence_stop_seconds: float = 45.0
    calibration_seconds: float = 3.0
    # Mic must exceed ambient RMS by this factor to count as speech.
    speech_rms_factor: float = 4.0
    # Absolute floor for the mic speech threshold, in case calibration
    # happens in a near-silent room.
    mic_rms_floor: float = 1e-3
    # The system-audio channel is digital silence when nothing plays;
    # anything above this counts as active audio.
    system_rms_threshold: float = 5e-4

    # Emit an RMS status line every this many seconds.
    status_interval_seconds: float = 5.0

    # Chunks below this RMS are skipped before transcription: Whisper
    # hallucinates high-prior phrases ("Thank you") on silence.
    transcribe_rms_floor: float = 1e-3
    # Post-transcription hallucination filter (see filters.py). Segments with a
    # window no_speech_prob above this are dropped (silence fillers)...
    transcribe_no_speech_threshold: float = 0.6
    # ...and those with a compression_ratio above this (repetition loops).
    transcribe_compression_ratio_threshold: float = 2.4

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_seconds * self.sample_rate)

    @property
    def hop_samples(self) -> int:
        return int((self.chunk_seconds - self.overlap_seconds) * self.sample_rate)
