from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Runtime configuration for a recording session."""

    out_dir: Path
    sample_rate: int = 48_000

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

    # Post-transcription hallucination filter (see filters.py). Segments with a
    # window no_speech_prob above this are dropped (silence fillers)...
    transcribe_no_speech_threshold: float = 0.6
    # ...and those with a compression_ratio above this (repetition loops).
    transcribe_compression_ratio_threshold: float = 2.4

    # Echo cancellation (see aec.py). Filter length in taps at the recording
    # rate: 4096 @ 16 kHz covers ~256 ms of echo tail, comfortably more than
    # the output + acoustic delay of speakers-to-mic.
    aec_taps: int = 4096
    aec_mu: float = 0.2
    # Adapt only on blocks that look echo-only (near power below this fraction
    # of far power); holding the filter through double-talk stops it
    # mis-adapting and injecting far-end audio into the mic.
    aec_dtd_ratio: float = 0.05
    # Skip AEC entirely below this mic/system coherence: with headphones (or an
    # app that already cancels) there is no echo path, and adapting against one
    # that isn't there fits noise and diverges. Real room coupling sits well
    # above 0.1 even when the echo is faint.
    aec_coherence_threshold: float = 0.1

    # Diarization (see diarize.py). Clustering threshold: smaller splits more
    # readily into distinct speakers. A conference mix reaches the diarizer
    # through a different codec and AGC per participant, which scatters the
    # embeddings badly -- 0.5 found 47 speakers in a 2-speaker far end, so the
    # default errs toward merging. Pass --speakers when you know the count;
    # that beats any threshold.
    diarize_threshold: float = 0.8
    # Known far-end speaker count; -1 detects it from the clustering threshold.
    # Note this counts the *far end* only: you are on the mic channel.
    diarize_num_speakers: int = -1
    # Speaker turns shorter than this are discarded as segmentation noise. Much
    # below ~0.7 s there is too little audio for a usable speaker embedding, so
    # short turns land arbitrarily in the cluster space and seed false speakers.
    diarize_min_duration_on: float = 0.7
    diarize_min_duration_off: float = 0.5
