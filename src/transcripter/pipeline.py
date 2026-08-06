"""Offline processing of a finished recording -> transcript.

Runs once the meeting has stopped, in order:

    read recording.wav
      -> AEC (subtract system bleed from the mic)
      -> transcribe both channels
      -> hallucination filter
      -> diarize the system channel, label its segments
      -> interleave into transcript.md

Each stage is skippable, and each degrades to a warning rather than losing the
transcript: a diarization failure costs speaker labels, not the meeting.
"""

import logging
from pathlib import Path

import numpy as np

from .aec import cancel_echo, erle_db, measure_echo_delay
from .config import Config
from .filters import drop_hallucinations
from .recorder import RECORDING_RATE
from .transcript import MIC, SYSTEM, Transcript
from .wavio import read_wav, read_wav_stereo

log = logging.getLogger(__name__)


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Read a session recording as (mic, system, rate). System is None if mono."""
    import wave

    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
    if channels == 1:
        audio = read_wav(path, target_rate=RECORDING_RATE)
        return audio, None, RECORDING_RATE
    mic, system, rate = read_wav_stereo(path, target_rate=RECORDING_RATE)
    return mic, system, rate


def process_recording(
    recording_path: Path,
    backend,
    cfg: Config,
    *,
    use_aec: bool = True,
    diarize: bool = True,
    label_speakers: bool = True,
) -> Transcript:
    """Turn a finished recording into a `Transcript`."""
    mic, system, rate = load_recording(recording_path)
    log.info("processing %.0fs of audio", len(mic) / rate if rate else 0)

    if system is not None and use_aec:
        mic = _cancel(mic, system, cfg)

    transcript = Transcript(label_speakers=label_speakers)

    log.info("transcribing mic...")
    transcript.add(MIC, _transcribe(backend, mic, cfg))

    if system is not None:
        log.info("transcribing system audio...")
        segments = _transcribe(backend, system, cfg)
        if diarize and segments:
            segments = _diarize(system, segments, cfg)
        transcript.add(SYSTEM, segments)

    return transcript


def _cancel(mic: np.ndarray, system: np.ndarray, cfg: Config) -> np.ndarray:
    """Measure the echo delay, align, cancel, and put the timeline back.

    The recorder interleaves two independently-clocked streams without
    timestamps, so the mic trails the system channel by an amount that is
    specific to the recording -- 127 ms and 110 ms on the two meetings measured.
    Cancelling without correcting for it costs most of the achievable ERLE, so
    the delay is measured here rather than configured.
    """
    delay = measure_echo_delay(
        mic, system, rate=RECORDING_RATE, min_sharpness=cfg.aec_sharpness_threshold
    )
    if delay.sharpness < cfg.aec_sharpness_threshold:
        # Headphones, or the conferencing app already cancelled it. Adapting
        # against an absent echo path fits mic noise to a loud reference and
        # diverges, so skipping is not just an optimization.
        log.info(
            "no echo path detected (correlation sharpness %.1f < %.1f); "
            "skipping echo cancellation",
            delay.sharpness,
            cfg.aec_sharpness_threshold,
        )
        return mic

    n = min(len(mic), len(system))
    lag = delay.samples
    log.info(
        "cancelling echo (delay %.0f ms, sharpness %.1f)...",
        1000 * lag / RECORDING_RATE,
        delay.sharpness,
    )
    aligned, reference = mic[lag:n], system[: n - lag]
    cleaned = cancel_echo(
        aligned,
        reference,
        block=cfg.aec_block,
        reach=cfg.aec_reach,
        mu=cfg.aec_mu,
        dtd_ratio=cfg.aec_dtd_ratio,
    )
    erle = erle_db(aligned, cleaned, reference, dtd_ratio=cfg.aec_dtd_ratio)
    if erle <= 0.0:
        # There is a real echo path, but the DTD found too few echo-only blocks
        # to converge on -- typically speakers loud enough that the echo alone
        # exceeds aec_dtd_ratio.
        log.warning(
            "echo cancellation removed nothing measurable (ERLE %.1f dB) despite a "
            "clear echo path (sharpness %.1f) -- try a larger --aec-dtd-ratio "
            "(currently %.2f)",
            erle,
            delay.sharpness,
            cfg.aec_dtd_ratio,
        )
    else:
        log.info("echo cancellation: %.1f dB ERLE", erle)
    # Undo the shift, so mic and system timestamps still line up downstream.
    return np.concatenate([mic[:lag], cleaned, mic[n:]])


def _transcribe(backend, samples: np.ndarray, cfg: Config) -> list[dict]:
    if not len(samples):
        return []
    raw = backend.transcribe(samples)
    segments = drop_hallucinations(
        raw,
        no_speech_threshold=cfg.transcribe_no_speech_threshold,
        compression_ratio_threshold=cfg.transcribe_compression_ratio_threshold,
    )
    log.info(
        "  %d segment(s) (%d dropped as hallucination)", len(segments), len(raw) - len(segments)
    )
    return segments


def _diarize(system: np.ndarray, segments: list[dict], cfg: Config) -> list[dict]:
    """Label system segments by speaker; on failure, leave them generic.

    The import is inside the guard too: a missing sherpa-onnx should degrade to
    an unlabelled transcript like any other diarization failure, not lose the
    meeting.
    """
    log.info("diarizing system audio...")
    try:
        from .diarize import Diarizer, assign_speakers

        diarizer = Diarizer(
            threshold=cfg.diarize_threshold,
            min_duration_on=cfg.diarize_min_duration_on,
            min_duration_off=cfg.diarize_min_duration_off,
            num_speakers=cfg.diarize_num_speakers,
        )
        turns = diarizer.diarize(system, RECORDING_RATE)
    except Exception:
        log.exception("diarization failed; falling back to unlabelled 'them'")
        return segments
    speakers = len({t.speaker for t in turns})
    log.info("  %d turn(s), %d speaker(s)", len(turns), speakers)
    return assign_speakers(segments, turns)
