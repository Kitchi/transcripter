"""Speaker diarization of the system channel via sherpa-onnx.

The mic channel is one voice by construction, so only the far end needs
splitting into speakers. We run pyannote's segmentation network and a WeSpeaker
embedding extractor through onnxruntime rather than using pyannote directly:
same segmentation model, but no HuggingFace gate, no access token, and no torch
dependency -- which keeps a bundled build tractable.

Models are fetched once into a cache dir and checksum-verified. Both are MIT
licensed (the segmentation archive ships its LICENSE alongside the weights).
"""

import bz2
import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DIARIZE_RATE = 16_000

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class _Model:
    url: str
    sha256: str
    # Path of the weights relative to the cache dir once unpacked.
    target: str
    member: str | None = None  # file to extract, for archives


# NOTE: the upstream release tag really is spelled "recongition".
_SEGMENTATION = _Model(
    url=f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    sha256="24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    target="sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    member="sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
)
_EMBEDDING = _Model(
    url=f"{_RELEASES}/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B_LM.onnx",
    sha256="e197af7e9d473030cf486b3124149a19bf37014d0e4485e4c70c483b0ec10cb2",
    target="wespeaker_en_voxceleb_CAM++_LM.onnx",
)


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: int


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "transcripter" / "models"


def ensure_models(dest: Path | None = None) -> tuple[Path, Path]:
    """Download and verify both ONNX models if absent. Returns their paths."""
    dest = dest or cache_dir()
    return _ensure(_SEGMENTATION, dest), _ensure(_EMBEDDING, dest)


def _ensure(model: _Model, dest: Path) -> Path:
    path = dest / model.target
    if path.exists():
        return path
    dest.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", model.url.rsplit("/", 1)[-1])
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        blob = Path(tmp) / "download"
        urllib.request.urlretrieve(model.url, blob)  # noqa: S310 (pinned https URL)
        digest = hashlib.sha256(blob.read_bytes()).hexdigest()
        if digest != model.sha256:
            raise RuntimeError(
                f"checksum mismatch for {model.url}\n  expected {model.sha256}\n"
                f"  got      {digest}"
            )
        extracted = _extract(model, blob, Path(tmp))
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(path))
    return path


def _extract(model: _Model, blob: Path, tmp: Path) -> Path:
    if model.member is None:
        return blob
    with bz2.open(blob) as fh, tarfile.open(fileobj=fh) as tar:
        member = tar.getmember(model.member)
        if not member.isfile():
            raise RuntimeError(f"{model.member} is not a regular file in the archive")
        # Extract by hand: no member paths are trusted from the archive itself.
        out = tmp / "member.onnx"
        src = tar.extractfile(member)
        if src is None:
            raise RuntimeError(f"could not read {model.member} from the archive")
        with src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return out


class Diarizer:
    """Wraps sherpa-onnx offline diarization. Model load is deferred to first use."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
        num_speakers: int = -1,
        models_dir: Path | None = None,
    ):
        self.threshold = threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.num_speakers = num_speakers
        self.models_dir = models_dir
        self._impl = None

    def _get(self):
        if self._impl is None:
            import sherpa_onnx  # deferred: pulls onnxruntime

            segmentation, embedding = ensure_models(self.models_dir)
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(segmentation)
                    ),
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding)),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=self.num_speakers, threshold=self.threshold
                ),
                min_duration_on=self.min_duration_on,
                min_duration_off=self.min_duration_off,
            )
            if not config.validate():
                raise RuntimeError("invalid diarization config (missing model files?)")
            self._impl = sherpa_onnx.OfflineSpeakerDiarization(config)
        return self._impl

    def diarize(self, samples: np.ndarray, rate: int = DIARIZE_RATE) -> list[Turn]:
        """Return speaker turns for a mono 16 kHz track, sorted by start time."""
        if not len(samples):
            return []
        impl = self._get()
        if rate != impl.sample_rate:
            raise ValueError(f"diarizer expects {impl.sample_rate} Hz, got {rate}")
        result = impl.process(samples.astype(np.float32)).sort_by_start_time()
        return [Turn(start=r.start, end=r.end, speaker=r.speaker) for r in result]


def speaker_label(index: int) -> str:
    return f"Speaker {index + 1}"


def assign_speakers(segments: list[dict], turns: list[Turn]) -> list[dict]:
    """Label each transcribed segment with the speaker it overlaps most.

    Whisper's segment boundaries and the diarizer's turn boundaries are set
    independently, so they rarely coincide; maximum time overlap is the standard
    reconciliation. A segment spanning a speaker change gets a single label --
    splitting it would need word-level timestamps, which is a later refinement.

    Segments overlapping no turn keep `speaker=None` and fall back to the
    generic channel label.
    """
    if not turns:
        return segments
    labelled = []
    for seg in segments:
        best, best_overlap = None, 0.0
        for turn in turns:
            overlap = min(seg["end"], turn.end) - max(seg["start"], turn.start)
            if overlap > best_overlap:
                best, best_overlap = turn.speaker, overlap
        labelled.append({**seg, "speaker": speaker_label(best) if best is not None else None})
    return labelled
