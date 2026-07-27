"""Pure transcript assembly: channel interleave, bleed dedup, markdown rendering."""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

MIC = "mic"
SYSTEM = "system"

ME_LABEL = "me"
THEM_LABEL = "them"

_PUNCT = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    return _PUNCT.sub("", text.lower()).strip()


def _overlaps(a: "Segment", b: "Segment") -> bool:
    return a.start < b.end and b.start < a.end


@dataclass(frozen=True)
class Segment:
    channel: str
    start: float  # seconds from session start
    end: float
    text: str
    speaker: str | None = None  # diarized label; system channel only

    @property
    def label(self) -> str:
        if self.channel == MIC:
            return ME_LABEL
        return self.speaker or THEM_LABEL


@dataclass
class Transcript:
    """Segments from both channels, rendered to markdown in timestamp order."""

    label_speakers: bool = True  # False for single-voice notes: no speaker tags
    # Min normalized-text similarity for a mic segment to count as system bleed.
    cross_channel_text_ratio: float = 0.6
    segments: list[Segment] = field(default_factory=list)

    def add(self, channel: str, raw_segments: list[dict]) -> None:
        """Add a channel's transcribed segments; start/end are session-absolute."""
        for seg in raw_segments:
            text = seg["text"].strip()
            if not text:
                continue
            self.segments.append(
                Segment(
                    channel=channel,
                    start=seg["start"],
                    end=seg["end"],
                    text=text,
                    speaker=seg.get("speaker"),
                )
            )

    def _kept_segments(self) -> list[Segment]:
        """Drop mic segments that echo an overlapping, similar-text system segment.

        Bleed is one-directional: system audio leaves the speakers and re-enters
        the mic, so a mic segment can shadow a system one but never the reverse.
        The mic copy is therefore always the echo to drop. This is a net for
        residual echo the canceller did not fully remove -- with AEC upstream it
        should fire rarely.
        """
        system = [s for s in self.segments if s.channel == SYSTEM]
        return [
            seg
            for seg in self.segments
            if not (seg.channel == MIC and self._is_bleed(seg, system))
        ]

    def _is_bleed(self, mic_seg: Segment, system: list[Segment]) -> bool:
        norm = _normalize(mic_seg.text)
        if not norm:
            return False
        return any(
            _overlaps(mic_seg, sys_seg)
            and SequenceMatcher(None, norm, _normalize(sys_seg.text)).ratio()
            >= self.cross_channel_text_ratio
            for sys_seg in system
        )

    def render(self) -> str:
        lines = ["# Transcript", ""]
        for seg in sorted(self._kept_segments(), key=lambda s: (s.start, s.channel)):
            if self.label_speakers:
                lines.append(f"`[{_fmt(seg.start)}]` **{seg.label}**: {seg.text}")
            else:
                lines.append(f"`[{_fmt(seg.start)}]` {seg.text}")
            lines.append("")
        return "\n".join(lines)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
