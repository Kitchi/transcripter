"""Pure transcript assembly: overlap dedup, channel interleave, markdown rendering."""

from dataclasses import dataclass, field

CHANNEL_LABELS = {"mic": "me", "system": "them"}


@dataclass(frozen=True)
class Segment:
    channel: str
    start: float  # seconds from session start
    end: float
    text: str


@dataclass
class TranscriptBuilder:
    overlap_seconds: float
    segments: list[Segment] = field(default_factory=list)

    def add_chunk(
        self,
        channel: str,
        chunk_index: int,
        chunk_start_seconds: float,
        raw_segments: list[dict],
    ) -> None:
        """Add one transcribed chunk. `raw_segments` have chunk-local start/end/text.

        Segments whose midpoint falls in the leading overlap region were already
        covered by the previous chunk and are dropped (except for chunk 0, which
        has no predecessor).
        """
        for seg in raw_segments:
            text = seg["text"].strip()
            if not text:
                continue
            midpoint = (seg["start"] + seg["end"]) / 2
            if chunk_index > 0 and midpoint < self.overlap_seconds:
                continue
            self.segments.append(
                Segment(
                    channel=channel,
                    start=chunk_start_seconds + seg["start"],
                    end=chunk_start_seconds + seg["end"],
                    text=text,
                )
            )

    def render(self) -> str:
        lines = ["# Transcript", ""]
        for seg in sorted(self.segments, key=lambda s: (s.start, s.channel)):
            label = CHANNEL_LABELS.get(seg.channel, seg.channel)
            lines.append(f"`[{_fmt(seg.start)}]` **{label}**: {seg.text}")
            lines.append("")
        return "\n".join(lines)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
