"""Drop Whisper hallucinations using its own decode signals plus a text blocklist.

Whisper emits three kinds of junk on this pipeline's audio:

1. **Silence fillers** — confident high-prior phrases ("Thank you.") on non-speech.
   Signalled by a high ``no_speech_prob`` when the decode window is truly silent.
2. **Repetition loops** — "I remember I remember I remember ..." where decoding
   degenerates. Signalled by a high ``compression_ratio`` (repetitive text
   compresses well). 2.4 is Whisper's own default threshold for this.
3. **Fillers that leak past (1)** — the same stereotyped phrases when they share
   a decode window with real speech, so the window-level ``no_speech_prob`` is
   low. Caught only by matching the *entire* segment text against a blocklist of
   phrases that are never meaningful as a standalone line.

Signals (1) and (2) are per decode *window* in mlx-whisper, so every segment in a
window shares them; that is coarse but the blocklist mops up what slips through.
"""

import re

# A segment whose *entire* normalized text equals one of these is dropped. These
# are Whisper's non-speech priors, never meaningful as a whole standalone line.
# Real speech that merely contains the words is untouched (match is exact/whole).
# Deliberately excludes ambiguous backchannel ("okay", "yeah", "thanks") that is
# usually genuine.
BLOCKLIST = frozenset(
    {
        "thank you",
        "thank you very much",
        "thank you for watching",
        "thanks for watching",
        "thanks for watching everyone",
        "please subscribe",
        "please subscribe to my channel",
        "you",
        "bye",
        "bye bye",
    }
)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
# CJK / fullwidth ranges: a stray glyph like "底" in an English meeting is garbage.
_CJK = re.compile(r"[　-鿿豈-﫿＀-￯]")
_LATIN = re.compile(r"[a-zA-Z]")


def _normalize(text: str) -> str:
    return _PUNCT.sub("", text.lower()).strip()


def _is_cjk_junk(text: str) -> bool:
    # CJK present with no Latin letters at all -> not real English speech.
    return bool(_CJK.search(text)) and not _LATIN.search(text)


def is_hallucination(
    seg: dict,
    *,
    no_speech_threshold: float,
    compression_ratio_threshold: float,
) -> bool:
    if seg.get("no_speech_prob", 0.0) > no_speech_threshold:
        return True
    if seg.get("compression_ratio", 0.0) > compression_ratio_threshold:
        return True
    if _is_cjk_junk(seg["text"]):
        return True
    return _normalize(seg["text"]) in BLOCKLIST


def drop_hallucinations(
    segments: list[dict],
    *,
    no_speech_threshold: float = 0.6,
    compression_ratio_threshold: float = 2.4,
) -> list[dict]:
    """Filter Whisper hallucinations from a chunk's segments (see module docstring)."""
    return [
        s
        for s in segments
        if not is_hallucination(
            s,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
        )
    ]
