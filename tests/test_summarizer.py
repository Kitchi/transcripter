"""Map-reduce orchestration tests.

Uses a fake backend so no model is loaded: count_tokens is a word count and
complete records every call, classifying it by what it was handed -- the final
structured pass (PROMPT), a map over raw transcript ("word" present), or a merge
over note stand-ins. That lets us assert windowing and reduction from token
budgets alone.
"""

from transcripter.summarizer import (
    NOTE_PROMPT,
    PROMPT,
    LlamaCppSummarizer,
    MlxLmSummarizer,
    Summarizer,
)

NOTE_WORDS = 20  # stand-in note length; also the fake's max_tokens


class FakeSummarizer(Summarizer):
    prompt_overhead = 0  # count_tokens already includes the prompt template

    def __init__(self, context_tokens):
        self._context_tokens = context_tokens
        self.max_tokens = NOTE_WORDS
        self.calls = []  # kind per complete(): "final" | "map" | "merge"

    @property
    def context_tokens(self):
        return self._context_tokens

    def count_tokens(self, text):
        return len(text.split())

    def complete(self, prompt_text):
        if prompt_text.startswith(PROMPT[:20]):
            kind = "final"
        elif "word" in prompt_text:  # raw transcript text -> a map window
            kind = "map"
        else:  # only note stand-ins -> an intermediate merge
            kind = "merge"
        self.calls.append(kind)
        return " ".join(["note"] * NOTE_WORDS)


def transcript(n_lines, words_per_line=40):
    body = " ".join(["word"] * words_per_line)
    return "\n".join(f"`[0:{i:02d}]` {body}" for i in range(n_lines))


def test_short_transcript_is_single_shot():
    s = FakeSummarizer(context_tokens=10_000)
    s.summarize_meeting(transcript(3))
    assert s.calls == ["final"]


def test_long_transcript_maps_then_single_reduce():
    # Several map windows, but the few notes fit one final reduce (no merges).
    s = FakeSummarizer(context_tokens=400)
    s.summarize_meeting(transcript(40))
    assert s.calls.count("map") >= 2
    assert s.calls.count("merge") == 0
    assert s.calls == ["map"] * s.calls.count("map") + ["final"]


def test_reduce_merges_when_notes_overflow():
    # Small context so the map notes themselves overflow one reduce: forces at
    # least one intermediate merge level before the final structured pass.
    s = FakeSummarizer(context_tokens=180)
    s.summarize_meeting(transcript(60))
    assert s.calls.count("map") >= 2
    assert s.calls.count("merge") >= 1
    assert s.calls.count("final") == 1
    assert s.calls[-1] == "final"


def test_degenerate_context_still_terminates():
    # Overhead swamps a tiny context (usable < 0): the progress guard must still
    # drive the reduce to a single final summary instead of looping forever.
    s = FakeSummarizer(context_tokens=10)
    s.prompt_overhead = 512
    s.summarize_meeting(transcript(20))
    assert s.calls.count("final") == 1
    assert s.calls[-1] == "final"


def test_single_shot_used_by_note_path():
    s = FakeSummarizer(context_tokens=10_000)
    s.summarize("a dictated note", prompt=NOTE_PROMPT)
    assert len(s.calls) == 1


# -- backend load caching --------------------------------------------------
#
# Map-reduce calls complete() once per window/batch; each backend must load its
# model only on the first call. These assert the cache guard short-circuits
# _load() *without* touching the heavy deferred imports -- so they run on any
# platform (mlx_lm / llama_cpp need not be installed): a broken guard would fall
# through to the import and raise.


def test_mlx_load_is_cached():
    s = MlxLmSummarizer(model="/does/not/exist")
    s._model = object()  # pretend already loaded
    s._tok = s._wrapped = object()
    s._load()  # must return immediately; no import, no model load


def test_llama_load_is_cached():
    s = LlamaCppSummarizer(model="/does/not/exist")
    sentinel = object()
    s._llm = sentinel
    assert s._load() is sentinel
