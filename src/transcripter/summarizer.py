"""Summarization backends + hierarchical map-reduce for long meetings.

Selected by platform (see `make_summarizer`):
- macOS: mlx-lm over a local MLX model (default: gemma-4-e4b).
- Linux: llama-cpp-python over a GGUF model (CUDA if built with it, else CPU).

Short transcripts are summarized single-shot. Long ones that would overflow the
model's context are split into token-bounded windows, each condensed to notes
(map), then the notes are reduced -- recursively when even the notes overflow --
into the final structured summary. See PLAN.md for the design rationale.
"""

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MODEL = Path.home() / ".omlx/models/gemma-4-e4b-it-4bit"

# Linux default: pulled from HuggingFace on first use via llama-cpp's
# from_pretrained. Override with --summary-model (a local .gguf path).
LLAMA_DEFAULT_REPO = "bartowski/gemma-2-2b-it-GGUF"
LLAMA_DEFAULT_FILE = "gemma-2-2b-it-Q4_K_M.gguf"

# Fraction of the usable input budget we actually fill per window/batch. The
# headroom absorbs chat-template overhead and keeps each window loosely packed
# for better summarization quality (very full contexts degrade attention).
FILL = 0.75
# Chat-template wrapper (BOS + role/turn markers + generation prompt) that
# count_tokens, working on the raw string, does not see. MLX measures this
# exactly at load; this small value is the conservative default, used by
# llama-cpp whose template we can't cheaply render to count.
PROMPT_OVERHEAD_TOKENS = 64
# Auto-detected MLX context can be enormous (e.g. 131072 for gemma multimodal
# exports). Feeding that many tokens to a small local model is slow and low
# quality, so clamp the detected value to a sane working ceiling.
MLX_CONTEXT_CAP = 32_768
# Segment lines carried from each map window into the next, so a point split
# across a window boundary still appears whole in one window. NOTES_PROMPT is
# told to merge duplicates, so the overlap is harmless.
MAP_OVERLAP_LINES = 2


def make_summarizer(model: str | Path | None = None, n_ctx: int = 8192):
    """Pick the summarization backend for the current platform.

    `n_ctx` sets the llama-cpp context window (Linux); it is ignored on macOS,
    where MLX's context is fixed by the model export.
    """
    if sys.platform == "darwin":
        return MlxLmSummarizer(model or DEFAULT_MODEL)
    return LlamaCppSummarizer(model, n_ctx=n_ctx)


PROMPT = """\
Below is a meeting transcript. "me" is the local user; "them" is the other side
of the call. Write a concise summary in markdown with these sections:

## Summary
A short paragraph on what the meeting was about and what was concluded.

## Key points
Bulleted list of the substantive points discussed.

## Action items
Bulleted list of concrete follow-ups, each tagged with who owns it (me/them)
if that is clear from the transcript. Write "None." if there are none.

Transcript:

{transcript}
"""

# Map / intermediate-merge prompt: condense a portion of a meeting (raw
# transcript, or already-condensed notes) into terse notes. No final structure
# yet -- that is applied once at the top by PROMPT.
NOTES_PROMPT = """\
Below is one part of a longer meeting ("me" = local user, "them" = the other
side). Write terse notes capturing the substantive points, decisions, and any
action items with their owner (me/them). Preserve specifics -- names, numbers,
dates. Merge obvious duplicates. Do not add headings or preamble; just the notes.

Part:

{transcript}
"""

# Note mode: a single-voice dictated note. The speaker's opening words may steer
# the output format, so the model must both obey that and strip it from the body.
NOTE_PROMPT = """\
Below is a transcript of a spoken note dictated by one person.

The speaker's opening words may be an instruction for how to format the rest --
for example "make a bullet list", "turn this into a to-do list", or "this is a
technical summary for a GitHub ticket". If the opening is such an instruction,
follow it and do NOT include the instruction itself as content. If the opening
is not an instruction, default to a short titled summary followed by bullet
points of the substantive content.

Write clean markdown. Output only the finished note -- no preamble, no mention of
these directions.

Note:

{transcript}
"""


def _pack(items: list[str], counts: list[int], budget: int) -> list[list[str]]:
    """Greedily group items (transcript lines or notes) into batches whose
    combined token count (from the parallel `counts`) stays under `budget`. An
    item larger than `budget` on its own gets its own batch."""
    batches: list[list[str]] = []
    current: list[str] = []
    used = 0
    for item, n in zip(items, counts, strict=True):
        if current and used + n > budget:
            batches.append(current)
            current, used = [], 0
        current.append(item)
        used += n
    if current:
        batches.append(current)
    return batches


def _with_overlap(windows: list[list[str]], k: int) -> list[list[str]]:
    """Prepend the last `k` lines of each window to the next one."""
    if k <= 0 or len(windows) < 2:
        return windows
    out = [windows[0]]
    for prev, cur in zip(windows, windows[1:], strict=False):
        out.append(prev[-k:] + cur)
    return out


def _mlx_context(config: dict) -> int:
    """Model context length. Multimodal gemma exports nest it under text_config
    rather than at the top level, so check both."""
    return int(
        config.get("max_position_embeddings")
        or config.get("text_config", {}).get("max_position_embeddings", 8192)
    )


class Summarizer:
    """Shared map-reduce orchestration over backend primitives.

    Subclasses provide: `complete(prompt_text) -> str`, `count_tokens(text) ->
    int`, and the `context_tokens` / `max_tokens` limits.
    """

    max_tokens: int
    context_tokens: int
    # Overridable so backends (or tests) can tune headroom; see module constants.
    fill = FILL
    prompt_overhead = PROMPT_OVERHEAD_TOKENS

    def complete(self, prompt_text: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Token counts for many strings. Backends may override to tokenize in a
        single call; the default loops over count_tokens."""
        return [self.count_tokens(t) for t in texts]

    # -- budgets ---------------------------------------------------------------

    def _usable(self) -> int:
        """Input tokens available once output + template overhead are reserved."""
        return self.context_tokens - self.max_tokens - self.prompt_overhead

    def _budget(self) -> int:
        """Target fill per window/batch -- a fraction of usable for headroom."""
        return int(self.fill * self._usable())

    def _fits(self, body: str) -> bool:
        return self.count_tokens(PROMPT.format(transcript=body)) <= self._usable()

    # -- entry points ----------------------------------------------------------

    def summarize(self, transcript_md: str, prompt: str = PROMPT) -> str:
        """Single-shot summary (used by note mode)."""
        return self.complete(prompt.format(transcript=transcript_md))

    def summarize_meeting(self, transcript_md: str) -> str:
        """Structured meeting summary, hierarchical when it won't fit one call."""
        if self._fits(transcript_md):
            return self.complete(PROMPT.format(transcript=transcript_md))

        lines = [ln for ln in transcript_md.splitlines() if ln.strip()]
        windows = _pack(lines, self.count_tokens_batch(lines), self._budget())
        windows = _with_overlap(windows, MAP_OVERLAP_LINES)
        log.info("summarizing %d transcript windows (map-reduce)", len(windows))
        notes = [self.complete(NOTES_PROMPT.format(transcript="\n".join(w))) for w in windows]
        return self._reduce(notes)

    def _reduce(self, notes: list[str]) -> str:
        """Reduce condensed notes to the final summary, merging on overflow.

        Loops rather than recurses, and guarantees progress: a batching pass that
        fails to shrink the list (pathologically small context) falls back to
        pairwise merges, so the note count strictly decreases toward one.
        """
        while True:
            joined = "\n\n".join(notes)
            if len(notes) == 1 or self._fits(joined):
                return self.complete(PROMPT.format(transcript=joined))
            batches = _pack(notes, self.count_tokens_batch(notes), self._budget())
            if len(batches) >= len(notes):  # no progress -> force shrinkage
                batches = [notes[i : i + 2] for i in range(0, len(notes), 2)]
            log.info("reducing %d notes into %d batches", len(notes), len(batches))
            notes = [
                self.complete(NOTES_PROMPT.format(transcript="\n\n".join(b))) for b in batches
            ]


class MlxLmSummarizer(Summarizer):
    def __init__(self, model: str | Path = DEFAULT_MODEL, max_tokens: int = 1024):
        self.model_path = Path(model).expanduser()
        self.max_tokens = max_tokens
        self._model = None
        self._tok = None  # HF tokenizer (counting + chat template)
        self._wrapped = None  # mlx_lm TokenizerWrapper (generation)
        self._ctx = 8192

    def _load(self):
        if self._model is not None:
            return
        # Deferred heavy imports, mirroring transcriber.py.
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        from mlx_lm.utils import load_model
        from transformers import AutoTokenizer

        # strict=False: omlx exports materialize the KV weights that gemma-4's
        # shared-KV layers (24+) duplicate from earlier layers; mlx_lm's module
        # doesn't declare them, and ignoring them is lossless.
        self._model, config = load_model(self.model_path, strict=False)
        self._ctx = min(_mlx_context(config), MLX_CONTEXT_CAP)
        hf = AutoTokenizer.from_pretrained(str(self.model_path))
        # Gemma ends chat turns with <turn|>, not <eos>; without it generation
        # runs past the end of the answer.
        eos_ids = {hf.eos_token_id, hf.convert_tokens_to_ids("<turn|>")}
        self._tok = hf
        self._wrapped = TokenizerWrapper(hf, eos_token_ids=eos_ids)
        # Measure the chat-template wrapper (BOS + role markers + generation
        # prompt) once, so the budget math uses the real overhead not a guess.
        self.prompt_overhead = len(
            hf.apply_chat_template(
                [{"role": "user", "content": ""}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )

    @property
    def context_tokens(self) -> int:
        self._load()
        return self._ctx

    def count_tokens(self, text: str) -> int:
        self._load()
        return len(self._tok.encode(text, add_special_tokens=False))

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        self._load()
        if not texts:
            return []
        # One tokenizer call for all strings, vs one per line.
        return [len(ids) for ids in self._tok(texts, add_special_tokens=False)["input_ids"]]

    def complete(self, prompt_text: str) -> str:
        self._load()
        from mlx_lm.generate import generate

        prompt = self._tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return generate(
            self._model, self._wrapped, prompt=prompt, max_tokens=self.max_tokens
        ).strip()


class LlamaCppSummarizer(Summarizer):
    def __init__(
        self, model: str | Path | None = None, max_tokens: int = 1024, n_ctx: int = 8192
    ):
        self.model = model  # local .gguf path, or None for the HF default
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama  # deferred: heavy native import

        # n_gpu_layers=-1 offloads everything to the GPU when the build has
        # CUDA; a CPU-only build ignores it and runs on CPU.
        if self.model is None:
            self._llm = Llama.from_pretrained(
                repo_id=LLAMA_DEFAULT_REPO,
                filename=LLAMA_DEFAULT_FILE,
                n_gpu_layers=-1,
                n_ctx=self.n_ctx,
                verbose=False,
            )
        else:
            self._llm = Llama(
                model_path=str(Path(self.model).expanduser()),
                n_gpu_layers=-1,
                n_ctx=self.n_ctx,
                verbose=False,
            )
        return self._llm

    @property
    def context_tokens(self) -> int:
        return self.n_ctx

    def count_tokens(self, text: str) -> int:
        llm = self._load()
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False))

    def complete(self, prompt_text: str) -> str:
        llm = self._load()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=self.max_tokens,
        )
        return out["choices"][0]["message"]["content"].strip()
