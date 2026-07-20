"""Summarization backends.

Selected by platform (see `make_summarizer`):
- macOS: mlx-lm over a local MLX model (default: gemma-4-e4b).
- Linux: llama-cpp-python over a GGUF model (CUDA if built with it, else CPU).
"""

import sys
from pathlib import Path

DEFAULT_MODEL = Path.home() / ".omlx/models/gemma-4-e4b-it-4bit"

# Linux default: pulled from HuggingFace on first use via llama-cpp's
# from_pretrained. Override with --summary-model (a local .gguf path).
LLAMA_DEFAULT_REPO = "bartowski/gemma-2-2b-it-GGUF"
LLAMA_DEFAULT_FILE = "gemma-2-2b-it-Q4_K_M.gguf"


def make_summarizer(model: str | Path | None = None):
    """Pick the summarization backend for the current platform."""
    if sys.platform == "darwin":
        return MlxLmSummarizer(model or DEFAULT_MODEL)
    return LlamaCppSummarizer(model)

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


class MlxLmSummarizer:
    def __init__(self, model: str | Path = DEFAULT_MODEL, max_tokens: int = 1024):
        self.model_path = Path(model).expanduser()
        self.max_tokens = max_tokens

    def summarize(self, transcript_md: str) -> str:
        # Deferred heavy imports, mirroring transcriber.py.
        from mlx_lm.generate import generate
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        from mlx_lm.utils import load_model
        from transformers import AutoTokenizer

        # strict=False: omlx exports materialize the KV weights that gemma-4's
        # shared-KV layers (24+) duplicate from earlier layers; mlx_lm's module
        # doesn't declare them, and ignoring them is lossless.
        model, _config = load_model(self.model_path, strict=False)
        hf_tok = AutoTokenizer.from_pretrained(str(self.model_path))
        # Gemma ends chat turns with <turn|>, not <eos>; without it generation
        # runs past the end of the answer.
        eos_ids = {hf_tok.eos_token_id, hf_tok.convert_tokens_to_ids("<turn|>")}
        tokenizer = TokenizerWrapper(hf_tok, eos_token_ids=eos_ids)

        prompt = hf_tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(transcript=transcript_md)}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return generate(model, tokenizer, prompt=prompt, max_tokens=self.max_tokens).strip()


class LlamaCppSummarizer:
    def __init__(self, model: str | Path | None = None, max_tokens: int = 1024):
        self.model = model  # local .gguf path, or None for the HF default
        self.max_tokens = max_tokens

    def _load(self):
        from llama_cpp import Llama  # deferred: heavy native import

        # n_gpu_layers=-1 offloads everything to the GPU when the build has
        # CUDA; a CPU-only build ignores it and runs on CPU.
        if self.model is None:
            return Llama.from_pretrained(
                repo_id=LLAMA_DEFAULT_REPO,
                filename=LLAMA_DEFAULT_FILE,
                n_gpu_layers=-1,
                n_ctx=8192,
                verbose=False,
            )
        return Llama(
            model_path=str(Path(self.model).expanduser()),
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False,
        )

    def summarize(self, transcript_md: str) -> str:
        llm = self._load()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": PROMPT.format(transcript=transcript_md)}],
            max_tokens=self.max_tokens,
        )
        return out["choices"][0]["message"]["content"].strip()
