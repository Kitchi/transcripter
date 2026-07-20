"""Pure chunking logic: accumulate samples, emit fixed-size overlapping windows."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Chunk:
    index: int
    start_sample: int
    samples: np.ndarray  # mono float32


class OverlappingChunker:
    """Accumulates a mono stream and yields windows of `chunk_samples`,
    advancing by `hop_samples` (hop < chunk gives overlap)."""

    def __init__(self, chunk_samples: int, hop_samples: int):
        if not 0 < hop_samples <= chunk_samples:
            raise ValueError("hop_samples must be in (0, chunk_samples]")
        self.chunk_samples = chunk_samples
        self.hop_samples = hop_samples
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0  # absolute sample index of _buffer[0]
        self._next_index = 0

    def push(self, samples: np.ndarray) -> Iterator[Chunk]:
        """Feed samples; yield any chunks that are now complete."""
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32).ravel()])
        while len(self._buffer) >= self.chunk_samples:
            yield Chunk(
                index=self._next_index,
                start_sample=self._buffer_start,
                samples=self._buffer[: self.chunk_samples].copy(),
            )
            self._next_index += 1
            self._buffer = self._buffer[self.hop_samples :]
            self._buffer_start += self.hop_samples

    def flush(self) -> Chunk | None:
        """Return the final partial chunk, if it holds any new (non-overlap) audio."""
        new_samples_start = self.chunk_samples - self.hop_samples if self._next_index else 0
        if len(self._buffer) <= new_samples_start:
            return None
        return Chunk(
            index=self._next_index,
            start_sample=self._buffer_start,
            samples=self._buffer.copy(),
        )
