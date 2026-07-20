import numpy as np
import pytest

from transcripter.chunker import OverlappingChunker


def collect(chunker, samples, block=1000):
    chunks = []
    for i in range(0, len(samples), block):
        chunks.extend(chunker.push(samples[i : i + block]))
    return chunks


def test_emits_overlapping_windows():
    c = OverlappingChunker(chunk_samples=100, hop_samples=80)
    samples = np.arange(300, dtype=np.float32)
    chunks = collect(c, samples, block=37)
    assert [ch.start_sample for ch in chunks] == [0, 80, 160]
    for ch in chunks:
        assert len(ch.samples) == 100
        assert ch.samples[0] == ch.start_sample  # content matches position


def test_overlap_region_shared_between_chunks():
    c = OverlappingChunker(chunk_samples=100, hop_samples=80)
    chunks = collect(c, np.arange(200, dtype=np.float32))
    np.testing.assert_array_equal(chunks[0].samples[80:], chunks[1].samples[:20])


def test_flush_returns_partial_tail():
    c = OverlappingChunker(chunk_samples=100, hop_samples=80)
    chunks = collect(c, np.arange(150, dtype=np.float32))
    assert len(chunks) == 1
    tail = c.flush()
    assert tail is not None
    assert tail.start_sample == 80
    assert tail.samples[-1] == 149


def test_flush_skips_pure_overlap_tail():
    c = OverlappingChunker(chunk_samples=100, hop_samples=80)
    chunks = collect(c, np.arange(100, dtype=np.float32))
    assert len(chunks) == 1
    assert c.flush() is None  # remaining 20 samples are all overlap, no new audio


def test_flush_on_short_stream_returns_everything():
    c = OverlappingChunker(chunk_samples=100, hop_samples=80)
    assert collect(c, np.arange(50, dtype=np.float32)) == []
    tail = c.flush()
    assert tail is not None and len(tail.samples) == 50


def test_invalid_hop_rejected():
    with pytest.raises(ValueError):
        OverlappingChunker(chunk_samples=100, hop_samples=0)
    with pytest.raises(ValueError):
        OverlappingChunker(chunk_samples=100, hop_samples=101)
