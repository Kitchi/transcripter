import numpy as np
import pytest

from transcripter.resample import Decimator, decimate


def _stream(x, factor, block_sizes):
    """Push x through a Decimator in the given block sizes, return the output."""
    d = Decimator(factor)
    out, pos = [], 0
    for n in block_sizes:
        out.append(d.push(x[pos : pos + n]))
        pos += n
    out.append(d.push(x[pos:]))
    out.append(d.flush())
    return np.concatenate(out)


def test_streaming_matches_one_shot_uniform_blocks():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20_000).astype(np.float32)
    got = _stream(x, 3, [1000] * 19)
    np.testing.assert_allclose(got, decimate(x, 3), atol=1e-5)


def test_streaming_matches_one_shot_ragged_blocks():
    """Block sizes that are not multiples of the factor must not shift phase."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(20_000).astype(np.float32)
    got = _stream(x, 3, [7, 1, 913, 4001, 2, 5000, 33])
    np.testing.assert_allclose(got, decimate(x, 3), atol=1e-5)


def test_streaming_matches_one_shot_at_capture_ratio():
    """The real path: 48 kHz -> 16 kHz, blocks the size of a capture poll."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(48_000 * 3).astype(np.float32)
    got = _stream(x, 3, [12_000] * 11)
    np.testing.assert_allclose(got, decimate(x, 3), atol=1e-5)


def test_output_length_tracks_input():
    x = np.zeros(9_000, dtype=np.float32)
    assert len(_stream(x, 3, [500] * 17)) == len(decimate(x, 3)) == 3_000


def test_blocks_smaller_than_the_kernel_are_buffered():
    """A stall-sized trickle of tiny blocks still reconstructs exactly."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(4_000).astype(np.float32)
    np.testing.assert_allclose(_stream(x, 3, [1] * 3999), decimate(x, 3), atol=1e-5)


def test_factor_one_is_passthrough():
    x = np.linspace(-1, 1, 500, dtype=np.float32)
    np.testing.assert_array_equal(_stream(x, 1, [100] * 4), x)


def test_alias_is_suppressed():
    """A tone above the decimated Nyquist must not fold back into the output."""
    rate, factor = 48_000, 3
    t = np.arange(rate) / rate
    # 12 kHz: well above the 8 kHz Nyquist of the 16 kHz output.
    x = np.sin(2 * np.pi * 12_000 * t).astype(np.float32)
    out = _stream(x, factor, [4800] * 10)
    # Skip the filter's edge transients; bare x[::3] would alias to full amplitude.
    assert np.abs(out[200:-200]).max() < 0.005


def test_passband_tone_survives():
    rate, factor = 48_000, 3
    t = np.arange(rate) / rate
    x = np.sin(2 * np.pi * 1_000 * t).astype(np.float32)
    out = _stream(x, factor, [4800] * 10)
    # Ignore the filter's ramp-up/down at the very edges.
    assert np.abs(out[200:-200]).max() > 0.95


def test_rejects_zero_factor():
    with pytest.raises(ValueError):
        Decimator(0)
