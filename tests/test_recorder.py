import wave

import numpy as np
import pytest

from transcripter.recorder import Recorder
from transcripter.resample import decimate

MIC, SYSTEM = "mic", "system"
SRC = 48_000
DST = 16_000


def _read(path):
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return raw.reshape(-1, nch).astype(np.float32) / 32767, rate


def test_writes_stereo_at_target_rate(tmp_path):
    r = Recorder(tmp_path / "rec.wav", [MIC, SYSTEM], SRC, DST)
    r.write({MIC: np.full(48_000, 0.5, np.float32), SYSTEM: np.full(48_000, -0.25, np.float32)})
    r.close()

    audio, rate = _read(tmp_path / "rec.wav")
    assert rate == DST
    assert audio.shape[1] == 2
    assert audio.shape[0] == pytest.approx(16_000, abs=2)
    # Skip filter edge transients on the constant signal.
    np.testing.assert_allclose(audio[100:-100, 0], 0.5, atol=1e-3)
    np.testing.assert_allclose(audio[100:-100, 1], -0.25, atol=1e-3)


def test_mic_is_left_system_is_right(tmp_path):
    r = Recorder(tmp_path / "rec.wav", [MIC, SYSTEM], SRC, DST)
    r.write({MIC: np.full(9_000, 0.8, np.float32), SYSTEM: np.zeros(9_000, np.float32)})
    r.close()

    audio, _ = _read(tmp_path / "rec.wav")
    assert np.abs(audio[:, 0]).max() > 0.5
    assert np.abs(audio[:, 1]).max() == 0


def test_content_matches_one_shot_decimation(tmp_path):
    """Chopping the input into ragged polls must not change the written audio."""
    rng = np.random.default_rng(0)
    mic = rng.standard_normal(48_000).astype(np.float32) * 0.3
    r = Recorder(tmp_path / "rec.wav", [MIC], SRC, DST)
    pos = 0
    for n in [1000, 7, 12_000, 3, 20_000, 14_990]:
        r.write({MIC: mic[pos : pos + n]})
        pos += n
    r.close()

    audio, _ = _read(tmp_path / "rec.wav")
    expected = decimate(mic, 3)
    np.testing.assert_allclose(audio[:, 0], expected[: len(audio)], atol=2e-3)


def test_unequal_channel_arrival_stays_aligned(tmp_path):
    """A channel that lags then catches up must not shift the other in time."""
    rng = np.random.default_rng(1)
    mic = rng.standard_normal(48_000).astype(np.float32) * 0.3
    sysd = rng.standard_normal(48_000).astype(np.float32) * 0.3

    r = Recorder(tmp_path / "rec.wav", [MIC, SYSTEM], SRC, DST)
    # System stalls for the first two polls, then delivers its backlog.
    r.write({MIC: mic[:16_000], SYSTEM: np.empty(0, np.float32)})
    r.write({MIC: mic[16_000:32_000], SYSTEM: np.empty(0, np.float32)})
    r.write({MIC: mic[32_000:], SYSTEM: sysd})
    r.close()

    audio, _ = _read(tmp_path / "rec.wav")
    np.testing.assert_allclose(audio[:, 0], decimate(mic, 3)[: len(audio)], atol=2e-3)
    np.testing.assert_allclose(audio[:, 1], decimate(sysd, 3)[: len(audio)], atol=2e-3)


def test_close_flushes_ragged_tail(tmp_path):
    """Channels ending at different lengths still write every frame captured."""
    r = Recorder(tmp_path / "rec.wav", [MIC, SYSTEM], SRC, DST)
    r.write({MIC: np.full(48_000, 0.4, np.float32), SYSTEM: np.full(30_000, 0.4, np.float32)})
    seconds = r.close()

    audio, _ = _read(tmp_path / "rec.wav")
    assert len(audio) == pytest.approx(16_000, abs=2)  # the longer channel survives
    assert seconds == pytest.approx(1.0, abs=1e-3)


def test_empty_session_writes_valid_empty_wav(tmp_path):
    r = Recorder(tmp_path / "rec.wav", [MIC, SYSTEM], SRC, DST)
    assert r.close() == 0.0
    audio, rate = _read(tmp_path / "rec.wav")
    assert len(audio) == 0
    assert rate == DST


def test_mono_session(tmp_path):
    r = Recorder(tmp_path / "rec.wav", [MIC], SRC, DST)
    r.write({MIC: np.full(9_000, 0.6, np.float32)})
    r.close()
    audio, _ = _read(tmp_path / "rec.wav")
    assert audio.shape[1] == 1


def test_clipping_is_bounded(tmp_path):
    r = Recorder(tmp_path / "rec.wav", [MIC], SRC, DST)
    r.write({MIC: np.full(9_000, 4.0, np.float32)})
    r.close()
    audio, _ = _read(tmp_path / "rec.wav")
    assert audio.max() <= 1.0


def test_rejects_non_integer_rate_ratio(tmp_path):
    with pytest.raises(ValueError):
        Recorder(tmp_path / "rec.wav", [MIC], 44_100, 16_000)


def test_rejects_no_channels(tmp_path):
    with pytest.raises(ValueError):
        Recorder(tmp_path / "rec.wav", [], SRC, DST)
