import wave

import numpy as np

from transcripter.wavio import read_wav, write_wav, write_wav_stereo


def test_write_read_roundtrip(tmp_path):
    path = tmp_path / "a.wav"
    samples = np.sin(np.linspace(0, 10, 4800)).astype(np.float32)
    write_wav(path, samples, sample_rate=16_000)
    # Same rate -> no resampling, so values survive within int16 quantization.
    back = read_wav(path, target_rate=16_000)
    np.testing.assert_allclose(back, samples, atol=1e-3)


def test_write_wav_clips_out_of_range(tmp_path):
    path = tmp_path / "loud.wav"
    write_wav(path, np.array([2.0, -2.0], dtype=np.float32), sample_rate=16_000)
    back = read_wav(path, target_rate=16_000)
    assert back.max() <= 1.0 and back.min() >= -1.0


def test_read_wav_decimates(tmp_path):
    path = tmp_path / "hi.wav"
    write_wav(path, np.zeros(4800, dtype=np.float32), sample_rate=48_000)
    back = read_wav(path, target_rate=16_000)
    assert len(back) == 1600  # 48k -> 16k is a factor-of-3 decimation


def test_write_wav_stereo_channels_independent(tmp_path):
    path = tmp_path / "s.wav"
    left = np.full(100, 0.5, dtype=np.float32)
    right = np.full(100, -0.25, dtype=np.float32)
    write_wav_stereo(path, left, right, sample_rate=16_000)
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 2
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    stereo = frames.reshape(-1, 2).astype(np.float32) / 32767
    np.testing.assert_allclose(stereo[:, 0], 0.5, atol=1e-3)
    np.testing.assert_allclose(stereo[:, 1], -0.25, atol=1e-3)


def test_write_wav_stereo_zero_pads_shorter_track(tmp_path):
    path = tmp_path / "pad.wav"
    write_wav_stereo(
        path,
        np.ones(50, dtype=np.float32),
        np.ones(20, dtype=np.float32),
        sample_rate=16_000,
    )
    with wave.open(str(path), "rb") as w:
        assert w.getnframes() == 50  # padded to the longer track
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    right = frames.reshape(-1, 2)[:, 1]
    assert right[20:].max() == 0  # tail of the short channel is silence
