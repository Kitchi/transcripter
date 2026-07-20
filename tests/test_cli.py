import wave

import numpy as np

from transcripter import cli
from transcripter.capture import MIC, SYSTEM, ChunkFile
from transcripter.wavio import write_wav

RATE = 16_000


def _make_chunk(dir_, channel, index, start_sample, samples):
    path = dir_ / f"{channel}-{index:04d}.wav"
    write_wav(path, samples, RATE)
    return ChunkFile(
        channel=channel,
        index=index,
        start_seconds=start_sample / RATE,
        path=path,
    )


def _read_stereo(path):
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 2
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return frames.reshape(-1, 2).astype(np.float32) / 32767


# ---- _unique_path -------------------------------------------------------

def test_unique_path_untaken(tmp_path):
    p = tmp_path / "x.md"
    assert cli._unique_path(p) == p


def test_unique_path_suffixes_on_collision(tmp_path):
    (tmp_path / "x.md").write_text("")
    (tmp_path / "x-2.md").write_text("")
    assert cli._unique_path(tmp_path / "x.md") == tmp_path / "x-3.md"


# ---- _reconstruct_tracks ------------------------------------------------

def test_reconstruct_overlap_is_seamless(tmp_path):
    # Two overlapping chunks of a ramp: chunk0 [0:100], chunk1 [80:180].
    ramp = np.linspace(0, 0.9, 180, dtype=np.float32)
    chunks = [
        _make_chunk(tmp_path, MIC, 0, 0, ramp[0:100]),
        _make_chunk(tmp_path, MIC, 1, 80, ramp[80:180]),
    ]
    tracks = cli._reconstruct_tracks(chunks, RATE, np, _read_native)
    np.testing.assert_allclose(tracks[MIC], ramp, atol=1e-3)


# ---- _concat_recording --------------------------------------------------

def test_concat_puts_mic_left_system_right(tmp_path):
    dest = tmp_path / "rec.wav"
    chunks = [
        _make_chunk(tmp_path, MIC, 0, 0, np.full(100, 0.4, dtype=np.float32)),
        _make_chunk(tmp_path, SYSTEM, 0, 0, np.full(100, -0.3, dtype=np.float32)),
    ]
    cli._concat_recording(dest, chunks, RATE)
    stereo = _read_stereo(dest)
    np.testing.assert_allclose(stereo[:, 0], 0.4, atol=1e-3)
    np.testing.assert_allclose(stereo[:, 1], -0.3, atol=1e-3)


def test_concat_single_channel_leaves_other_silent(tmp_path):
    dest = tmp_path / "rec.wav"
    chunks = [_make_chunk(tmp_path, MIC, 0, 0, np.ones(100, dtype=np.float32))]
    cli._concat_recording(dest, chunks, RATE)
    stereo = _read_stereo(dest)
    assert stereo[:, 1].max() == 0  # no system track


# ---- _flatten_session ---------------------------------------------------

def test_flatten_default_removes_folder(tmp_path):
    out = tmp_path / "2026-07-20-meeting"
    (out / "chunks").mkdir(parents=True)
    transcript = out / "transcript.md"
    transcript.write_text("hello")

    final = cli._flatten_session(out, transcript, [], RATE, keep_audio=False)

    assert final == tmp_path / "2026-07-20-meeting.md"
    assert final.read_text() == "hello"
    assert not out.exists()


def test_flatten_keep_audio_writes_recording_and_keeps_folder(tmp_path):
    out = tmp_path / "2026-07-20-meeting"
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(parents=True)
    transcript = out / "transcript.md"
    transcript.write_text("hello")
    chunks = [_make_chunk(chunk_dir, MIC, 0, 0, np.ones(100, dtype=np.float32))]

    final = cli._flatten_session(out, transcript, chunks, RATE, keep_audio=True)

    assert final.read_text() == "hello"
    assert (out / "recording.wav").exists()
    assert not chunk_dir.exists()  # loose chunks cleaned up
    assert not transcript.exists()  # moved out


def test_flatten_collision_gets_suffix(tmp_path):
    (tmp_path / "2026-07-20-meeting.md").write_text("old")
    out = tmp_path / "2026-07-20-meeting"
    out.mkdir()
    transcript = out / "transcript.md"
    transcript.write_text("new")

    final = cli._flatten_session(out, transcript, [], RATE, keep_audio=False)

    assert final == tmp_path / "2026-07-20-meeting-2.md"


def _read_native(path, target_rate):
    """read_wav at native rate (no resample), for reconstruction tests."""
    from transcripter.wavio import read_wav

    return read_wav(path, target_rate=RATE)
