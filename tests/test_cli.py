from datetime import datetime, timedelta

from transcripter import cli
from transcripter.capture import RECORDING_NAME

# ---- _unique_path -------------------------------------------------------

def test_unique_path_untaken(tmp_path):
    p = tmp_path / "x.md"
    assert cli._unique_path(p) == p


def test_unique_path_suffixes_on_collision(tmp_path):
    (tmp_path / "x.md").write_text("")
    (tmp_path / "x-2.md").write_text("")
    assert cli._unique_path(tmp_path / "x.md") == tmp_path / "x-3.md"


# ---- _fmt_duration / _session_header ------------------------------------

def test_fmt_duration_minutes_only():
    assert cli._fmt_duration(timedelta(minutes=48)) == "48m"


def test_fmt_duration_hours_and_minutes():
    assert cli._fmt_duration(timedelta(hours=1, minutes=5)) == "1h 5m"


def test_fmt_duration_whole_hour():
    assert cli._fmt_duration(timedelta(hours=2)) == "2h"


def test_fmt_duration_negative_clamps_to_zero():
    assert cli._fmt_duration(timedelta(seconds=-10)) == "0m"


def test_session_header_fields_and_trailing_blank_line():
    started = datetime(2026, 7, 22, 14, 3)
    ended = datetime(2026, 7, 22, 14, 51)
    header = cli._session_header(started, ended)
    assert "**Started:** 2026-07-22 14:03" in header
    assert "**Ended:** 2026-07-22 14:51" in header
    assert "**Duration:** 48m" in header
    assert header.endswith("\n\n")


# ---- _flatten_session ---------------------------------------------------

def test_flatten_default_removes_folder_and_audio(tmp_path):
    out = tmp_path / "2026-07-20-meeting"
    out.mkdir(parents=True)
    (out / RECORDING_NAME).write_bytes(b"audio")
    transcript = out / "transcript.md"
    transcript.write_text("hello")

    final = cli._flatten_session(out, transcript, keep_audio=False)

    assert final == tmp_path / "2026-07-20-meeting.md"
    assert final.read_text() == "hello"
    assert not out.exists()  # recording deleted with the folder


def test_flatten_keep_audio_retains_recording(tmp_path):
    out = tmp_path / "2026-07-20-meeting"
    out.mkdir(parents=True)
    (out / RECORDING_NAME).write_bytes(b"audio")
    transcript = out / "transcript.md"
    transcript.write_text("hello")

    final = cli._flatten_session(out, transcript, keep_audio=True)

    assert final.read_text() == "hello"
    assert (out / RECORDING_NAME).read_bytes() == b"audio"
    assert not transcript.exists()  # moved out


def test_flatten_collision_gets_suffix(tmp_path):
    (tmp_path / "2026-07-20-meeting.md").write_text("old")
    out = tmp_path / "2026-07-20-meeting"
    out.mkdir()
    transcript = out / "transcript.md"
    transcript.write_text("new")

    final = cli._flatten_session(out, transcript, keep_audio=False)

    assert final == tmp_path / "2026-07-20-meeting-2.md"
