from transcripter.transcript import Transcript


def seg(start, end, text, speaker=None):
    s = {"start": start, "end": end, "text": text}
    if speaker is not None:
        s["speaker"] = speaker
    return s


def test_interleaves_channels_by_time():
    t = Transcript()
    t.add("mic", [seg(1, 3, "hello"), seg(10, 12, "bye")])
    t.add("system", [seg(4, 6, "hi there")])
    out = t.render()
    assert out.index("**me**: hello") < out.index("**them**: hi there") < out.index("**me**: bye")


def test_skips_empty_text():
    t = Transcript()
    t.add("mic", [seg(0, 1, "   "), seg(2, 3, "ok")])
    assert [s.text for s in t.segments] == ["ok"]


def test_unlabeled_render_omits_speaker_tags():
    t = Transcript(label_speakers=False)
    t.add("mic", [seg(1, 3, "a dictated note")])
    out = t.render()
    assert "**me**" not in out
    assert "**them**" not in out
    assert "`[0:01]` a dictated note" in out


def test_render_timestamps():
    t = Transcript()
    t.add("mic", [seg(75, 78, "minute mark")])
    assert "`[1:15]`" in t.render()


def test_render_timestamps_past_an_hour():
    t = Transcript()
    t.add("mic", [seg(3725, 3728, "long meeting")])
    assert "`[1:02:05]`" in t.render()


# ---- diarized speaker labels --------------------------------------------

def test_diarized_speaker_label_replaces_them():
    t = Transcript()
    t.add("system", [seg(1, 3, "first point", speaker="Speaker 1")])
    t.add("system", [seg(4, 6, "second point", speaker="Speaker 2")])
    out = t.render()
    assert "**Speaker 1**: first point" in out
    assert "**Speaker 2**: second point" in out
    assert "**them**" not in out


def test_undiarized_system_segment_falls_back_to_them():
    t = Transcript()
    t.add("system", [seg(1, 3, "unlabelled")])
    assert "**them**: unlabelled" in t.render()


def test_mic_is_always_me_even_if_a_speaker_leaks_in():
    """The mic is one voice by construction; a stray label must not override it."""
    t = Transcript()
    t.add("mic", [seg(1, 3, "mine", speaker="Speaker 3")])
    assert "**me**: mine" in t.render()


# ---- cross-channel bleed dedup -------------------------------------------

def test_drops_mic_bleed_of_system_audio():
    # System audio leaks into the mic: same words, overlapping time. The mic
    # copy is the echo and should be dropped.
    t = Transcript()
    t.add("system", [seg(10, 13, "please hold the line")])
    t.add("mic", [seg(10, 13, "Please hold the line.")])
    out = t.render()
    assert "**them**: please hold the line" in out
    assert "**me**" not in out


def test_keeps_mic_when_text_differs():
    # Genuine talk-over: overlapping in time but different words -> keep both.
    t = Transcript()
    t.add("system", [seg(10, 13, "please hold the line")])
    t.add("mic", [seg(10, 13, "sorry, can you repeat that")])
    out = t.render()
    assert "**them**: please hold the line" in out
    assert "**me**: sorry, can you repeat that" in out


def test_keeps_mic_when_not_overlapping_in_time():
    # Same words but disjoint in time: not an echo, keep both.
    t = Transcript()
    t.add("system", [seg(10, 13, "thanks everyone")])
    t.add("mic", [seg(40, 43, "thanks everyone")])
    texts = [s.text for s in t._kept_segments()]
    assert texts.count("thanks everyone") == 2


def test_never_drops_system_segments():
    # Direction is fixed: a system segment is never treated as a bleed of the mic.
    t = Transcript()
    t.add("mic", [seg(10, 13, "good morning")])
    t.add("system", [seg(10, 13, "good morning")])
    kept = {(s.channel, s.text) for s in t._kept_segments()}
    assert ("system", "good morning") in kept
    assert ("mic", "good morning") not in kept


def test_bleed_dedup_still_works_against_a_diarized_segment():
    t = Transcript()
    t.add("system", [seg(10, 13, "please hold the line", speaker="Speaker 2")])
    t.add("mic", [seg(10, 13, "Please hold the line.")])
    out = t.render()
    assert "**Speaker 2**: please hold the line" in out
    assert "**me**" not in out
