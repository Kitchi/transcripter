from transcripter.transcript import TranscriptBuilder


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_interleaves_channels_by_time():
    b = TranscriptBuilder(overlap_seconds=2.0)
    b.add_chunk("mic", 0, 0.0, [seg(1, 3, "hello"), seg(10, 12, "bye")])
    b.add_chunk("system", 0, 0.0, [seg(4, 6, "hi there")])
    out = b.render()
    assert out.index("**me**: hello") < out.index("**them**: hi there") < out.index("**me**: bye")


def test_drops_overlap_region_segments_on_later_chunks():
    b = TranscriptBuilder(overlap_seconds=2.0)
    # chunk 1 starts at 28s; a segment with midpoint < 2s (chunk-local) is overlap
    b.add_chunk("mic", 1, 28.0, [seg(0.5, 1.5, "duplicate"), seg(3, 5, "fresh")])
    texts = [s.text for s in b.segments]
    assert texts == ["fresh"]
    assert b.segments[0].start == 31.0  # offset by chunk start


def test_chunk_zero_keeps_leading_segments():
    b = TranscriptBuilder(overlap_seconds=2.0)
    b.add_chunk("mic", 0, 0.0, [seg(0.0, 1.0, "first words")])
    assert [s.text for s in b.segments] == ["first words"]


def test_skips_empty_text():
    b = TranscriptBuilder(overlap_seconds=2.0)
    b.add_chunk("mic", 0, 0.0, [seg(0, 1, "   "), seg(2, 3, "ok")])
    assert [s.text for s in b.segments] == ["ok"]


def test_unlabeled_render_omits_speaker_tags():
    b = TranscriptBuilder(overlap_seconds=2.0, label_speakers=False)
    b.add_chunk("mic", 0, 0.0, [seg(1, 3, "a dictated note")])
    out = b.render()
    assert "**me**" not in out
    assert "**them**" not in out
    assert "`[0:01]` a dictated note" in out


def test_render_timestamps():
    b = TranscriptBuilder(overlap_seconds=2.0)
    b.add_chunk("mic", 0, 0.0, [seg(75, 78, "minute mark")])
    assert "`[1:15]`" in b.render()


def test_drops_mic_bleed_of_system_audio():
    # System audio leaks into the mic: same words, overlapping time. The mic
    # copy is the echo and should be dropped.
    b = TranscriptBuilder(overlap_seconds=2.0)
    b.add_chunk("system", 0, 0.0, [seg(10, 13, "please hold the line")])
    b.add_chunk("mic", 0, 0.0, [seg(10, 13, "Please hold the line.")])
    out = b.render()
    assert "**them**: please hold the line" in out
    assert "**me**" not in out


def test_keeps_mic_when_text_differs():
    # Genuine talk-over: overlapping in time but different words -> keep both.
    b = TranscriptBuilder(overlap_seconds=2.0)
    b.add_chunk("system", 0, 0.0, [seg(10, 13, "please hold the line")])
    b.add_chunk("mic", 0, 0.0, [seg(10, 13, "sorry, can you repeat that")])
    out = b.render()
    assert "**them**: please hold the line" in out
    assert "**me**: sorry, can you repeat that" in out


def test_keeps_mic_when_not_overlapping_in_time():
    # Same words but disjoint in time: not an echo, keep both.
    b = TranscriptBuilder(overlap_seconds=2.0)
    b.add_chunk("system", 0, 0.0, [seg(10, 13, "thanks everyone")])
    b.add_chunk("mic", 0, 0.0, [seg(40, 43, "thanks everyone")])
    texts = [s.text for s in b._kept_segments()]
    assert texts.count("thanks everyone") == 2


def test_never_drops_system_segments():
    # Direction is fixed: a system segment is never treated as a bleed of the mic.
    b = TranscriptBuilder(overlap_seconds=2.0)
    b.add_chunk("mic", 0, 0.0, [seg(10, 13, "good morning")])
    b.add_chunk("system", 0, 0.0, [seg(10, 13, "good morning")])
    kept = {(s.channel, s.text) for s in b._kept_segments()}
    assert ("system", "good morning") in kept
    assert ("mic", "good morning") not in kept
