from transcripter.diarize import Turn, assign_speakers, speaker_label


def _seg(start, end, text="hi"):
    return {"start": start, "end": end, "text": text}


def test_speaker_label_is_one_based():
    assert speaker_label(0) == "Speaker 1"
    assert speaker_label(3) == "Speaker 4"


def test_assigns_by_maximum_overlap():
    turns = [Turn(0, 10, 0), Turn(10, 20, 1)]
    segments = [_seg(1, 3), _seg(12, 18)]
    got = assign_speakers(segments, turns)
    assert [s["speaker"] for s in got] == ["Speaker 1", "Speaker 2"]


def test_straddling_segment_takes_the_larger_share():
    """A segment spanning a speaker change gets one label -- the dominant one."""
    turns = [Turn(0, 10, 0), Turn(10, 20, 1)]
    got = assign_speakers([_seg(9, 16)], turns)  # 1s of spk0, 6s of spk1
    assert got[0]["speaker"] == "Speaker 2"


def test_segment_outside_every_turn_is_unlabelled():
    turns = [Turn(0, 5, 0)]
    got = assign_speakers([_seg(30, 35)], turns)
    assert got[0]["speaker"] is None


def test_touching_but_not_overlapping_is_not_a_match():
    """Zero-length overlap must not count; boundaries are half-open."""
    turns = [Turn(0, 10, 0)]
    got = assign_speakers([_seg(10, 12)], turns)
    assert got[0]["speaker"] is None


def test_no_turns_leaves_segments_untouched():
    segments = [_seg(1, 2)]
    assert assign_speakers(segments, []) == segments


def test_original_fields_are_preserved():
    turns = [Turn(0, 10, 0)]
    got = assign_speakers([_seg(1, 3, text="hello there")], turns)
    assert got[0]["text"] == "hello there"
    assert got[0]["start"] == 1


def test_does_not_mutate_input():
    turns = [Turn(0, 10, 0)]
    segments = [_seg(1, 3)]
    assign_speakers(segments, turns)
    assert "speaker" not in segments[0]
