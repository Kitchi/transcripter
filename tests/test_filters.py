from transcripter.filters import drop_hallucinations, is_hallucination


def seg(text, no_speech_prob=0.0, compression_ratio=0.0):
    return {
        "start": 0.0,
        "end": 1.0,
        "text": text,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
    }


DEFAULTS = {"no_speech_threshold": 0.6, "compression_ratio_threshold": 2.4}


def test_drops_silence_filler_by_no_speech_prob():
    assert is_hallucination(seg("Thank you.", no_speech_prob=0.95), **DEFAULTS)


def test_drops_repetition_loop_by_compression_ratio():
    loop = "I remember I remember I remember I remember"
    assert is_hallucination(seg(loop, compression_ratio=3.1), **DEFAULTS)


def test_drops_blocklisted_filler_even_in_speech_window():
    # low no_speech_prob (shares window with real speech) but exact junk text
    assert is_hallucination(seg("Thank you.", no_speech_prob=0.1), **DEFAULTS)


def test_drops_cjk_junk():
    assert is_hallucination(seg("底"), **DEFAULTS)


def test_keeps_real_backchannel():
    for text in ["Okay.", "Yeah.", "Yep.", "Sure.", "Thanks."]:
        assert not is_hallucination(seg(text), **DEFAULTS), text


def test_keeps_real_speech_containing_thanks():
    # "thank you" only matches as the whole segment, not as a substring
    assert not is_hallucination(seg("thank you so much for that update"), **DEFAULTS)


def test_keeps_speech_in_silent_looking_window_below_threshold():
    assert not is_hallucination(seg("let's move on", no_speech_prob=0.5), **DEFAULTS)


def test_drop_hallucinations_filters_list():
    segs = [
        seg("Okay."),
        seg("Thank you.", no_speech_prob=0.9),
        seg("real content here"),
        seg("底"),
    ]
    kept = [s["text"] for s in drop_hallucinations(segs)]
    assert kept == ["Okay.", "real content here"]
