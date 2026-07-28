import numpy as np

from transcripter.aec import cancel_echo, echo_coherence, erle_db

RATE = 16_000


def _speechlike(n, seed):
    """Band-limited noise bursts with pauses -- crude but speech-shaped."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    kernel = np.hanning(64).astype(np.float32)
    x = np.convolve(x, kernel / kernel.sum(), mode="same")
    # Gate into ~0.5 s bursts so there are echo-only and quiet stretches.
    env = np.zeros(n, dtype=np.float32)
    for start in range(0, n, RATE):
        env[start : start + RATE // 2] = 1.0
    return (x * env).astype(np.float32)


def _echo_of(far, delay, gain=0.3):
    """Delayed, filtered copy of far -- what the mic picks up off the speakers."""
    room = np.array([gain, 0.0, gain * 0.4, 0.0, 0.0, gain * 0.2], dtype=np.float32)
    echo = np.convolve(far, room, mode="full")[: len(far)]
    return np.concatenate([np.zeros(delay, dtype=np.float32), echo])[: len(far)]


def test_removes_pure_echo():
    far = _speechlike(RATE * 8, seed=0)
    near = _echo_of(far, delay=400, gain=0.15)
    cleaned = cancel_echo(near, far, taps=1024)
    assert erle_db(near, cleaned, far) > 15


def test_quieter_echo_is_cancelled_harder():
    """Lower speaker-to-mic coupling leaves the DTD more blocks to adapt on."""
    far = _speechlike(RATE * 8, seed=0)
    loud = _echo_of(far, delay=400, gain=0.15)
    quiet = _echo_of(far, delay=400, gain=0.08)
    erle_loud = erle_db(loud, cancel_echo(loud, far, taps=1024), far)
    erle_quiet = erle_db(quiet, cancel_echo(quiet, far, taps=1024), far)
    assert erle_quiet > erle_loud > 15


def test_coupling_above_dtd_ratio_disables_adaptation():
    """Documents a real operational hazard, not a desired behaviour.

    The double-talk detector only adapts when near power is below
    `dtd_ratio` x far power. If the speakers are loud enough that the echo
    alone exceeds that, almost no block qualifies, the filter never converges,
    and the canceller removes nothing measurable. The pipeline logs ERLE so
    this shows up as 0.0 dB rather than failing invisibly.
    """
    far = _speechlike(RATE * 8, seed=0)
    near = _echo_of(far, delay=400, gain=0.3)  # coupling power ratio ~0.23

    # ERLE's own frame selection uses dtd_ratio too, so hold the *measurement*
    # threshold fixed at one that can see these frames and vary only the filter's.
    starved = cancel_echo(near, far, taps=1024, dtd_ratio=0.05)
    recovered = cancel_echo(near, far, taps=1024, dtd_ratio=0.5)

    assert erle_db(near, starved, far, dtd_ratio=0.5) < 1
    assert erle_db(near, recovered, far, dtd_ratio=0.5) > 15
    # At the default measurement threshold the starved case reports a flat 0.0,
    # which is the signal to raise --aec-dtd-ratio.
    assert erle_db(near, starved, far) == 0.0


def test_preserves_near_speech_during_double_talk():
    """My voice must survive, and the echo under it must still be subtracted.

    The far end talks alone first so the filter can converge, then we both
    talk; through that the DTD holds the filter and keeps subtracting.
    """
    n = RATE * 16
    far = _speechlike(n, seed=1)
    mine = np.zeros(n, dtype=np.float32)
    mine[n // 2 :] = _speechlike(n, seed=2)[n // 2 :] * 0.8  # I join halfway
    near = _echo_of(far, delay=400, gain=0.15) + mine

    cleaned = cancel_echo(near, far, taps=1024)

    dt = slice(n // 2, n)  # the double-talk stretch
    # My voice survives: cleaned tracks what I said better than the raw mic did.
    assert np.corrcoef(cleaned[dt], mine[dt])[0, 1] > np.corrcoef(near[dt], mine[dt])[0, 1]
    assert np.corrcoef(cleaned[dt], mine[dt])[0, 1] > 0.95
    # And the residual is closer to my voice alone than the raw mic was.
    assert np.mean((cleaned[dt] - mine[dt]) ** 2) < np.mean((near[dt] - mine[dt]) ** 2)


def test_no_far_signal_leaves_mic_untouched():
    mine = _speechlike(RATE * 4, seed=3)
    cleaned = cancel_echo(mine, np.zeros_like(mine), taps=1024)
    np.testing.assert_allclose(cleaned, mine, atol=1e-6)


def test_handles_unequal_lengths():
    far = _speechlike(RATE * 3, seed=4)
    near = _echo_of(far, delay=200)[: RATE * 2]
    cleaned = cancel_echo(near, far, taps=512)
    assert len(cleaned) == RATE * 2


def test_empty_input():
    empty = np.zeros(0, dtype=np.float32)
    assert len(cancel_echo(empty, empty)) == 0
    assert erle_db(empty, empty, empty) == 0.0


def test_erle_zero_when_nothing_played():
    mine = _speechlike(RATE * 4, seed=5)
    silence = np.zeros_like(mine)
    assert erle_db(mine, mine, silence) == 0.0


def test_coherence_separates_a_real_echo_path_from_none():
    """The gate that decides whether running the filter is worthwhile at all."""
    far = _speechlike(RATE * 8, seed=7)
    echoed = _echo_of(far, delay=400, gain=0.15)
    unrelated = _speechlike(RATE * 8, seed=8)  # headphones: mic hears only me

    assert echo_coherence(echoed, far, RATE) > 0.5
    assert echo_coherence(unrelated, far, RATE) < 0.1


def test_coherence_is_zero_when_nothing_played():
    mine = _speechlike(RATE * 4, seed=9)
    assert echo_coherence(mine, np.zeros_like(mine), RATE) == 0.0
    assert echo_coherence(np.zeros(0, np.float32), np.zeros(0, np.float32), RATE) == 0.0


def test_never_amplifies_when_there_is_no_echo_path():
    """The divergence guard, on the case that motivated it.

    With no echo path the filter fits mic noise against a loud uncorrelated
    reference and used to random-walk until it buried the signal -- a real
    recording came out 25 dB louder than it went in. Cancellation can only ever
    remove energy, so the output must never exceed the input.
    """
    far = _speechlike(RATE * 30, seed=10)
    mine = _speechlike(RATE * 30, seed=11)  # uncorrelated with far

    cleaned = cancel_echo(mine, far, taps=1024)

    assert np.sqrt(np.mean(cleaned**2)) <= np.sqrt(np.mean(mine**2)) * 1.01
    assert np.abs(cleaned).max() <= np.abs(mine).max() * 1.01


def test_erle_reports_zero_for_a_noop_canceller():
    far = _speechlike(RATE * 8, seed=6)
    near = _echo_of(far, delay=400)
    assert erle_db(near, near, far) == 0.0  # cleaned == near -> no enhancement
