import numpy as np

from transcripter.aec import cancel_echo, erle_db, measure_echo_delay

RATE = 16_000


def _speechlike(n, seed, smoothing=8):
    """Band-limited noise bursts with pauses -- crude but speech-shaped.

    `smoothing` sets the lowpass width. The default puts ~80% of the energy in
    the 300-3400 Hz band the delay measurement looks at, as real speech does.
    Widening it starves that band; see the narrowband test below.

    Burst and gap lengths are irregular and seeded, so two speakers share
    neither onsets nor a rhythm. Both matter for the delay measurement: two
    talkers switching on at identical samples correlate at zero lag, and two
    switching on at a *common period* correlate at the phase offset between
    them. Either manufactures an echo path that is not there. Real speech has
    no such periodicity, but a fixture built on a fixed on/off cycle does.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    kernel = np.hanning(smoothing).astype(np.float32)
    x = np.convolve(x, kernel / kernel.sum(), mode="same")
    # Gate into bursts so there are echo-only and quiet stretches to work with.
    env = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        on = int(rng.integers(RATE // 4, RATE))
        env[pos : pos + on] = 1.0
        pos += on + int(rng.integers(RATE // 4, RATE))
    return (x * env).astype(np.float32)


def _echo_of(far, delay, gain=0.3):
    """Delayed, filtered copy of far -- what the mic picks up off the speakers."""
    room = np.array([gain, 0.0, gain * 0.4, 0.0, 0.0, gain * 0.2], dtype=np.float32)
    echo = np.convolve(far, room, mode="full")[: len(far)]
    return np.concatenate([np.zeros(delay, dtype=np.float32), echo])[: len(far)]


def test_removes_pure_echo():
    far = _speechlike(RATE * 8, seed=0)
    near = _echo_of(far, delay=400, gain=0.15)
    cleaned = cancel_echo(near, far, reach=1024)
    assert erle_db(near, cleaned, far) > 15


def test_quieter_echo_is_cancelled_harder():
    """Lower speaker-to-mic coupling leaves the DTD more blocks to adapt on."""
    far = _speechlike(RATE * 8, seed=0)
    loud = _echo_of(far, delay=400, gain=0.15)
    quiet = _echo_of(far, delay=400, gain=0.08)
    erle_loud = erle_db(loud, cancel_echo(loud, far, reach=1024), far)
    erle_quiet = erle_db(quiet, cancel_echo(quiet, far, reach=1024), far)
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
    starved = cancel_echo(near, far, reach=1024, dtd_ratio=0.05)
    recovered = cancel_echo(near, far, reach=1024, dtd_ratio=0.5)

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

    cleaned = cancel_echo(near, far, reach=1024)

    dt = slice(n // 2, n)  # the double-talk stretch
    # My voice survives: cleaned tracks what I said better than the raw mic did.
    assert np.corrcoef(cleaned[dt], mine[dt])[0, 1] > np.corrcoef(near[dt], mine[dt])[0, 1]
    assert np.corrcoef(cleaned[dt], mine[dt])[0, 1] > 0.95
    # And the residual is closer to my voice alone than the raw mic was.
    assert np.mean((cleaned[dt] - mine[dt]) ** 2) < np.mean((near[dt] - mine[dt]) ** 2)


def test_no_far_signal_leaves_mic_untouched():
    mine = _speechlike(RATE * 4, seed=3)
    cleaned = cancel_echo(mine, np.zeros_like(mine), reach=1024)
    np.testing.assert_allclose(cleaned, mine, atol=1e-6)


def test_handles_unequal_lengths():
    far = _speechlike(RATE * 3, seed=4)
    near = _echo_of(far, delay=200)[: RATE * 2]
    cleaned = cancel_echo(near, far, reach=1024)
    assert len(cleaned) == RATE * 2


def test_empty_input():
    empty = np.zeros(0, dtype=np.float32)
    assert len(cancel_echo(empty, empty)) == 0
    assert erle_db(empty, empty, empty) == 0.0


def test_erle_zero_when_nothing_played():
    mine = _speechlike(RATE * 4, seed=5)
    silence = np.zeros_like(mine)
    assert erle_db(mine, mine, silence) == 0.0


def test_measures_a_known_delay():
    far = _speechlike(RATE * 30, seed=7)
    near = _echo_of(far, delay=2000, gain=0.15)  # 125 ms, as seen in the wild

    found = measure_echo_delay(near, far, RATE)

    assert abs(found.samples - 2000) < 16  # within 1 ms
    assert found.sharpness > 15.0


def test_sharpness_separates_a_real_echo_path_from_none():
    """The gate that decides whether running the filter is worthwhile at all.

    Thresholds here were set from negative controls on real recordings, where
    impossible pairings scored ~5-6 and true echo paths 27-102. Anything that
    cannot contain an echo must land far below `aec_sharpness_threshold`.
    """
    far = _speechlike(RATE * 30, seed=7)
    echoed = _echo_of(far, delay=2000, gain=0.15)
    unrelated = _speechlike(RATE * 30, seed=8)  # headphones: mic hears only me
    reversed_far = far[::-1].copy()  # same spectrum, no causal relationship

    assert measure_echo_delay(echoed, far, RATE).sharpness > 15.0
    assert measure_echo_delay(unrelated, far, RATE).sharpness < 15.0
    assert measure_echo_delay(echoed, reversed_far, RATE).sharpness < 15.0


def test_narrowband_audio_does_not_fake_an_echo_path():
    """The hazard the floored PHAT weighting exists for.

    Pure PHAT gives every frequency bin an equal vote, including bins holding
    nothing but the residue of whatever filter shaped the signal. When both
    channels are thin inside the analysis band, that shared residue is all
    there is to correlate, and two unrelated signals score like a real echo.
    This is not hypothetical: the same mechanism made a mic correlate with its
    own system channel played *backwards* at 30, and mis-read a real 127 ms
    delay as 0.1 ms, until the weighting was floored.
    """
    far = _speechlike(RATE * 30, seed=20, smoothing=64)  # ~5% of energy in band
    unrelated = _speechlike(RATE * 30, seed=21, smoothing=64)

    assert measure_echo_delay(unrelated, far, RATE).sharpness < 15.0


def test_delay_reports_the_minimum_across_a_shifting_path():
    """The delay moves mid-meeting when a buffer resyncs.

    Overshooting is far more costly than undershooting -- the filter cannot
    model an echo that precedes its reference -- so the minimum is the safe
    choice, not the mean.
    """
    far = _speechlike(RATE * 240, seed=12)
    early = _echo_of(far[: RATE * 120], delay=1600, gain=0.15)
    late = _echo_of(far[RATE * 120 :], delay=2600, gain=0.15)
    near = np.concatenate([early, late])

    found = measure_echo_delay(near, far, RATE)

    assert abs(found.samples - 1600) < 32


def test_delay_is_zero_when_nothing_played():
    mine = _speechlike(RATE * 4, seed=9)
    assert measure_echo_delay(mine, np.zeros_like(mine), RATE).samples == 0
    empty = np.zeros(0, np.float32)
    assert measure_echo_delay(empty, empty, RATE).sharpness == 0.0


def test_never_amplifies_when_there_is_no_echo_path():
    """The divergence guard, on the case that motivated it.

    With no echo path the filter fits mic noise against a loud uncorrelated
    reference and used to random-walk until it buried the signal -- a real
    recording came out 25 dB louder than it went in. Cancellation can only ever
    remove energy, so the output must never exceed the input.
    """
    far = _speechlike(RATE * 30, seed=10)
    mine = _speechlike(RATE * 30, seed=11)  # uncorrelated with far

    cleaned = cancel_echo(mine, far, reach=1024)

    assert np.sqrt(np.mean(cleaned**2)) <= np.sqrt(np.mean(mine**2)) * 1.01
    assert np.abs(cleaned).max() <= np.abs(mine).max() * 1.01


def test_erle_reports_zero_for_a_noop_canceller():
    far = _speechlike(RATE * 8, seed=6)
    near = _echo_of(far, delay=400)
    assert erle_db(near, near, far) == 0.0  # cleaned == near -> no enhancement
