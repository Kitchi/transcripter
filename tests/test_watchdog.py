from transcripter.watchdog import SilenceWatchdog, State


def make(**overrides):
    kwargs = dict(
        silence_stop_seconds=45.0,
        calibration_seconds=3.0,
        speech_rms_factor=4.0,
        mic_rms_floor=1e-3,
        system_rms_threshold=5e-4,
    )
    kwargs.update(overrides)
    return SilenceWatchdog(**kwargs)


def calibrate(w, ambient=0.01, block=0.25):
    while w.state is State.CALIBRATING:
        assert not w.update(ambient, 0.0, block)


def test_calibration_sets_threshold_from_ambient():
    w = make()
    calibrate(w, ambient=0.01)
    assert w.mic_speech_threshold == 0.04
    assert w.state is State.WAITING_FOR_SPEECH


def test_threshold_floor_in_silent_room():
    w = make()
    calibrate(w, ambient=1e-5)
    assert w.mic_speech_threshold == 1e-3


def test_does_not_fire_before_any_speech():
    w = make()
    calibrate(w)
    for _ in range(1000):  # 250s of silence, never armed
        assert not w.update(0.01, 0.0, 0.25)
    assert w.state is State.WAITING_FOR_SPEECH


def test_arms_on_mic_speech_then_fires_after_sustained_silence():
    w = make()
    calibrate(w, ambient=0.01)
    assert not w.update(0.2, 0.0, 0.25)  # speech
    assert w.state is State.ARMED
    fired = 0.0
    while not w.update(0.01, 0.0, 0.25):
        fired += 0.25
        assert fired < 60
    assert fired >= 45 - 0.25


def test_arms_on_system_audio_alone():
    w = make()
    calibrate(w)
    assert not w.update(0.01, 0.05, 0.25)  # remote audio, mic quiet
    assert w.state is State.ARMED


def test_activity_resets_silence_timer():
    w = make()
    calibrate(w, ambient=0.01)
    w.update(0.2, 0.0, 0.25)
    for _ in range(100):  # 25s quiet
        assert not w.update(0.01, 0.0, 0.25)
    w.update(0.0, 0.01, 0.25)  # system audio blip resets the clock
    for _ in range(179):  # another ~44.75s quiet: still under threshold
        assert not w.update(0.01, 0.0, 0.25)
    assert w.update(0.01, 0.0, 0.25)


def test_zoom_muted_quiet_room_still_fires():
    # Mic carries ambient noise (Zoom mute is software-only); watchdog must
    # still fire on ambient-level input.
    w = make()
    calibrate(w, ambient=0.008)
    w.update(0.3, 0.0, 0.25)
    fired = False
    for _ in range(200):
        if w.update(0.012, 0.0, 0.25):  # ambient-ish, below 4x threshold
            fired = True
            break
    assert fired
