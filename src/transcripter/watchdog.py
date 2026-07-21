"""Silence watchdog: pure logic, fed RMS observations, decides when to stop.

States: CALIBRATING (measure ambient mic level) -> WAITING (armed only after
first speech/system audio) -> ARMED (stop after sustained silence).
"""

from enum import Enum, auto


class State(Enum):
    CALIBRATING = auto()
    WAITING_FOR_SPEECH = auto()
    ARMED = auto()


class SilenceWatchdog:
    def __init__(
        self,
        silence_stop_seconds: float,
        calibration_seconds: float,
        speech_rms_factor: float,
        mic_rms_floor: float,
        system_rms_threshold: float,
    ):
        self.silence_stop_seconds = silence_stop_seconds
        self.calibration_seconds = calibration_seconds
        self.speech_rms_factor = speech_rms_factor
        self.mic_rms_floor = mic_rms_floor
        self.system_rms_threshold = system_rms_threshold

        self.state = State.CALIBRATING
        self.mic_speech_threshold: float | None = None
        self._calibration_samples: list[float] = []
        self._calibration_elapsed = 0.0
        self._silence_elapsed = 0.0

    @property
    def silence_elapsed(self) -> float:
        return self._silence_elapsed

    def update(
        self, mic_rms: float, system_rms: float | None, block_seconds: float
    ) -> bool:
        """Feed one block's RMS per channel. Returns True when recording should stop.

        ``system_rms`` is ``None`` in mic-only sessions (e.g. note mode), where
        activity is decided by the mic alone.
        """
        if self.state is State.CALIBRATING:
            self._calibration_samples.append(mic_rms)
            self._calibration_elapsed += block_seconds
            if self._calibration_elapsed >= self.calibration_seconds:
                ambient = sorted(self._calibration_samples)[len(self._calibration_samples) // 2]
                self.mic_speech_threshold = max(
                    ambient * self.speech_rms_factor, self.mic_rms_floor
                )
                self.state = State.WAITING_FOR_SPEECH
            return False

        active = mic_rms > self.mic_speech_threshold or (
            system_rms is not None and system_rms > self.system_rms_threshold
        )

        if self.state is State.WAITING_FOR_SPEECH:
            if active:
                self.state = State.ARMED
            return False

        # ARMED
        if active:
            self._silence_elapsed = 0.0
        else:
            self._silence_elapsed += block_seconds
        return self._silence_elapsed >= self.silence_stop_seconds
