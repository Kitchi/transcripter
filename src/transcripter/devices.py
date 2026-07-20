"""Audio device discovery."""

import sys

import sounddevice as sd

BLACKHOLE_NAME = "BlackHole"
MONITOR_SUFFIX = ".monitor"  # PulseAudio/PipeWire loopback of an output sink


class DeviceError(RuntimeError):
    pass


def find_system_capture() -> int:
    """Return the sounddevice index of the system-audio capture device."""
    if sys.platform == "darwin":
        return _find_blackhole()
    return _find_monitor()


def _find_blackhole() -> int:
    for i, dev in enumerate(sd.query_devices()):
        if BLACKHOLE_NAME in dev["name"] and dev["max_input_channels"] > 0:
            return i
    raise DeviceError(
        "BlackHole input device not found. Install it (brew install blackhole-2ch) "
        "and route system output through a Multi-Output Device that includes it."
    )


def _find_monitor() -> int:
    """Find a PulseAudio/PipeWire monitor source (Linux: no manual setup needed)."""
    candidates = [
        i
        for i, dev in enumerate(sd.query_devices())
        if MONITOR_SUFFIX in dev["name"].lower() and dev["max_input_channels"] > 0
    ]
    if not candidates:
        raise DeviceError(
            "No PulseAudio/PipeWire monitor source found. Ensure sounddevice is "
            "using the pulse/pipewire backend (not raw ALSA)."
        )
    return candidates[0]


def default_mic() -> int:
    """Return the system default input device, ensuring it isn't BlackHole itself."""
    idx = sd.default.device[0]
    if idx is None or idx < 0:
        raise DeviceError("No default input device configured.")
    if BLACKHOLE_NAME in sd.query_devices(idx)["name"]:
        raise DeviceError(
            "Default input device is BlackHole; select a real microphone as the "
            "system default input."
        )
    return idx


def describe(idx: int) -> str:
    return sd.query_devices(idx)["name"]
