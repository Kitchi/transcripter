"""Audio device discovery."""

import sys
from pathlib import Path

import sounddevice as sd

MONITOR_SUFFIX = ".monitor"  # PulseAudio/PipeWire loopback of an output sink
BLACKHOLE_NAME = "BlackHole"  # still guarded against as a stray default input

# Sentinel returned by find_system_capture() on macOS: system audio is captured
# via the bundled Core Audio process-tap helper, not a sounddevice input index.
SYSTEM_TAP = "coreaudio-tap"

# Bundled, code-signed universal binary built from helper/SystemAudioTap.swift.
TAP_HELPER = Path(__file__).parent / "_bin" / "system-audio-tap"


class DeviceError(RuntimeError):
    pass


def find_system_capture() -> int | str:
    """Return the system-audio capture source.

    macOS: the SYSTEM_TAP sentinel (a non-invasive Core Audio tap on the current
    output device -- no BlackHole, output and volume keys unaffected). Linux: the
    sounddevice index of a PulseAudio/PipeWire monitor source.
    """
    if sys.platform == "darwin":
        if not TAP_HELPER.exists():
            raise DeviceError(
                f"System-audio tap helper missing: {TAP_HELPER}. "
                "Build it with helper/build.sh."
            )
        return SYSTEM_TAP
    return _find_monitor()


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


def describe(source: int | str) -> str:
    if source == SYSTEM_TAP:
        return "Core Audio system tap"
    return sd.query_devices(source)["name"]
