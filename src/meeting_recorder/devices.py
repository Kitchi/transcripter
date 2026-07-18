"""Audio device discovery."""

import sounddevice as sd

BLACKHOLE_NAME = "BlackHole"


class DeviceError(RuntimeError):
    pass


def find_blackhole() -> int:
    """Return the sounddevice index of the BlackHole input device."""
    for i, dev in enumerate(sd.query_devices()):
        if BLACKHOLE_NAME in dev["name"] and dev["max_input_channels"] > 0:
            return i
    raise DeviceError(
        "BlackHole input device not found. Install it (brew install blackhole-2ch) "
        "and route system output through a Multi-Output Device that includes it."
    )


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
