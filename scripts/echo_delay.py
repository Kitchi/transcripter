#!/usr/bin/env python3
"""Measure the echo path between the mic and system channels of a recording.

Diagnostic only -- changes nothing. Reports what `aec.py` bases its decisions
on, plus the detail `aec.py` does not need but a human does when the numbers
look wrong.

  1. BULK DELAY. How far does the mic trail the system channel? The two streams
     are recorded on independent clocks and interleaved by arrival order with no
     timestamp alignment, so this is specific to each recording -- 127 ms and
     110 ms on the two meetings this was built against.

  2. DRIFT. Does that delay move? Clock offset walks it a few ms over a meeting,
     and a buffer resync can step it tens of ms at once. Reported per segment,
     with a drift rate fitted only within runs, so one step does not masquerade
     as a huge drift.

  3. COUPLING AND TAIL. How loud is the echo, and how long does it ring? The
     tail sets how much `reach` the canceller needs; the coupling sets whether
     the double-talk detector will ever fire.

  4. --test-aec runs the real canceller, aligned and not, so the measurement
     above can be checked against what it actually buys.

    python scripts/echo_delay.py "/path/to/recording.wav"
    python scripts/echo_delay.py "<...>/recording.wav" --test-aec
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from transcripter.aec import PHAT_FLOOR, SPEECH_BAND  # noqa: E402


def read_stereo(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (left=mic, right=system, rate) as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 2 and w.getsampwidth() == 2, "expected 16-bit stereo"
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = raw.astype(np.float32) / 32768.0
    return x[0::2].copy(), x[1::2].copy(), rate


def active_windows(far: np.ndarray, win: int) -> np.ndarray:
    """Indices of windows where the far end is actually playing.

    Silent windows carry no information about the echo path and would only
    dilute the average, so every measurement here is gated on far activity.
    """
    nw = len(far) // win
    if nw == 0:
        return np.zeros(0, dtype=int)
    power = (far[: nw * win].reshape(nw, win).astype(np.float64) ** 2).mean(axis=1)
    return np.flatnonzero(power > max(float(np.median(power)), 1e-12))


def gcc_phat(
    near: np.ndarray,
    far: np.ndarray,
    rate: int,
    win: int,
    idx: np.ndarray,
    max_lag: int,
    band: tuple[float, float] | None = SPEECH_BAND,
    floor: float = PHAT_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Averaged, band-limited, floored GCC-PHAT. Returns (lags, correlation).

    Mirrors `aec._gcc_phat` but returns the whole correlation curve rather than
    just its peak, so the caller can show the shape. `band=None` reproduces the
    unrestricted version, which is useful only for demonstrating why it fails:
    both channels carry the same decimation-filter fingerprint at zero lag, and
    full-band correlation locks onto that instead of the echo.
    """
    size = 2 * win
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    keep = np.ones(len(freqs), bool) if band is None else (freqs >= band[0]) & (freqs < band[1])
    acc = np.zeros(len(freqs), dtype=np.complex128)
    for i in idx:
        s = slice(i * win, (i + 1) * win)
        R = np.fft.rfft(near[s], size) * np.conj(np.fft.rfft(far[s], size))
        mag = np.abs(R)
        R /= mag + floor * float(mag[keep].mean()) + 1e-12
        R[~keep] = 0.0
        acc += R
    r = np.fft.irfft(acc / max(len(idx), 1), size)
    r = np.roll(r, max_lag)[: 2 * max_lag + 1]
    return np.arange(-max_lag, max_lag + 1), r


def cross_spectra(
    near: np.ndarray, far: np.ndarray, frame: int, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Averaged (Sxy, Sxx, Syy) over far-active frames, Hann windowed."""
    w = np.hanning(frame)
    Sxy = np.zeros(frame // 2 + 1, dtype=np.complex128)
    Sxx = np.zeros(frame // 2 + 1)
    Syy = np.zeros(frame // 2 + 1)
    for i in idx:
        s = slice(i * frame, (i + 1) * frame)
        D = np.fft.rfft(near[s] * w)
        X = np.fft.rfft(far[s] * w)
        Sxy += D * np.conj(X)
        Sxx += np.abs(X) ** 2
        Syy += np.abs(D) ** 2
    k = max(len(idx), 1)
    return Sxy / k, Sxx / k, Syy / k


def bar(value: float, peak: float, width: int = 40) -> str:
    return "#" * int(round(width * max(value, 0.0) / (peak + 1e-30)))


def sharpness(r: np.ndarray, peak: int) -> float:
    return float(r[peak]) / (float(np.median(np.abs(r))) + 1e-30)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("recording", type=Path, help="stereo recording.wav (mic=L, system=R)")
    ap.add_argument("--max-lag-ms", type=float, default=500.0, help="delay search range, +/- ms")
    ap.add_argument("--win-s", type=float, default=2.0, help="correlation window, seconds")
    ap.add_argument("--segment-s", type=float, default=60.0, help="drift report interval")
    ap.add_argument("--test-aec", action="store_true", help="run the canceller, aligned vs not")
    args = ap.parse_args()

    mic, system, rate = read_stereo(args.recording)
    n = min(len(mic), len(system))
    mic, system = mic[:n], system[:n]
    print(f"{args.recording}")
    print(f"  {n / rate:.0f}s @ {rate} Hz")
    print(
        f"  mic rms {np.sqrt(np.mean(mic**2)):.5f}   "
        f"system rms {np.sqrt(np.mean(system**2)):.5f}"
    )

    win = int(args.win_s * rate)
    max_lag = int(args.max_lag_ms * rate / 1000)
    idx = active_windows(system, win)
    print(f"  {len(idx)}/{n // win} windows have the far end active")
    if len(idx) == 0:
        print("  nothing played on the system channel; no echo path to measure")
        return

    # ---- 1. bulk delay ------------------------------------------------------
    lags, r = gcc_phat(mic, system, rate, win, idx, max_lag)
    best = int(np.argmax(r))
    bulk = int(lags[best])
    print("\n--- 1. bulk delay (mic relative to system) ---")
    print(f"  peak lag  {1000 * bulk / rate:+.2f} ms      sharpness {sharpness(r, best):.1f}")
    # The same measurement without the band limit, to show what it protects
    # against: on a real recording this reported 0.1 ms for a true 127 ms delay.
    _, rw = gcc_phat(mic, system, rate, win, idx, max_lag, band=None)
    bw = int(np.argmax(rw))
    print(
        f"  (unrestricted band would say {1000 * int(lags[bw]) / rate:+.2f} ms, "
        f"sharpness {sharpness(rw, bw):.1f})"
    )
    seen: list[int] = []
    for j in np.argsort(r)[::-1]:
        lag = int(lags[j])
        if any(abs(lag - s) < rate // 1000 for s in seen):  # merge within 1 ms
            continue
        seen.append(lag)
        print(f"    {1000 * lag / rate:+8.2f} ms  {bar(float(r[j]), float(r[best]))}")
        if len(seen) == 6:
            break
    if bulk < 0:
        print("  NEGATIVE: the echo precedes its reference. The filter is causal-only")
        print("            and cannot model this; check the recorder's channel order.")

    # ---- 2. drift -----------------------------------------------------------
    print(f"\n--- 2. drift (per {args.segment_s:.0f}s) ---")
    per = max(int(args.segment_s / args.win_s), 2)
    times, delays = [], []
    for start in range(0, len(idx), per):
        chunk = idx[start : start + per]
        if len(chunk) < 2:
            continue
        _, rr = gcc_phat(mic, system, rate, win, chunk, max_lag)
        j = int(np.argmax(rr))
        if sharpness(rr, j) < 15.0:  # same gate the canceller uses
            continue
        times.append(float(chunk[0] * win / rate))
        delays.append(int(lags[j]))
    for t, d in zip(times, delays, strict=True):
        print(f"  t={t:7.0f}s  {1000 * d / rate:+8.2f} ms")
    if len(times) >= 3:
        # Fit drift only inside runs of continuous delay. A buffer resync steps
        # the delay tens of ms at once, and a single line through a step reports
        # a drift rate that is an artifact of the step, not a clock offset.
        d = np.diff(delays)
        breaks = [0, *(np.flatnonzero(np.abs(d) > 0.005 * rate) + 1).tolist(), len(delays)]
        slopes = []
        for a, b in zip(breaks, breaks[1:], strict=False):
            if b - a >= 3:
                slopes.append(float(np.polyfit(times[a:b], delays[a:b], 1)[0]))
        if len(breaks) > 2:
            print(f"  {len(breaks) - 2} step change(s) -- a buffer resync, not drift")
        if slopes:
            ppm = 1e6 * float(np.mean(slopes)) / rate
            print(f"  drift within runs {ppm:+.1f} ppm clock offset")
        print(
            f"  range {1000 * min(delays) / rate:.0f}-{1000 * max(delays) / rate:.0f} ms; "
            f"the canceller aligns to the minimum"
        )

    # ---- 3. coupling and tail -----------------------------------------------
    # Align first: a delay wider than the analysis frame smears the cross
    # spectrum and understates everything below.
    lo = max(min(delays, default=bulk), 0)
    a_mic, a_sys = mic[lo:], system[: n - lo]
    frame = 4096
    aidx = active_windows(a_sys, frame)
    Sxy, Sxx, Syy = cross_spectra(a_mic, a_sys, frame, aidx)
    pred = np.abs(Sxy) ** 2 / (Sxx + 1e-20)  # far-predictable mic power per bin
    coupling = float(pred.sum() / (Sxx.sum() + 1e-30))
    print(f"\n--- 3. coupling and tail (aligned to {1000 * lo / rate:.0f} ms) ---")
    print(f"  echo / far power  {coupling:.4f}  ({10 * np.log10(coupling + 1e-30):+.1f} dB)")
    print("  the filter adapts only where near/far power < dtd_ratio (default 0.05)")
    if coupling > 0.05:
        print("  ABOVE 0.05: the detector will rarely fire; raise --aec-dtd-ratio")

    h = np.fft.irfft(Sxy / (Sxx + 0.01 * float(Sxx.mean()) + 1e-20))
    env = np.abs(h[: frame // 2])  # causal half; the delay is already removed
    pk = float(env.max())
    print("  echo impulse response (each row 4 ms):")
    step = int(0.004 * rate)
    for i in range(0, min(int(0.160 * rate), len(env) - step), step):
        v = float(env[i : i + step].max())
        if v > 0.05 * pk:
            print(f"    {1000 * i / rate:6.0f} ms  {bar(v, pk)}")
    tail = np.flatnonzero(env > 0.1 * pk)
    if len(tail):
        print(
            f"  rings out to {1000 * int(tail[-1]) / rate:.0f} ms (-20 dB); "
            "set aec_reach above this"
        )

    # ---- 4. what the canceller actually does --------------------------------
    if args.test_aec:
        from transcripter.aec import cancel_echo, erle_db, measure_echo_delay

        print("\n--- 4. cancel_echo ---")
        d = measure_echo_delay(mic, system, rate)
        print(
            f"  measure_echo_delay: {1000 * d.samples / rate:.0f} ms, "
            f"sharpness {d.sharpness:.1f}"
        )
        for label, m, s in (
            ("unaligned", mic, system),
            (
                f"aligned @{1000 * d.samples / rate:.0f}ms",
                mic[d.samples : n],
                system[: n - d.samples],
            ),
        ):
            print(f"  {label:20s} ERLE {erle_db(m, cancel_echo(m, s), s):+6.2f} dB")


if __name__ == "__main__":
    main()
