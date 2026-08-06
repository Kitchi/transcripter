"""Acoustic echo cancellation: remove system-audio bleed from the mic track.

System audio leaves the speakers and re-enters the mic, so the far end's words
land on both channels. Subtracting the predicted echo before transcription beats
deleting duplicate text afterwards, because it also recovers **double-talk** --
the moments you and the far end speak at once, which text-level dedup throws
away wholesale.

Three stages, in order:

1. `measure_echo_delay` finds *when* the echo arrives, by cross-correlation.
2. The caller shifts the mic by that delay (see `pipeline._cancel`).
3. `cancel_echo` predicts the echo from the far signal and subtracts it.

Stage 1 is not optional. The mic and the system tap are separate CoreAudio
streams on independent clocks, and `recorder.py` interleaves them by arrival
order with no timestamp alignment, so the offset between the channels is
whatever the buffering happened to be. Measured on two real meetings it was
127 ms and 110 ms, drifting ~7 ppm and stepping 37 ms mid-session when a buffer
resynced. Unaligned, the same recordings cancel 0.3-2.8 dB; aligned, 3.5-6.7 dB.

Two properties of stage 3 drive the design.

**The filter can only look backwards.** It predicts the mic from *past* far
samples, so it cannot model an echo that arrives before its own reference.
Shifting the mic too far is therefore much worse than shifting it too little:
on a real recording, aligning 53 ms past the true delay dropped cancellation
from 6.7 dB to 0.7 dB. `measure_echo_delay` returns the *minimum* delay seen
across the session for exactly this reason.

**Filter length and block length are decoupled** (a partitioned block filter,
as in Speex/WebRTC). They pull in opposite directions, and the earlier
single-block version welded them to the same number, which made the filter
fragile: too short and it cannot cover the ~120 ms echo tail, so its predictions
are wrong, the divergence guard rejects them, and the filter is repeatedly
beaten back to zero; too long and each block spans so much time that almost none
qualifies as echo-only, so adaptation starves. On real recordings the usable
window was about one octave wide and sat in a different place per recording.
Splitting `reach` into `block`-sized partitions satisfies both at once: long
reach for the tail, short blocks for a responsive double-talk detector.

`transcript.py`'s bleed dedup stays on as a net for whatever survives; expect
this to reduce the bleed, not eliminate it.
"""

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Consecutive echo-only blocks that must come out louder before the filter is
# pulled back toward zero. A few is ordinary adaptation transient; a run means
# the filter is diverging. At the default 32 ms block this is ~0.5 s.
DIVERGENCE_RUN = 16

# Echo lives in the speech band, and so does the cross-correlation evidence for
# it. Above it sits a trap: both channels pass through the same decimation
# filter in `recorder.py`, which stamps an identical spectral fingerprint on
# each at *zero* lag. Full-band correlation locks onto that fingerprint instead
# of the echo -- it scored 30 on a control pairing a mic against its own system
# channel played backwards, where no echo can exist, and it mis-reported a real
# 127 ms delay as 0.1 ms. Band-limiting drops those controls to ~6.
SPEECH_BAND = (300.0, 3400.0)

# How far to back off pure PHAT weighting, as a fraction of a frame's mean
# cross-spectrum magnitude (see `_gcc_phat`). Pure PHAT gives an empty bin the
# same vote as a loud one, which lets shared filter artifacts decide the answer
# when either channel is thin inside the band.
PHAT_FLOOR = 0.2


@dataclass(frozen=True)
class EchoDelay:
    """Result of `measure_echo_delay`.

    `samples` is how far the mic trails the system channel, and is what the
    caller should shift the mic back by. `sharpness` is the confidence: how far
    the correlation peak stands above the background. Controls that cannot
    contain an echo score ~5-6; real echo paths score 30-100. Compare it against
    `Config.aec_sharpness_threshold` to decide whether to cancel at all.
    """

    samples: int
    sharpness: float


def _gcc_phat(
    near: np.ndarray,
    far: np.ndarray,
    rate: int,
    window: int,
    max_lag: int,
    band: tuple[float, float],
    floor: float = PHAT_FLOOR,
) -> tuple[int, float] | None:
    """Band-limited GCC-PHAT. Returns (lag in samples, sharpness), or None.

    PHAT normalizes each frame's cross-spectrum to unit magnitude before
    averaging, so the peak is decided by phase alignment alone. That keeps it
    sharp no matter how coloured speech is, and stops a few loud frames from
    dominating -- which plain cross-correlation does badly on speech.

    Taken literally that normalization is dangerous: a bin holding nothing but
    numerical dust is promoted to the same weight as one holding speech, so
    whatever the two channels share down there -- a common filter response,
    quantization -- drives the answer. `floor` divides by
    `|R| + floor x mean|R|` instead, which leaves loud bins alone and keeps
    empty ones empty. On the two real recordings it *raised* the true peak
    (38 -> 55) while leaving the negative controls at ~5.5.

    Averaged only over frames where the far end is actually playing; silent
    frames carry no information about the delay and would dilute the average.
    Returns None when there are too few such frames to average.
    """
    n = min(len(near), len(far))
    nf = n // window
    if nf < 2:
        return None
    far_f = far[: nf * window].reshape(nf, window)
    power = (far_f.astype(np.float64) ** 2).mean(axis=1)
    active = np.flatnonzero(power > max(float(np.median(power)), 1e-12))
    if len(active) < 2:
        return None

    size = 2 * window  # zero-padded, so the correlation is linear not circular
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    keep = (freqs >= band[0]) & (freqs < band[1])
    if not keep.any():
        return None

    acc = np.zeros(len(freqs), dtype=np.complex128)
    for i in active:
        s = slice(i * window, (i + 1) * window)
        R = np.fft.rfft(near[s], size) * np.conj(np.fft.rfft(far[s], size))
        mag = np.abs(R)
        R /= mag + floor * float(mag[keep].mean()) + 1e-12  # floored PHAT weight
        R[~keep] = 0.0
        acc += R

    r = np.fft.irfft(acc / len(active), size)
    # Roll so lag 0 sits at index `max_lag`: negative lags become visible
    # instead of wrapping around to the end of the buffer.
    r = np.roll(r, max_lag)[: 2 * max_lag + 1]
    peak = int(np.argmax(r))
    background = float(np.median(np.abs(r))) + 1e-30
    return peak - max_lag, float(r[peak]) / background


def measure_echo_delay(
    near: np.ndarray,
    far: np.ndarray,
    rate: int = 16_000,
    *,
    min_sharpness: float = 15.0,
    max_lag_s: float = 0.5,
    segment_s: float = 60.0,
    window_s: float = 2.0,
    band: tuple[float, float] = SPEECH_BAND,
) -> EchoDelay:
    """Find how far the mic trails the system channel, and how sure we are.

    Measured per `segment_s` rather than once over the whole recording, because
    the delay moves: clock drift walks it a few ms over a meeting, and a buffer
    resync can step it tens of ms at once. Segments scoring below
    `min_sharpness` are discarded as unreliable -- usually stretches where the
    far end barely spoke.

    Returns the **minimum** delay across the surviving segments, not the mean.
    Undershooting the delay costs a little cancellation; overshooting costs
    nearly all of it, because the filter cannot model an echo arriving before
    its reference. The reported `sharpness` is the median across all segments
    that produced a peak, so a caller that gates on it sees the typical
    confidence rather than the best or worst stretch.

    `samples` is 0 when nothing scored above `min_sharpness`; check `sharpness`
    against the same threshold rather than treating 0 as "no delay".
    """
    n = min(len(near), len(far))
    if n == 0:
        return EchoDelay(0, 0.0)
    window = int(window_s * rate)
    max_lag = int(max_lag_s * rate)
    if window < 2 or max_lag >= window:
        raise ValueError("window_s must be at least twice max_lag_s")

    step = max(int(segment_s * rate), window * 2)
    delays: list[int] = []
    scores: list[float] = []
    for start in range(0, n, step):
        stop = min(start + step, n)
        # The mic may trail by up to max_lag, so let it read past the segment
        # end; the far slice sets the alignment reference.
        found = _gcc_phat(
            near[start : min(stop + max_lag, len(near))],
            far[start:stop],
            rate,
            window,
            max_lag,
            band,
        )
        if found is None:
            continue
        lag, sharpness = found
        scores.append(sharpness)
        if sharpness >= min_sharpness and lag >= 0:
            delays.append(lag)

    if not scores:
        return EchoDelay(0, 0.0)
    typical = float(np.median(scores))
    if not delays:
        return EchoDelay(0, typical)
    log.debug(
        "echo delay: %d/%d segments usable, %.0f-%.0f ms",
        len(delays),
        len(scores),
        1000 * min(delays) / rate,
        1000 * max(delays) / rate,
    )
    return EchoDelay(min(delays), typical)


def cancel_echo(
    near: np.ndarray,
    far: np.ndarray,
    block: int = 512,
    reach: int = 2048,
    mu: float = 0.2,
    leak: float = 1e-3,
    reg_frac: float = 0.1,
    lam: float = 0.9,
    dtd_ratio: float = 0.05,
) -> np.ndarray:
    """Predict the echo of `far` present in `near` and subtract it.

    Returns the cleaned near signal, same length as the shorter input. Shift
    `near` by `measure_echo_delay` first; this filter models only what remains.

    A partitioned block frequency-domain NLMS. `reach` sets the echo-tail
    coverage (reach/rate seconds) and is split into ceil(reach/block)
    partitions, each convolved by overlap-save against a correspondingly delayed
    frame of `far`. `block` sets how often the filter updates and how finely the
    double-talk detector can decide -- the two are independent here, which is
    the point (see the module docstring).

    Per-bin power normalization whitens the colored far signal, but weak bins
    are floored to `reg_frac` of the *average* bin power -- flooring to a tiny
    constant instead lets low-energy bins of real speech starve and diverge.
    `leak` bleeds the filter toward zero each block so it cannot grow unbounded.

    Double-talk detector: the filter only *adapts* on blocks that look echo-only
    (near power below `dtd_ratio` x far power); when you are also talking it
    holds the current filter and keeps subtracting. Without this the filter
    mis-adapts on double-talk and injects far-end audio into the output.

    Divergence guard: subtracting an echo estimate can only ever *remove* energy
    from a block, so a block that comes out louder than it went in means the
    filter is injecting rather than cancelling. Those blocks pass through
    untouched, and if the block was echo-only -- where near speech cannot
    explain the excess -- the filter is also pulled back toward zero. This
    bounds the output by the input no matter how badly adaptation misbehaves,
    which matters most when there is no echo path to find in the first place.
    """
    B = block
    K = max(1, -(-reach // B))  # partitions, ceil division
    N = 2 * B
    n = min(len(near), len(far))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    pad = (-n) % B
    near_p = np.concatenate([near[:n], np.zeros(pad, np.float32)])
    far_p = np.concatenate([far[:n], np.zeros(pad, np.float32)])
    nblocks = len(near_p) // B

    W = np.zeros((K, N), dtype=np.complex128)  # one frequency-domain partition each
    # Far-frame history, newest first: partition k convolves against the frame
    # from k blocks ago, which is what makes the partitions tile a long filter.
    hist = np.zeros((K, N), dtype=np.complex128)
    # Per-bin far power (EMA) summed over the whole filter span, warm-started
    # off zero so the first blocks are not divided by nothing.
    P = np.full(N, K * N * float(np.mean(far_p**2) + 1e-12))
    x_prev = np.zeros(B, dtype=np.float32)
    out = np.zeros(len(near_p), dtype=np.float32)
    adapted = 0
    diverged = 0
    run = 0  # consecutive echo-only blocks that came out louder

    for b in range(nblocks):
        x = far_p[b * B : (b + 1) * B]
        d = near_p[b * B : (b + 1) * B]
        hist = np.roll(hist, 1, axis=0)
        hist[0] = np.fft.fft(np.concatenate([x_prev, x]))
        y = np.real(np.fft.ifft((W * hist).sum(axis=0)))[B:]  # overlap-save
        e = d - y
        P = lam * P + (1 - lam) * (np.abs(hist) ** 2).sum(axis=0)
        near_pow = float(np.mean(d**2))
        echo_only = near_pow < dtd_ratio * float(np.mean(x**2))

        if float(np.mean(e**2)) > near_pow:
            # The filter added energy, so keep the raw block: cancellation must
            # never make things louder. A stray block is just adaptation
            # transient, but a *run* of echo-only blocks coming out louder means
            # the filter is walking away from the solution, so pull it back.
            e = d
            diverged += 1
            if echo_only:
                run += 1
                if run >= DIVERGENCE_RUN:
                    W *= 0.5
                    run = 0
        elif echo_only:
            run = 0
            E = np.fft.fft(np.concatenate([np.zeros(B), e]))
            delta = reg_frac * float(P.mean()) + 1e-12  # floor weak bins to avg scale
            G = np.conj(hist) * (E / (P + delta))
            g = np.real(np.fft.ifft(G, axis=1))
            g[:, B:] = 0  # gradient constraint: keep only the causal B taps each
            W = (1 - leak) * W + mu * np.fft.fft(g, axis=1)
            adapted += 1

        x_prev = x
        out[b * B : (b + 1) * B] = e

    log.debug(
        "aec: %d partitions x %d taps, %d/%d blocks adapted, %d rejected as diverging",
        K,
        B,
        adapted,
        nblocks,
        diverged,
    )
    return out[:n].astype(np.float32)


def erle_db(
    near: np.ndarray, cleaned: np.ndarray, far: np.ndarray, dtd_ratio: float = 0.05
) -> float:
    """Echo Return Loss Enhancement in dB, measured over echo-only frames.

    Only frames where the system is playing AND the mic is near-quiet are true
    echo -- double-talk frames are full of your own voice, which must survive, so
    including them would understate the number. Higher = more echo removed. 0.0
    means no measurable echo-only frames existed (e.g. nothing ever played).
    """
    n = min(len(near), len(cleaned), len(far))
    if n == 0:
        return 0.0
    near, cleaned, far = near[:n], cleaned[:n], far[:n]
    fr = 2048
    nf = n // fr
    if nf == 0:
        return 0.0
    far_frames = [float(np.mean(far[i * fr : (i + 1) * fr] ** 2)) for i in range(nf)]
    far_med = float(np.median(far_frames))
    reduction = []
    for i in range(nf):
        s = slice(i * fr, (i + 1) * fr)
        far_p = far_frames[i]
        before = float(np.mean(near[s] ** 2))
        if far_p < far_med or before > dtd_ratio * far_p:  # not echo-only
            continue
        after = float(np.mean(cleaned[s] ** 2))
        if before > 1e-9 and after > 1e-12:
            reduction.append(before / after)
    if not reduction:
        return 0.0
    return float(10.0 * np.log10(np.median(reduction)))
