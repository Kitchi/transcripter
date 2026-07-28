"""Acoustic echo cancellation: remove system-audio bleed from the mic track.

System audio leaves the speakers and re-enters the mic, so the far end's words
land on both channels. Subtracting the predicted echo before transcription beats
deleting duplicate text afterwards, because it also recovers **double-talk** --
the moments you and the far end speak at once, which text-level dedup throws
away wholesale.

The filter is a constrained frequency-domain (overlap-save) NLMS, the same class
of algorithm Speex/WebRTC use internally, minus their residual-echo suppressor.
Expect decent-not-great cancellation; `transcript.py`'s bleed dedup stays on as
a net for whatever survives.

Two safeguards matter as much as the filter itself, because the common failure
mode in practice is not weak cancellation but *divergence*. When there is no
echo path at all -- headphones, or a conferencing app that already ran its own
AEC -- every adaptation fits mic noise against a loud, uncorrelated reference,
and the filter random-walks until it drowns the signal it was meant to clean.
So: `echo_coherence` lets a caller detect a null echo path and skip the filter
entirely, and `cancel_echo` never returns a block louder than it received.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# Consecutive echo-only blocks that must come out louder before the filter is
# pulled back toward zero. One or two is ordinary adaptation transient; a run
# means the filter is diverging.
DIVERGENCE_RUN = 3


def echo_coherence(
    near: np.ndarray,
    far: np.ndarray,
    rate: int = 16_000,
    frame: int = 2048,
    band: tuple[float, float] = (300.0, 3400.0),
) -> float:
    """Mean magnitude-squared coherence between mic and far end, 0..1.

    A linear echo path makes the mic partly a filtered copy of the far signal,
    which coherence detects regardless of the delay or the room response -- so
    this answers "is there any echo to cancel?" without first converging a
    filter. Measured only over frames where the far end is actually playing
    (above its median frame power); silent frames carry no information about
    the path and would just dilute the average.

    Restricted to the speech band because that is where the echo lives and where
    both channels have energy to compare. Values near 0 mean no echo path:
    headphones, or an app that already cancelled it. Real acoustic coupling in
    a room runs well above 0.1 even when the echo is quiet.
    """
    n = min(len(near), len(far))
    nf = n // frame
    if nf == 0:
        return 0.0
    near_f = near[: nf * frame].reshape(nf, frame)
    far_f = far[: nf * frame].reshape(nf, frame)
    power = (far_f.astype(np.float64) ** 2).mean(axis=1)
    active = power > max(float(np.median(power)), 1e-12)
    if not active.any():
        return 0.0

    window = np.hanning(frame)
    X = np.fft.rfft(far_f[active] * window, axis=1)
    D = np.fft.rfft(near_f[active] * window, axis=1)
    # Cross- and auto-spectra averaged over frames: coherence is only meaningful
    # as an average, since any single frame is trivially "coherent".
    Sxy = (D * np.conj(X)).mean(axis=0)
    Sxx = (np.abs(X) ** 2).mean(axis=0)
    Syy = (np.abs(D) ** 2).mean(axis=0)
    coh = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-20)

    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    in_band = (freqs >= band[0]) & (freqs < band[1])
    if not in_band.any():
        return 0.0
    return float(coh[in_band].mean())


def cancel_echo(
    near: np.ndarray,
    far: np.ndarray,
    taps: int = 4096,
    mu: float = 0.2,
    leak: float = 1e-3,
    reg_frac: float = 0.1,
    lam: float = 0.9,
    dtd_ratio: float = 0.05,
) -> np.ndarray:
    """Predict the echo of `far` present in `near` and subtract it.

    Returns the cleaned near signal, same length as the shorter input. `taps`
    sets the echo-tail coverage (taps/rate seconds) and must exceed the combined
    output and acoustic delay from speaker to mic.

    Per-bin power normalization whitens the colored far signal, but weak bins are
    floored to `reg_frac` of the *average* bin power -- flooring to a tiny
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
    L = taps
    N = 2 * L
    n = min(len(near), len(far))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    pad = (-n) % L
    near_p = np.concatenate([near[:n], np.zeros(pad, np.float32)])
    far_p = np.concatenate([far[:n], np.zeros(pad, np.float32)])
    nblocks = len(near_p) // L

    W = np.zeros(N, dtype=np.complex128)  # frequency-domain filter
    scale = N * float(np.mean(far_p**2) + 1e-12)  # typical bin power
    P = np.full(N, scale)  # per-bin far power (EMA), warm-started off zero
    x_prev = np.zeros(L, dtype=np.float32)
    out = np.zeros(len(near_p), dtype=np.float32)
    adapted = 0
    diverged = 0
    run = 0  # consecutive echo-only blocks that came out louder

    for b in range(nblocks):
        x = far_p[b * L : (b + 1) * L]
        d = near_p[b * L : (b + 1) * L]
        X = np.fft.fft(np.concatenate([x_prev, x]))
        y = np.real(np.fft.ifft(X * W))[L:]  # overlap-save: last L = linear conv
        e = d - y
        P = lam * P + (1 - lam) * np.abs(X) ** 2  # far statistics, every block
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
            E = np.fft.fft(np.concatenate([np.zeros(L), e]))
            delta = reg_frac * float(P.mean()) + 1e-12  # floor weak bins to avg scale
            G = np.conj(X) * E / (P + delta)
            g = np.real(np.fft.ifft(G))
            g[L:] = 0  # gradient constraint: keep only the causal L taps
            W = (1 - leak) * W + mu * np.fft.fft(g)
            adapted += 1

        x_prev = x
        out[b * L : (b + 1) * L] = e

    log.debug("aec: %d/%d blocks adapted, %d rejected as diverging", adapted, nblocks, diverged)
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
