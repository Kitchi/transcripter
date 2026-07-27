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
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


def cancel_echo(
    near: np.ndarray,
    far: np.ndarray,
    taps: int = 4096,
    mu: float = 0.5,
    leak: float = 1e-4,
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

    for b in range(nblocks):
        x = far_p[b * L : (b + 1) * L]
        d = near_p[b * L : (b + 1) * L]
        X = np.fft.fft(np.concatenate([x_prev, x]))
        y = np.real(np.fft.ifft(X * W))[L:]  # overlap-save: last L = linear conv
        e = d - y
        E = np.fft.fft(np.concatenate([np.zeros(L), e]))
        P = lam * P + (1 - lam) * np.abs(X) ** 2
        if float(np.mean(d**2)) < dtd_ratio * float(np.mean(x**2)):
            delta = reg_frac * float(P.mean()) + 1e-12  # floor weak bins to avg scale
            G = np.conj(X) * E / (P + delta)
            g = np.real(np.fft.ifft(G))
            g[L:] = 0  # gradient constraint: keep only the causal L taps
            W = (1 - leak) * W + mu * np.fft.fft(g)
            adapted += 1
        x_prev = x
        out[b * L : (b + 1) * L] = e

    log.debug("aec: %d/%d blocks adapted", adapted, nblocks)
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
