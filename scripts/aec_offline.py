#!/usr/bin/env python3
"""Offline acoustic echo cancellation over a saved stereo `recording.wav`.

SUPERSEDED for measurement -- use `scripts/echo_delay.py` instead. The `fdaf`
here is the original single-block filter with no delay alignment, so its ERLE
understates what the shipped canceller achieves (0.0-2.5 dB vs 3.5-6.4 dB on
the same recordings). Kept only for its `--transcribe` A/B, which counts how
many `me` lines were really echoes of `them`.

Standalone A/B tool -- NOT wired into the app, no install, numpy-only (whisper
is imported only if you pass --transcribe). Reads a `--keep-audio` recording
(mic = left, system = right), subtracts the system-audio bleed from the mic with
a pure-numpy adaptive filter, and reports how much echo it removed. With
--transcribe it also rebuilds the transcript both ways and counts how many "me"
lines were really echoes of "them".

    python scripts/aec_offline.py "/path/to/<session-dir>/recording.wav"
    python scripts/aec_offline.py "<...>/recording.wav" --transcribe

The filter is a constrained frequency-domain (overlap-save) NLMS -- the same
class of algorithm Speex/WebRTC use internally, minus their double-talk detector
and residual-echo suppressor, so expect decent-not-great cancellation.
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np


def read_stereo(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (left, right, rate) as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 2 and w.getsampwidth() == 2, "expected 16-bit stereo"
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    interleaved = raw.astype(np.float32) / 32768.0
    return interleaved[0::2].copy(), interleaved[1::2].copy(), rate


def write_mono(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def fdaf(
    near: np.ndarray,
    far: np.ndarray,
    block: int = 8192,
    mu: float = 0.5,
    leak: float = 1e-4,
    reg_frac: float = 0.1,
    lam: float = 0.9,
    dtd_ratio: float = 0.05,
) -> np.ndarray:
    """Constrained overlap-save frequency-domain NLMS echo canceller.

    Predicts the echo of `far` (system) present in `near` (mic) with an adaptive
    FIR filter of `block` taps and subtracts it. Returns the residual (cleaned
    near). `block` taps @ 48 kHz ≈ block/48 ms of echo-tail coverage -- must
    exceed the output+acoustic delay.

    Per-bin power normalization (whitens the colored far signal), but weak bins
    are floored to `reg_frac` of the *average* bin power -- flooring to a tiny
    constant instead lets low-energy bins of real speech starve and diverge.
    `leak` bleeds the filter toward zero each block so it can't grow unbounded.

    Double-talk detector: the filter only *adapts* on blocks that look echo-only
    (near power below `dtd_ratio` x far power); when you're also talking it holds
    the current filter and keeps subtracting. Without this the filter mis-adapts
    on double-talk and injects far-end audio into the output. `dtd_ratio` should
    sit a little above the echo-coupling power ratio.
    """
    L = block
    N = 2 * L
    n = min(len(near), len(far))
    pad = (-n) % L
    near = np.concatenate([near[:n], np.zeros(pad, np.float32)])
    far = np.concatenate([far[:n], np.zeros(pad, np.float32)])
    nblocks = len(near) // L

    W = np.zeros(N, dtype=np.complex128)  # frequency-domain filter
    scale = N * float(np.mean(far**2) + 1e-12)  # typical bin power
    P = np.full(N, scale)  # per-bin far power (EMA), warm-started off zero
    x_prev = np.zeros(L, dtype=np.float32)
    out = np.zeros(len(near), dtype=np.float32)

    for b in range(nblocks):
        x = far[b * L : (b + 1) * L]
        d = near[b * L : (b + 1) * L]
        X = np.fft.fft(np.concatenate([x_prev, x]))
        y = np.real(np.fft.ifft(X * W))[L:]  # overlap-save: last L = linear conv
        e = d - y
        E = np.fft.fft(np.concatenate([np.zeros(L), e]))
        P = lam * P + (1 - lam) * np.abs(X) ** 2
        # Double-talk hold: adapt only when the block looks echo-dominated.
        near_pow = float(np.mean(d**2))
        far_pow = float(np.mean(x**2))
        adapting = near_pow < dtd_ratio * far_pow
        if adapting:
            delta = reg_frac * float(P.mean()) + 1e-12  # floor weak bins to avg scale
            G = np.conj(X) * E / (P + delta)
            g = np.real(np.fft.ifft(G))
            g[L:] = 0  # gradient constraint: keep only the causal L taps
            W = (1 - leak) * W + mu * np.fft.fft(g)
        x_prev = x
        out[b * L : (b + 1) * L] = e
    return out[:n]


def erle_db(near: np.ndarray, cleaned: np.ndarray, far: np.ndarray, dtd_ratio: float = 0.05) -> float:
    """Echo Return Loss Enhancement, dB, over echo-only frames.

    Only frames where the system is playing AND the mic is near-quiet (near power
    below `dtd_ratio` x far power) are true echo -- double-talk frames are full of
    your own voice, which must survive, so including them would understate the
    number. Higher = more echo removed.
    """
    n = min(len(near), len(cleaned), len(far))
    near, cleaned, far = near[:n], cleaned[:n], far[:n]
    fr = 2048
    nf = n // fr
    far_med = np.median([np.mean(far[i * fr : (i + 1) * fr] ** 2) for i in range(nf)])
    reduction = []
    for i in range(nf):
        s = slice(i * fr, (i + 1) * fr)
        far_p = np.mean(far[s] ** 2)
        before = np.mean(near[s] ** 2)
        if far_p < far_med or before > dtd_ratio * far_p:  # not echo-only
            continue
        after = np.mean(cleaned[s] ** 2)
        if before > 1e-9 and after > 1e-12:
            reduction.append(before / after)
    if not reduction:
        return 0.0
    return 10.0 * np.log10(np.median(reduction))


# ---- optional transcription A/B (imports whisper only when used) ------------


def _transcribe_channels(cleaned, raw_mic, system, rate, workdir):
    """Transcribe system + raw-mic + cleaned-mic, count me/them duplication."""
    import difflib

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from transcripter.transcriber import make_backend  # noqa: E402
    from transcripter.wavio import read_wav  # noqa: E402

    backend = make_backend()

    def segs(name, samples):
        p = workdir / f"{name}.wav"
        write_mono(p, samples, rate)
        # Backends take 16 kHz mono samples now, so decimate on the way in.
        return backend.transcribe(read_wav(p))

    sys_seg = segs("system", system)
    raw_seg = segs("mic_raw", raw_mic)
    clean_seg = segs("mic_cleaned", cleaned)

    def dup_rate(me_segs):
        """Fraction of 'me' segments that echo a time-overlapping 'them' segment."""
        dup = 0
        for m in me_segs:
            mt = m["text"].strip().lower()
            if not mt:
                continue
            for t in sys_seg:
                if t["end"] < m["start"] - 1 or t["start"] > m["end"] + 1:
                    continue
                sim = difflib.SequenceMatcher(None, mt, t["text"].strip().lower()).ratio()
                if sim > 0.6:
                    dup += 1
                    break
        total = sum(1 for m in me_segs if m["text"].strip())
        return dup, total

    return raw_seg, clean_seg, sys_seg, dup_rate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path, help="stereo recording.wav (mic=L, system=R)")
    ap.add_argument("--block", type=int, default=8192, help="filter taps (echo-tail coverage)")
    ap.add_argument("--mu", type=float, default=0.5, help="adaptation step size")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: alongside input)")
    ap.add_argument("--transcribe", action="store_true", help="also run the me/them dedup A/B")
    args = ap.parse_args()

    outdir = args.out or args.recording.parent / "aec-ab"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.recording} ...")
    mic, system, rate = read_stereo(args.recording)
    print(f"  {len(mic) / rate:.0f}s @ {rate} Hz; block={args.block} (~{args.block * 1000 // rate}ms tail)")

    print("running FDAF echo cancellation ...")
    cleaned = fdaf(mic, system, block=args.block, mu=args.mu)

    erle = erle_db(mic, cleaned, system)
    print(f"\nERLE (echo removed where system is active): {erle:.1f} dB")
    write_mono(outdir / "mic_raw.wav", mic, rate)
    write_mono(outdir / "mic_cleaned.wav", cleaned, rate)
    print(f"wrote mic_raw.wav / mic_cleaned.wav to {outdir}  (listen to compare)")

    if args.transcribe:
        print("\ntranscribing system + raw-mic + cleaned-mic (this is the slow part) ...")
        raw_seg, clean_seg, _sys, dup_rate = _transcribe_channels(
            cleaned, mic, system, rate, outdir
        )
        rd, rt = dup_rate(raw_seg)
        cd, ct = dup_rate(clean_seg)
        print("\n--- me/them duplication (lower is better) ---")
        print(f"  raw mic:     {rd}/{rt} 'me' lines echo a 'them' line ({100 * rd / max(rt,1):.0f}%)")
        print(f"  cleaned mic: {cd}/{ct} 'me' lines echo a 'them' line ({100 * cd / max(ct,1):.0f}%)")


if __name__ == "__main__":
    main()
