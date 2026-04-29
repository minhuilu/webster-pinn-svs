#!/usr/bin/env python3
"""
HNR + Voiced-Frames Evaluator

Computes frame-wise HNR (in dB) with a robust, voiced-frame-only protocol:
- Optionally aligns predicted to reference via cross-correlation (to reduce shift bias)
- Uses reference F0 (if provided) to pick a narrow ACF search band per frame
- Otherwise searches a global lag band from [lo, hi] Hz
- Masks unvoiced/unstable frames (low energy or weak periodicity)
- Reports median and IQR (25/75 percentiles) for hat/ref, plus counts

Usage:
  python tools/hnr_voicing_eval.py \
    --hat exp/A_static/ckps/r1/a/a_hat_best.wav \
    --ref data/synthetic/vowels/a/wav/a.wav \
    --f0  data/synthetic/vowels/a/f0/a.npy \
    --sr 16000 --frame_ms 30 --hop 256 \
    --align 1 --ignore_ms 300 --max_shift_sec 0.30 \
    --lo 60 --hi 600 \
    --out_csv exp/A_static/ckps/r1/a/hnr_voicing_eval.csv \
    --plot exp/A_static/ckps/r1/a/hnr_hist.png

Outputs CSV columns:
  tag,hat,ref,f0,frames,voiced_frames,hnr_hat_med,hnr_hat_p25,hnr_hat_p75,hnr_ref_med,hnr_ref_p25,hnr_ref_p75
"""
from __future__ import annotations
import argparse, pathlib, sys, math
import numpy as np
import soundfile as sf

def _xcorr_align(x_hat: np.ndarray, x_ref: np.ndarray, sr: int, ignore_ms=300, max_shift_sec=0.30):
    ig = int(ignore_ms * 1e-3 * sr)
    m  = min(len(x_hat), len(x_ref))
    a, b = ig, max(ig, m - ig)
    if b <= a:
        return x_hat, x_ref, 0
    xh0 = x_hat[a:b].astype(np.float32)
    xr0 = x_ref[a:b].astype(np.float32)

    # zero-mean, unit-std
    def _norm(v):
        v = v - v.mean(); s = v.std() + 1e-8; return v / s
    n = 1 << (max(len(xh0), len(xr0)) - 1).bit_length()
    xh = np.zeros(n*2, dtype=np.float32); xr = np.zeros(n*2, dtype=np.float32)
    xh[:len(xh0)] = _norm(xh0); xr[:len(xr0)] = _norm(xr0)
    XH = np.fft.rfft(xh); XR = np.fft.rfft(xr)
    r  = np.fft.irfft(XH * np.conj(XR))
    r  = np.roll(r, len(xh0)-1)
    k  = int(np.argmax(r))
    shift = k - (len(xh0)-1)
    max_shift = int(max_shift_sec * sr)
    shift = max(-max_shift, min(max_shift, shift))

    if shift > 0:
        xh_aln = x_hat[shift:]; xr_aln = x_ref[:len(xh_aln)]
    elif shift < 0:
        xr_aln = x_ref[-shift:]; xh_aln = x_hat[:len(xr_aln)]
    else:
        xh_aln, xr_aln = x_hat, x_ref
    L = min(len(xh_aln), len(xr_aln))
    if L <= 0:
        return x_hat, x_ref, 0
    return xh_aln[:L], xr_aln[:L], shift

def _frame_signal(x: np.ndarray, sr: int, frame_ms=30, hop=256):
    N = len(x)
    wlen = int(round(frame_ms * 1e-3 * sr))
    if wlen <= 8: wlen = 8
    n_frames = 1 + max(0, (N - wlen) // hop)
    frames = np.zeros((n_frames, wlen), dtype=np.float32)
    for n in range(n_frames):
        i0 = n*hop; i1 = i0 + wlen
        frames[n, :] = x[i0:i1]
    # Hann window to stabilize ACF
    w = np.hanning(wlen).astype(np.float32)
    frames = frames * w[None, :]
    return frames

def _acf_power(fr: np.ndarray):
    # r = ifft( |FFT(fr)|^2 )
    X = np.fft.rfft(fr)
    pow_spec = (X.real*X.real + X.imag*X.imag)
    r = np.fft.irfft(pow_spec)
    return r

def _hnr_db_from_acf(r: np.ndarray, lag_lo: int, lag_hi: int):
    r0 = float(max(r[0], 1e-8))
    lag_lo = max(1, lag_lo); lag_hi = max(lag_lo, lag_hi)
    rp = float(np.max(r[lag_lo:lag_hi+1]))
    return 10.0 * math.log10(max(rp, 1e-12) / max(r0 - rp, 1e-12)), rp / r0

def _interp_f0_to_frames(f0_samples: np.ndarray, n_frames: int, hop: int, sr: int, frame_ms=30):
    # frame center times
    wlen = int(round(frame_ms * 1e-3 * sr))
    centers = (np.arange(n_frames)*hop + 0.5*wlen) / sr
    t = np.arange(len(f0_samples)) / sr
    return np.interp(centers, t, f0_samples).astype(np.float32)

def evaluate(hat: np.ndarray, ref: np.ndarray, sr: int, hop: int, frame_ms: int,
             f0_ref: np.ndarray | None, lo: float, hi: float,
             align=True, ignore_ms=300, max_shift_sec=0.30,
             rho_thresh=0.05, rms_thresh=1e-3):
    # normalize to unit std
    def _norm(x):
        x = x.astype(np.float32)
        x = x - x.mean(); s = x.std() + 1e-8; return x / s
    hat = _norm(hat); ref = _norm(ref)

    if align:
        hat, ref, _ = _xcorr_align(hat, ref, sr, ignore_ms=ignore_ms, max_shift_sec=max_shift_sec)

    Fh = _frame_signal(hat, sr, frame_ms=frame_ms, hop=hop)
    Fr = _frame_signal(ref, sr, frame_ms=frame_ms, hop=hop)
    nF = min(len(Fh), len(Fr))
    Fh = Fh[:nF]; Fr = Fr[:nF]

    if f0_ref is not None and len(f0_ref) == len(ref):
        f0_frames = _interp_f0_to_frames(f0_ref, nF, hop, sr, frame_ms)
        f0_frames = np.clip(f0_frames, lo, hi)
    else:
        f0_frames = None

    hnr_hat = np.zeros(nF, dtype=np.float32)
    hnr_ref = np.zeros(nF, dtype=np.float32)
    rho_hat = np.zeros(nF, dtype=np.float32)
    rho_ref = np.zeros(nF, dtype=np.float32)
    rms_hat = np.sqrt((Fh*Fh).mean(axis=1))
    rms_ref = np.sqrt((Fr*Fr).mean(axis=1))

    lag_lo_g = int(sr / hi)
    lag_hi_g = int(sr / lo)

    for i in range(nF):
        r_h = _acf_power(Fh[i])
        r_r = _acf_power(Fr[i])
        if f0_frames is not None and f0_frames[i] > 0:
            lag0 = int(sr / max(f0_frames[i], 1e-6))
            w = int(0.08 * lag0)  # +/-8% search
            l0, l1 = max(1, lag0 - w), min(len(r_h)-1, lag0 + w)
        else:
            l0, l1 = lag_lo_g, min(len(r_h)-1, lag_hi_g)
        hnr_hat[i], rho_hat[i] = _hnr_db_from_acf(r_h, l0, l1)
        hnr_ref[i], rho_ref[i] = _hnr_db_from_acf(r_r, l0, l1)

    # voiced mask: adequate energy and periodicity
    m_voiced = (rms_ref > rms_thresh) & (rho_ref > rho_thresh)
    if f0_frames is not None:
        m_voiced &= (f0_frames >= lo) & (f0_frames <= hi)

    # stats
    def stats(v: np.ndarray, m: np.ndarray):
        if m.sum() == 0:
            return float('nan'), float('nan'), float('nan')
        vv = v[m]
        p25 = float(np.percentile(vv, 25))
        p50 = float(np.percentile(vv, 50))
        p75 = float(np.percentile(vv, 75))
        return p50, p25, p75

    hat_med, hat_p25, hat_p75 = stats(hnr_hat, m_voiced)
    ref_med, ref_p25, ref_p75 = stats(hnr_ref, m_voiced)

    return {
        'frames': int(nF),
        'voiced_frames': int(m_voiced.sum()),
        'hnr_hat_med': hat_med, 'hnr_hat_p25': hat_p25, 'hnr_hat_p75': hat_p75,
        'hnr_ref_med': ref_med, 'hnr_ref_p25': ref_p25, 'hnr_ref_p75': ref_p75,
        'hnr_hat_all': hnr_hat, 'hnr_ref_all': hnr_ref, 'voiced_mask': m_voiced,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hat', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--f0', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--frame_ms', type=int, default=30)
    ap.add_argument('--hop', type=int, default=256)
    ap.add_argument('--align', type=int, default=1)
    ap.add_argument('--ignore_ms', type=int, default=300)
    ap.add_argument('--max_shift_sec', type=float, default=0.30)
    ap.add_argument('--lo', type=float, default=60.0)
    ap.add_argument('--hi', type=float, default=600.0)
    ap.add_argument('--out_csv', default='')
    ap.add_argument('--plot', default='')
    args = ap.parse_args()

    y_hat, sr_h = sf.read(args.hat)
    y_ref, sr_r = sf.read(args.ref)
    assert sr_h == sr_r == args.sr, f'SR mismatch: {sr_h} / {sr_r} vs {args.sr}'

    f0 = None
    if args.f0 and pathlib.Path(args.f0).exists():
        f0 = np.load(args.f0).astype(np.float32)

    res = evaluate(
        hat=y_hat.astype(np.float32), ref=y_ref.astype(np.float32), sr=args.sr,
        hop=args.hop, frame_ms=args.frame_ms, f0_ref=f0,
        lo=args.lo, hi=args.hi, align=bool(args.align),
        ignore_ms=args.ignore_ms, max_shift_sec=args.max_shift_sec,
    )

    # optional plot
    if args.plot:
        import matplotlib.pyplot as plt
        h = res['hnr_hat_all']; r = res['hnr_ref_all']; m = res['voiced_mask']
        plt.figure(figsize=(7,3))
        plt.hist(r[m], bins=40, alpha=0.6, label='REF voiced')
        plt.hist(h[m], bins=40, alpha=0.6, label='HAT voiced')
        plt.xlabel('HNR (dB)'); plt.ylabel('Count'); plt.legend(); plt.tight_layout()
        pathlib.Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.plot, dpi=160); plt.close()

    # CSV
    if args.out_csv:
        import csv
        p = pathlib.Path(args.out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not p.exists())
        with open(p, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['tag','hat','ref','f0','frames','voiced_frames',
                            'hnr_hat_med','hnr_hat_p25','hnr_hat_p75',
                            'hnr_ref_med','hnr_ref_p25','hnr_ref_p75'])
            w.writerow([
                args.tag, args.hat, args.ref, args.f0,
                res['frames'], res['voiced_frames'],
                f"{res['hnr_hat_med']:.3f}", f"{res['hnr_hat_p25']:.3f}", f"{res['hnr_hat_p75']:.3f}",
                f"{res['hnr_ref_med']:.3f}", f"{res['hnr_ref_p25']:.3f}", f"{res['hnr_ref_p75']:.3f}",
            ])

    # console summary
    print(f"[HNR] frames={res['frames']} voiced={res['voiced_frames']}")
    print(f"      HAT voiced median={res['hnr_hat_med']:.2f}dB  P25={res['hnr_hat_p25']:.2f}  P75={res['hnr_hat_p75']:.2f}")
    print(f"      REF voiced median={res['hnr_ref_med']:.2f}dB  P25={res['hnr_ref_p25']:.2f}  P75={res['hnr_ref_p75']:.2f}")

if __name__ == '__main__':
    main()

