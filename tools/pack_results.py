#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pack_results.py
Export main results in one pass(three paths×)robustness().
- main results(main):for /a,i,u/ generate or collect 3 :
    1) ddsp  baseline(skip with a message if missing)
2) pinn in-graph(direct in-graph training output; {ckps}/{rid}/{tag}/{tag}_hat_best.wav)
3) pinn post (independent FDTD–WebsterRender;reuse if already present)
+ Metrics metrics/main_summary.csv
- robustness(robust):fix {Ahat, zeta},run three perturbation types for each vowel and save figures plus metrics/robustness.csv

Dependencies matching this repository:
  exp.A_static.audio_losses.multi_stft_loss, lsd_db
  exp.A_static.synthesize_ref.webster_1d_fd
  exp.common.singer.get_profile
"""

import os, sys, pathlib, argparse, csv, warnings
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

# ---- Project-local tools ----
try:
    from exp.A_static.audio_losses import multi_stft_loss, lsd_db
except Exception as e:
    multi_stft_loss = None
    lsd_db = None
    warnings.warn(f"[WARN] import audio_losses failed: {e}\nWill use fallback LSD only.")

from exp.A_static.synthesize_ref import webster_1d_fd
from exp.common.singer import get_profile

# ---------------- Small utilities ----------------
def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True); return p

def load_wav(path: pathlib.Path, sr=16000):
    y, sr_ = sf.read(str(path))
    assert y.ndim == 1, f"mono expected: {path}"
    assert sr_ == sr, f"SR mismatch: {sr_} vs {sr} ({path})"
    return y.astype(np.float32), sr

def plot_spec_png(x, sr, out_png, title, use_mel=False, ymax=8000, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import librosa, librosa.display
        if use_mel:
            S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=256, n_mels=128, fmax=ymax)
            SdB = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(8,3), dpi=dpi)
            librosa.display.specshow(SdB, sr=sr, hop_length=256, x_axis="time", y_axis="mel", fmax=ymax)
        else:
            S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) + 1e-12
            SdB = 20*np.log10(S/np.max(S))
            plt.figure(figsize=(8,3), dpi=dpi)
            librosa.display.specshow(SdB, sr=sr, hop_length=256, x_axis="time", y_axis="linear")
        plt.ylim(0, ymax)
        plt.title(title)
        plt.colorbar(format="%+0.1f dB")
        plt.tight_layout()
        plt.savefig(str(out_png))
        plt.close()
    except Exception:
        # fall back to matplotlib.specgram
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,3), dpi=dpi)
        Pxx, freqs, bins, im = plt.specgram(x, NFFT=1024, Fs=sr, noverlap=768, cmap="magma")
        plt.ylim(0, ymax)
        plt.title(title)
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(str(out_png))
        plt.close()

def lsd_db_fallback(x, y, n_fft=2048, hop=512, eps=1e-8):
    """Simple LSD fallback: align length and compute L2 distance on log spectra"""
    import numpy as np
    N = min(len(x), len(y))
    x = x[:N]; y = y[:N]
    # simple STFT
    def stft_mag(z):
        import numpy as np
        w = np.hanning(n_fft)
        frames = []
        for i in range(0, N - n_fft + 1, hop):
            seg = z[i:i+n_fft] * w
            Z = np.fft.rfft(seg)
            frames.append(np.abs(Z) + eps)
        if not frames:
            return np.abs(np.fft.rfft(z[:n_fft]))[None, :] + eps
        return np.stack(frames, 0)
    X = stft_mag(x); Y = stft_mag(y)
    Xdb = 20*np.log10(X); Ydb = 20*np.log10(Y)
    return float(np.mean(np.sqrt(np.mean((Xdb - Ydb)**2, axis=1))))

def mrstft_loss_fallback(x, y):
    """Return zero as a fallback; reported values rely on LSD/HNR."""
    return 0.0

def hnr_framewise_db(x, sr, f0_series=None, f0_hz=None, win=0.04, hop=0.01, f0_band=0.10):
    """
HNR:ACFF0+/-band;ReturndB
    """
    x = np.asarray(x, dtype=np.float32)
    N = len(x)
    wlen = int(round(win * sr))
    hopn = int(round(hop * sr))
    if f0_series is None and f0_hz is None:
        raise ValueError("need f0_series or f0_hz")
    out = []
    for s in range(0, max(1, N - wlen + 1), hopn):
        seg = x[s:s+wlen]
        if len(seg) < wlen: break
        seg = (seg - seg.mean()) / (np.std(seg) + 1e-8)
        n2 = int(1 << (seg.size-1).bit_length())
        X = np.fft.rfft(seg, n2)
        r = np.fft.irfft(X * np.conj(X))
        r = r[:wlen]  # ACF
        r0 = max(r[0], 1e-8)

        if f0_series is not None:
            # Use the approximate f0 at the frame center
            idx = min(len(f0_series)-1, s + wlen//2)
            f0 = float(f0_series[idx])
        else:
            f0 = float(f0_hz)
        if f0 <= 0: 
            continue

        T0 = sr / f0
        lo = int(max(1, np.floor(T0 * (1.0 - f0_band))))
        hi = int(min(len(r)-1, np.ceil (T0 * (1.0 + f0_band))))
        if hi <= lo:
            continue
        rp = np.max(r[lo:hi+1])
        if rp <= 0 or rp >= r0:  # numerical guard
            continue
        hnr = 10.0 * np.log10(rp / (r0 - rp + 1e-8))
        out.append(hnr)
    if not out:
        return float("nan")
    return float(np.median(out))

def read_zeta_file(path: pathlib.Path):
    txt = path.read_text().strip()
    vals = []
    for p in txt.replace(",", " ").split():
        try: vals.append(float(p))
        except: pass
    if not vals: return None
    return float(vals[0])

def find_ckp_dir(rid: str, tag: str):
    return pathlib.Path(f"exp/A_static/ckps/{rid}/{tag}")

def find_ref_paths(tag: str, sr=16000):
    wav = pathlib.Path(f"data/synthetic/vowels/{tag}/wav/{tag}.wav")
    f0p = pathlib.Path(f"data/synthetic/vowels/{tag}/f0/{tag}.npy")
    assert wav.exists(), f"missing ref wav: {wav}"
    f0 = np.load(f0p) if f0p.exists() else None
    return wav, f0

def find_A_and_zeta(ckp_dir: pathlib.Path, tag: str):
    A1 = ckp_dir / f"{tag}_Ahat.npy"
    A2 = ckp_dir / f"{tag}_A_curve.npy"
    assert A1.exists() or A2.exists(), f"no Ahat/A_curve in {ckp_dir}"
    A = np.load(A1 if A1.exists() else A2).astype(np.float32).squeeze()
    zpath = ckp_dir / f"{tag}_zeta.txt"
    zeta = read_zeta_file(zpath) if zpath.exists() else 0.06
    return A, float(zeta)

def area_interp_fn(Avec, L):
    xp = np.linspace(0.0, float(L), len(Avec))
    def Ax(x_phys):
        return np.interp(x_phys, xp, np.asarray(Avec), left=Avec[0], right=Avec[-1]).astype(np.float32)
    return Ax

def compute_metrics(y_hat, y_ref, sr, f0_series=None, f0_hz=None):
    # Normalize
    y_hat = y_hat.astype(np.float32); y_ref = y_ref.astype(np.float32)
    y_hat = y_hat / (np.std(y_hat) + 1e-8)
    y_ref = y_ref / (np.std(y_ref) + 1e-8)

    # mSTFT/LSD
    if multi_stft_loss is not None and lsd_db is not None:
        with torch.no_grad():
            m = float(multi_stft_loss(torch.tensor(y_hat), torch.tensor(y_ref)).item())
            l = float(lsd_db(torch.tensor(y_hat), torch.tensor(y_ref)).item())
    else:
        m = mrstft_loss_fallback(y_hat, y_ref)
        l = lsd_db_fallback(y_hat, y_ref)

    # HNR()
    if f0_series is not None:
        h = hnr_framewise_db(y_hat, sr, f0_series=f0_series, f0_hz=None)
        h_ref = hnr_framewise_db(y_ref, sr, f0_series=f0_series, f0_hz=None)
    else:
        h = hnr_framewise_db(y_hat, sr, f0_series=None, f0_hz=f0_hz)
        h_ref = hnr_framewise_db(y_ref, sr, f0_series=None, f0_hz=f0_hz)
    return m, l, h, h_ref

def save_audio_and_spec(y, sr, wav_path: pathlib.Path, png_path: pathlib.Path, title: str):
    ensure_dir(wav_path.parent); ensure_dir(png_path.parent)
    sf.write(str(wav_path), y.astype(np.float32), sr)
    plot_spec_png(y, sr, png_path, title=title, use_mel=False, ymax=5000, dpi=300)

# ---------------- export:three paths× ----------------
def run_main_export(rid: str, tags, out_audio="audio", out_figs="figs", out_csv="metrics/main_summary.csv"):
    sr = 16000
    prof = get_profile()  # L, f0 default
    out_csv = pathlib.Path(out_csv)
    ensure_dir(out_csv.parent)

    header = ["vowel","method","mSTFT","LSD_dB","HNR_frame_median_dB","HNR_ref_dB","run_id"]
    rows = []

    for tag in tags:
        print(f"\n=== Vowel: {tag} ===")
        ckp = find_ckp_dir(rid, tag)
        ref_wav_path, f0_arr = find_ref_paths(tag, sr=sr)
        y_ref, _ = load_wav(ref_wav_path, sr=sr)
        f0_hz = float(np.median(f0_arr)) if f0_arr is not None else float(prof["f0_by_vowel"].get(tag, 200.0))

        # ---- 1) DDSP-only baseline(optional)----
        # Adjust this path if needed; missing files are skipped
        ddsp_cands = [
            pathlib.Path(f"exp/A_static/ckps/r0/{tag}/{tag}_ddsp_best.wav"),
            pathlib.Path(f"exp/DDSP/{rid}/{tag}/{tag}_hat.wav"),
        ]
        ddsp_wav = next((p for p in ddsp_cands if p.exists()), None)
        if ddsp_wav is not None:
            y_hat, _ = load_wav(ddsp_wav, sr=sr)
            # Copy output and plot spectrogram
            wav_out = pathlib.Path(out_audio) / tag / "ddsp.wav"
            png_out = pathlib.Path(out_figs) / tag / "ddsp_spec.png"
            save_audio_and_spec(y_hat, sr, wav_out, png_out, f"{tag} • DDSP-only")
            m,l,h,hr = compute_metrics(y_hat, y_ref, sr, f0_series=f0_arr, f0_hz=f0_hz)
            rows.append([tag,"ddsp", m,l,h,hr, rid])
        else:
            print(f"[SKIP] DDSP baseline not found for {tag}")

        # ---- 2) PINN in-graph(direct in-graph training output)----
        in_graph_cands = [
            ckp / f"{tag}_hat_best.wav",
            ckp / f"hat_best.wav",
        ]
        in_graph = next((p for p in in_graph_cands if p.exists()), None)
        if in_graph is not None:
            y_hat, _ = load_wav(in_graph, sr=sr)
            wav_out = pathlib.Path(out_audio) / tag / "pinn_ingraph.wav"
            png_out = pathlib.Path(out_figs) / tag / "pinn_ingraph_spec.png"
            save_audio_and_spec(y_hat, sr, wav_out, png_out, f"{tag} • PINN in-graph")
            m,l,h,hr = compute_metrics(y_hat, y_ref, sr, f0_series=f0_arr, f0_hz=f0_hz)
            rows.append([tag,"pinn_ingraph", m,l,h,hr, rid])
        else:
            print(f"[SKIP] in-graph not found for {tag} in {ckp}")

        # ---- 3) PINN post-render(independent FDTD Render/)----
        # 3.1 Ahat & zeta
        Ahat, zeta = find_A_and_zeta(ckp, tag)
        L = float(prof["L"])
        Ax = area_interp_fn(Ahat, L)
        dur = len(y_ref)/sr

        # 3.2 post/hat_robin.wav ,Render
        post_dir = ensure_dir(ckp / "post")
        post_wav = post_dir / "hat_robin.wav"
        if post_wav.exists():
            y_post, _ = load_wav(post_wav, sr=sr)
        else:
            y_post, _ = webster_1d_fd(
                Ax_fn=Ax, f0=f0_hz, dur=dur, sr_out=sr, c=343.0, L=L,
                zeta_ref=float(zeta), Oq=0.6, Cq=0.3, beta=10.0
            )
            sf.write(str(post_wav), y_post.astype(np.float32), sr)

        # 3.3 Copy to unified directory and plot spectrogram
        wav_out = pathlib.Path(out_audio) / tag / "pinn_post.wav"
        png_out = pathlib.Path(out_figs) / tag / "pinn_post_spec.png"
        save_audio_and_spec(y_post, sr, wav_out, png_out, f"{tag} • PINN post (FDTD)")

        # 3.4 Metrics
        m,l,h,hr = compute_metrics(y_post, y_ref, sr, f0_series=f0_arr, f0_hz=f0_hz)
        rows.append([tag,"pinn_post", m,l,h,hr, rid])

        print(f"[OK] {tag} done.")

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n[SAVED] {out_csv}  ({len(rows)} rows)")
    print("[HINT] WAV/PNG under ./audio and ./figs")

# ---------------- robustness:three perturbation types ----------------
def run_robust_export(rid: str, tags, out_figs="figs", out_csv="metrics/robustness.csv"):
    """
    three perturbation types:
      (a) numerical/discretization: Nx, cfl_max, beta
      (b) source shape: Oq, Cq
      (c) f0: +/-10%
    Outputs:
      - one small panel per type, with baseline plus two perturbations
      - CSV records:vowel, case, setting, mSTFT, LSD_dB, HNR, Delta LSD, Delta HNR
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sr = 16000
    prof = get_profile()
    out_csv = pathlib.Path(out_csv)
    ensure_dir(out_csv.parent)

    header = ["vowel","case","setting","mSTFT","LSD_dB","HNR_frame_dB","HNR_ref_dB","delta_LSD","delta_HNR","run_id"]
    rows = []

    for tag in tags:
        print(f"\n=== Robustness: {tag} ===")
        ckp = find_ckp_dir(rid, tag)
        Ahat, zeta = find_A_and_zeta(ckp, tag)
        L = float(prof["L"])
        Ax = area_interp_fn(Ahat, L)

        ref_wav_path, f0_arr = find_ref_paths(tag, sr=sr)
        y_ref, _ = load_wav(ref_wav_path, sr=sr)
        f0_hz = float(np.median(f0_arr)) if f0_arr is not None else float(prof["f0_by_vowel"].get(tag, 200.0))
        dur = len(y_ref)/sr

        # baseline Render(for)
        base = dict(Nx=256, cfl_max=0.8, beta=10.0, Oq=0.6, Cq=0.3, f0=f0_hz)
        y_base, _ = webster_1d_fd(Ax_fn=Ax, f0=base["f0"], dur=dur, sr_out=sr,
                                  c=343.0, L=L, zeta_ref=zeta, beta=base["beta"],
                                  Oq=base["Oq"], Cq=base["Cq"])
        m0,l0,h0,hr0 = compute_metrics(y_base, y_ref, sr, f0_series=f0_arr, f0_hz=f0_hz)

        # ---------- (a) numerical/discretization ----------
        cases_a = [
            dict(Nx=192, cfl_max=0.7, beta=base["beta"], Oq=base["Oq"], Cq=base["Cq"], f0=base["f0"], name="Nx192,CFL0.7"),
            dict(Nx=320, cfl_max=0.9, beta=14.0,        Oq=base["Oq"], Cq=base["Cq"], f0=base["f0"], name="Nx320,CFL0.9,β14"),
        ]
        fig_a, axes_a = plt.subplots(1, 3, figsize=(12,3), dpi=300)
        for j, cfg in enumerate([{"name":"baseline", **base}] + cases_a):
            y, _ = webster_1d_fd(Ax_fn=Ax, f0=cfg["f0"], dur=dur, sr_out=sr,
                                 c=343.0, L=L, zeta_ref=zeta, beta=cfg["beta"],
                                 Oq=cfg["Oq"], Cq=cfg["Cq"], Nx=cfg["Nx"], cfl_max=cfg["cfl_max"])
            m,l,h,hr = compute_metrics(y, y_ref, sr, f0_series=f0_arr, f0_hz=f0_hz)
            rows.append([tag,"discret", cfg["name"], m,l,h,hr, l-l0, h-h0, rid])

            # :
            try:
                import librosa, librosa.display
                S = np.abs(librosa.stft(y/ (np.std(y)+1e-8), n_fft=1024, hop_length=256)) + 1e-12
                SdB = 20*np.log10(S/np.max(S))
                librosa.display.specshow(SdB, sr=sr, hop_length=256, x_axis="time", y_axis="linear", ax=axes_a[j], cmap="magma")
                axes_a[j].set_ylim(0, 5000); axes_a[j].set_title(cfg["name"])
            except Exception:
                axes_a[j].specgram(y, NFFT=1024, Fs=sr, noverlap=768, cmap="magma")
                axes_a[j].set_ylim(0, 5000); axes_a[j].set_title(cfg["name"])
        ensure_dir(pathlib.Path(out_figs)/tag)
        fig_a.tight_layout()
        fig_a.savefig(str(pathlib.Path(out_figs)/tag/"robust_discret.png")); plt.close(fig_a)

        # ---------- (b) source shape ----------
        cases_b = [
            dict(Oq=0.50, Cq=0.25, name="Oq0.50,Cq0.25"),
            dict(Oq=0.70, Cq=0.40, name="Oq0.70,Cq0.40"),
        ]
        fig_b, axes_b = plt.subplots(1, 3, figsize=(12,3), dpi=300)
        for j, cfg in enumerate([{"name":"baseline", **base}] + cases_b):
            y, _ = webster_1d_fd(Ax_fn=Ax, f0=base["f0"], dur=dur, sr_out=sr,
                                 c=343.0, L=L, zeta_ref=zeta, beta=base["beta"],
                                 Oq=cfg.get("Oq", base["Oq"]), Cq=cfg.get("Cq", base["Cq"]))
            m,l,h,hr = compute_metrics(y, y_ref, sr, f0_series=f0_arr, f0_hz=f0_hz)
            rows.append([tag,"source", cfg["name"], m,l,h,hr, l-l0, h-h0, rid])
            try:
                import librosa, librosa.display
                S = np.abs(librosa.stft(y/(np.std(y)+1e-8), n_fft=1024, hop_length=256)) + 1e-12
                SdB = 20*np.log10(S/np.max(S))
                librosa.display.specshow(SdB, sr=sr, hop_length=256, x_axis="time", y_axis="linear", ax=axes_b[j], cmap="magma")
                axes_b[j].set_ylim(0, 5000); axes_b[j].set_title(cfg["name"])
            except Exception:
                axes_b[j].specgram(y, NFFT=1024, Fs=sr, noverlap=768, cmap="magma")
                axes_b[j].set_ylim(0, 5000); axes_b[j].set_title(cfg["name"])
        fig_b.tight_layout()
        fig_b.savefig(str(pathlib.Path(out_figs)/tag/"robust_source.png")); plt.close(fig_b)

        # ---------- (c) f0 ----------
        cases_c = [
            dict(f0=0.90*base["f0"], name="f0-10%"),
            dict(f0=1.10*base["f0"], name="f0+10%"),
        ]
        fig_c, axes_c = plt.subplots(1, 3, figsize=(12,3), dpi=300)
        for j, cfg in enumerate([{"name":"baseline", **base}] + cases_c):
            y, _ = webster_1d_fd(Ax_fn=Ax, f0=cfg["f0"], dur=dur, sr_out=sr,
                                 c=343.0, L=L, zeta_ref=zeta, beta=base["beta"],
                                 Oq=base["Oq"], Cq=base["Cq"])
            m,l,h,hr = compute_metrics(y, y_ref, sr, f0_series=f0_arr, f0_hz=float(cfg["f0"]))
            rows.append([tag,"f0", cfg["name"], m,l,h,hr, l-l0, h-h0, rid])
            try:
                import librosa, librosa.display
                S = np.abs(librosa.stft(y/(np.std(y)+1e-8), n_fft=1024, hop_length=256)) + 1e-12
                SdB = 20*np.log10(S/np.max(S))
                librosa.display.specshow(SdB, sr=sr, hop_length=256, x_axis="time", y_axis="linear", ax=axes_c[j], cmap="magma")
                axes_c[j].set_ylim(0, 5000); axes_c[j].set_title(cfg["name"])
            except Exception:
                axes_c[j].specgram(y, NFFT=1024, Fs=sr, noverlap=768, cmap="magma")
                axes_c[j].set_ylim(0, 5000); axes_c[j].set_title(cfg["name"])
        fig_c.tight_layout()
        fig_c.savefig(str(pathlib.Path(out_figs)/tag/"robust_f0.png")); plt.close(fig_c)

        print(f"[OK] robust figs saved under {out_figs}/{tag}")

    # CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n[SAVED] {out_csv}  ({len(rows)} rows)")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    apm = sub.add_parser("main", help="export main results: three paths by three vowels")
    apm.add_argument("--rid", default=os.environ.get("RID","main"))
    apm.add_argument("--tags", nargs="+", default=["a","i","u"])
    apm.add_argument("--out_audio", default="audio")
    apm.add_argument("--out_figs",  default="figs")
    apm.add_argument("--out_csv",   default="metrics/main_summary.csv")

    apr = sub.add_parser("robust", help="export robustness experiments")
    apr.add_argument("--rid", default=os.environ.get("RID","main"))
    apr.add_argument("--tags", nargs="+", default=["a","i","u"])
    apr.add_argument("--out_figs",  default="figs")
    apr.add_argument("--out_csv",   default="metrics/robustness.csv")

    args = ap.parse_args()
    if args.cmd == "main":
        run_main_export(args.rid, args.tags, args.out_audio, args.out_figs, args.out_csv)
    else:
        run_robust_export(args.rid, args.tags, args.out_figs, args.out_csv)

if __name__ == "__main__":
    main()
