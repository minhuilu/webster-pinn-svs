#!/usr/bin/env python3
"""
Post-render evaluator (supports explicit paths or checkpoint inference):
- Prefer --Ahat --zeta explicitly provided
- If not provided, infer automatically from the ckps directory:
    Ahat: first look for {ckp_dir}/{tag}_Ahat.npy,fall back to {tag}_A_curve.npy instead
zeta: first look for {ckp_dir}/{tag}_zeta.txt, exp/A_static/ckps/full_eval_all_sA.csv records
          If still missing, robin: 0.06 / neumann:0 / dirichlet:1e6 fallback;robin_fd falls back to a single zeta value, as described below

Usage examples:
  # Minimal usage: provide only tag; other paths are inferred from defaults.
  python tools/post_render_eval.py --tag a

  # Explicitly provide Ahat, zeta, ref, and output directory
  python tools/post_render_eval.py \
    --Ahat exp/A_static/ckps/r0/a/a_Ahat.npy \
    --zeta exp/A_static/ckps/r0/a/a_zeta.txt \
    --ref  data/synthetic/vowels/a/wav/a.wav \
    --bc robin --tag a --out_dir exp/A_static/ckps/r0/a/post

Notes:
- The FDTD implementation uses a single-parameter Robin radiation coefficient zeta. For bc=robin_fd, this script tries to read two values, zeta0 and zeta1.
  Rendering still falls back to one effective zeta, defaulting to zeta0, for lightweight compatibility with training.
"""

import os, sys, pathlib, argparse, json, csv
import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# ------------- Repository path setup -------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp.A_static.audio_losses import multi_stft_loss, lsd_db
from exp.A_static.synthesize_ref import webster_1d_fd
from exp.common.singer import get_profile


# ----------------- Small utilities -----------------
def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True); return p

def hnr_db(x, sr=16000):
    x = torch.tensor(x, dtype=torch.float32).view(-1)
    x = (x - x.mean()) / (x.std().clamp_min(1e-8))
    n = int(1 << (x.numel()-1).bit_length())
    pad = F.pad(x, (0, n - x.numel()))
    X = torch.fft.rfft(pad)
    r = torch.fft.irfft(X * torch.conj(X))
    r = r[:min(int(sr*0.03), r.numel())]
    r0 = r[0].clamp_min(1e-8); rp = r[1:].max()
    return 10.0 * torch.log10((rp / (r0 - rp + 1e-8)).clamp_min(1e-8)).item()

def plot_spec(x, sr, title, fpath):
    try:
        import librosa, librosa.display
        S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) + 1e-8
        S_db = 20*np.log10(S / S.max())
        plt.figure(figsize=(6,3))
        librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis="time", y_axis="linear")
        plt.title(title); plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()
    except Exception:
        plt.specgram(x, NFFT=1024, Fs=sr, noverlap=768, cmap="magma")
        plt.title(title); plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()

def read_last_zeta_from_csv(csv_path: pathlib.Path, rid: str, bc: str, tag: str):
    if not csv_path.exists(): return None, None, None
    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if r.get("run_id")==rid and r.get("bc")==bc and r.get("tag")==tag]
    if not rows: return None, None, None
    z = rows[-1].get("zeta", "")
    try:
        zeta = float(z)
    except Exception:
        zeta = None
    # robin_fd Compatibility: if zeta0/zeta1 are split in the CSV later, try to read them here too
    z0 = rows[-1].get("zeta0", None); z1 = rows[-1].get("zeta1", None)
    zeta0 = float(z0) if (z0 not in (None,"")) else None
    zeta1 = float(z1) if (z1 not in (None,"")) else None
    return zeta, zeta0, zeta1

def parse_zeta_file(path: pathlib.Path):
    txt = path.read_text().strip()
    parts = txt.replace(",", " ").split()
    # Allow "0.08" "0.08 0.01"
    vals = []
    for p in parts:
        try: vals.append(float(p))
        except: pass
    if not vals: return None, None, None
    if len(vals) == 1: return vals[0], None, None
    # >=2: take the first two
    return vals[0], vals[0], vals[1]

def make_Ax_fn(Avec, L):
    """Build Ax_fn(x) by linear interpolation from a uniform grid on 0..L."""
    xp = np.linspace(0.0, float(L), len(Avec))
    def Ax_fn(x_phys):
        return np.interp(x_phys, xp, np.asarray(Avec), left=Avec[0], right=Avec[-1]).astype(np.float32)
    return Ax_fn


# ------------- Main flow -------------
def main():
    ap = argparse.ArgumentParser()
    # Explicit(optional)
    ap.add_argument("--Ahat",  default=None, help="path to *_Ahat.npy (infer automatically if missing)")
    ap.add_argument("--zeta",  default=None, help="path to *_zeta.txt; may be a single value or two values: z0 z1 (robin_fd)")
    ap.add_argument("--ref",   default=None, help="reference audio; defaults to data/synthetic/vowels/{tag}/wav/{tag}.wav")

    # Logical location for automatic inference
    ap.add_argument("--tag",   default="a", choices=["a","i","u"])
    ap.add_argument("--rid",   default=os.environ.get("RID", "r0"))
    ap.add_argument("--bc",    default=os.environ.get("BC", "robin"),
                    choices=["robin","robin_fd","neumann","dirichlet"])
    ap.add_argument("--ckp_dir", default=None,
                    help="default=exp/A_static/ckps/{rid}/{tag}")

    # Other parameters
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--c",  type=float, default=343.0)
    ap.add_argument("--L",  type=float, default=None, help="tube length in meters; default from profile")
    ap.add_argument("--f0", type=float, default=None, help="if omitted, try to read f0/{tag}.npy; otherwise use profile default")
    ap.add_argument("--Oq", type=float, default=0.6)
    ap.add_argument("--Cq", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=10.0)

    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--cfl", type=float, default=0.8, help="CFL number for stability bound")
    ap.add_argument("--noise_db", type=float, default=-28.0, help="aspiration noise level in dB")
    ap.add_argument("--probe", action="store_true",
                help="also estimate formants (F1/F2/F3) and write MAE (needs exp.B_probes.formant_probe)")
    args = ap.parse_args()

    # ---- Resolve directories and default paths ----
    tag = args.tag
    rid = args.rid
    bc  = args.bc.lower()
    sr  = args.sr

    ckp_dir = pathlib.Path(args.ckp_dir) if args.ckp_dir else pathlib.Path(f"exp/A_static/ckps/{rid}/{tag}")
    csv_global = pathlib.Path("exp/A_static/ckps") / "full_eval_all_sA.csv"

    # ---- Reference audio ----
    if args.ref is None:
        ref_path = pathlib.Path(f"data/synthetic/vowels/{tag}/wav/{tag}.wav")
    else:
        ref_path = pathlib.Path(args.ref)
    assert ref_path.exists(), f"[ERR] ref not found: {ref_path}"
    y_ref, sr_ = sf.read(ref_path)
    assert y_ref.ndim == 1, "ref must be mono"
    assert sr_ == sr, f"SR mismatch: {sr_} vs {sr}"

    # ---- L f0 ----
    prof = get_profile()
    L = float(args.L if args.L is not None else prof["L"])
    if args.f0 is not None:
        f0 = float(args.f0)
    else:
        f0_cand = ref_path.parent.parent / "f0" / f"{tag}.npy"
        if f0_cand.exists():
            f0_arr = np.load(f0_cand)
            f0 = float(np.median(f0_arr))
        else:
            f0 = float(prof["f0_by_vowel"].get(tag, 140.0))

    # ---- Ahat Resolve:Explicit -> Ahat.npy -> A_curve.npy ----
    if args.Ahat is not None:
        A_path = pathlib.Path(args.Ahat)
        assert A_path.exists(), f"[ERR] Ahat not found: {A_path}"
    else:
        p1 = ckp_dir / f"{tag}_Ahat.npy"
        p2 = ckp_dir / f"{tag}_A_curve.npy"
        if p1.exists():
            A_path = p1
            print(f"[A] use Ahat: {p1}")
        elif p2.exists():
            A_path = p2
            print(f"[A] use A_curve as Ahat: {p2}")
        else:
            raise FileNotFoundError(f"Cannot find Ahat: neither {p1} nor {p2}")

    Ahat = np.load(A_path).astype(np.float32).squeeze()

    # ---- zeta Resolve:Explicit -> local txt -> CSV -> default ----
    zeta, zeta0, zeta1 = None, None, None
    if args.zeta is not None:
        z_path = pathlib.Path(args.zeta)
        assert z_path.exists(), f"[ERR] zeta file not found: {z_path}"
        zeta, zeta0, zeta1 = parse_zeta_file(z_path)
        print(f"[zeta] from file {z_path}: z={zeta} (z0={zeta0}, z1={zeta1})")
    else:
        z_txt = ckp_dir / f"{tag}_zeta.txt"
        if z_txt.exists():
            zeta, zeta0, zeta1 = parse_zeta_file(z_txt)
            print(f"[zeta] from local {z_txt}: z={zeta} (z0={zeta0}, z1={zeta1})")
        else:
            z_csv, z0_csv, z1_csv = read_last_zeta_from_csv(csv_global, rid, bc, tag)
            zeta  = z_csv  if z_csv  is not None else zeta
            zeta0 = z0_csv if z0_csv is not None else zeta0
            zeta1 = z1_csv if z1_csv is not None else zeta1
            if zeta is not None or zeta0 is not None:
                print(f"[zeta] from CSV {csv_global.name}: z={zeta} (z0={zeta0}, z1={zeta1})")

    # ---- zeta fallback(by bc)----
    if bc == "neumann":
        z_eff = 0.0
    elif bc == "dirichlet":
        z_eff = 1e6
    elif bc == "robin_fd":
        # FDTD zeta,( zeta0, zeta,default0.06)
        z_eff = (zeta0 if zeta0 is not None else (zeta if zeta is not None else 0.06))
        print(f"[bc=robin_fd] fallback single-zeta rendering with z_eff={z_eff:.4f} (uses z0 if available)")
    else:  # robin
        z_eff = (zeta if zeta is not None else 0.06)

    # ---- Render ----
    Ax_fn = make_Ax_fn(Ahat, L)
    dur = len(y_ref) / sr
    y_hat, _ = webster_1d_fd(
        Ax_fn, f0=f0, dur=dur, sr_out=sr, c=args.c, L=L,
        Nx=args.Nx, cfl_max=args.cfl,
        zeta_ref=z_eff, Oq=args.Oq, Cq=args.Cq, beta=args.beta,
        noise_db=args.noise_db
    )

    # Normalize
    y_hat = y_hat / (np.std(y_hat) + 1e-8)
    y_ref = y_ref / (np.std(y_ref) + 1e-8)

    # Metrics
    with torch.no_grad():
        t_hat = torch.tensor(y_hat, dtype=torch.float32)
        t_ref = torch.tensor(y_ref, dtype=torch.float32)
        m = float(multi_stft_loss(t_hat, t_ref).item())
        l = float(lsd_db(t_hat, t_ref).item())
    h_hat = hnr_db(y_hat, sr=sr); h_ref = hnr_db(y_ref, sr=sr)

    # Output directory
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (A_path.resolve().parent / "post")
    ensure_dir(out_dir)

    # Figures and audio
    ref_png = out_dir / "spec_ref.png"
    if not ref_png.exists(): plot_spec(y_ref, sr, f"REF {tag}", ref_png)
    hat_png = out_dir / f"spec_hat_{bc}.png"
    plot_spec(y_hat, sr, f"HAT {tag} ({bc})", hat_png)
    sf.write(str(out_dir / f"hat_{bc}.wav"), y_hat.astype(np.float32), sr)

    F1_MAE = F2_MAE = F3_MAE = ""
    if args.probe:
        from exp.B_probes.formant_probe import probe_all
        import torch.nn.functional as F
        # Use a middle window for probing
        nfft = 2048
        mid0 = max(0, len(y_hat)//2 - nfft//2); mid1 = mid0 + nfft
        yseg = torch.tensor(y_hat[mid0:mid1], dtype=torch.float32)
        rseg = torch.tensor(y_ref[mid0:mid1], dtype=torch.float32)
        # f0 trajectories: f0()
        f0_seg = torch.full_like(yseg, float(f0))
        ph = probe_all(yseg, sr=sr, f0_samples=f0_seg)
        pr = probe_all(rseg, sr=sr, f0_samples=f0_seg)
        mae = (ph["F"] - pr["F"]).abs().mean(dim=1)
        F1_MAE, F2_MAE, F3_MAE = [float(mae[i].cpu().item()) for i in range(3)]


    # CSV
    csv_path = out_dir / "post_render_metrics.csv"
    write_header = (not csv_path.exists())
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "tag","bc","rid","Ahat","ref","mSTFT","LSD_dB","HNR_hat_dB","HNR_ref_dB",
                "L","f0","z_eff","zeta","zeta0","zeta1","Nx","cfl","noise_db",
                "F1_MAE_Hz","F2_MAE_Hz","F3_MAE_Hz"
            ])
        w.writerow([
            tag, bc, rid, str(A_path), str(ref_path), m, l, h_hat, h_ref,
            L, f0, z_eff,
            (zeta  if zeta  is not None else ""),
            (zeta0 if zeta0 is not None else ""),
            (zeta1 if zeta1 is not None else ""),
            args.Nx, args.cfl, args.noise_db,
            F1_MAE, F2_MAE, F3_MAE
        ])

    print(f"[POST] tag={tag} bc={bc}  mSTFT={m:.4f}  LSD={l:.3f}dB  HNR_hat={h_hat:.1f}dB  -> {csv_path}")
    print(f"[OUT] wav={out_dir / ('hat_'+bc+'.wav')}  spec={hat_png}")

if __name__ == "__main__":
    main()
