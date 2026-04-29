#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export A(x) curve plots and data, supporting two sources:
- :--Ahat *_Ahat.npy(default ckps )
- Fallback: load DualNet from a checkpoint and sample A(x) -> save {tag}_A_curve.npy / {tag}_A_curve.png

Usage examples:
  # 1) Easiest mode: find Ahat or checkpoint automatically by RID/tag
  python tools/export_artifacts.py --rid main --tag a

  # 2) Explicit Ahat.npy
  python tools/export_artifacts.py --Ahat exp/A_static/ckps/main/a/a_Ahat.npy

  # 3) Sample from a checkpoint(reads config_sA.json in the same directory or infers Lt from the reference wav)
  python tools/export_artifacts.py --rid main --tag a --from_ckpt

Outputs:
  exp/A_static/ckps/<rid>/<tag>/{tag}_A_curve.png
  exp/A_static/ckps/<rid>/<tag>/{tag}_A_curve.npy
  exp/A_static/ckps/<rid>/<tag>/{tag}_x_grid.npy
"""

import os, sys, json, pathlib, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository path setup
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from exp.A_static.train_with_audio import DualNet
from exp.A_static.phys_consts import Lx
import soundfile as sf

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True); return p

def load_A_from_ckpt(ckp_dir: pathlib.Path, tag: str, sr: int = 16000):
    """
    Load the network from a checkpoint directory and sample A(x):
      - requires {tag}_best_sA.pt
      - Lt prefer config_sA.json; otherwise infer from the reference wav
    """
    ckpt = ckp_dir / f"{tag}_best_sA.pt"
    assert ckpt.exists(), f"[ERR] ckpt not found: {ckpt}"

    # Lt: first check config_sA.json
    cfgj = ckp_dir / "config_sA.json"
    Lt = None
    if cfgj.exists():
        try:
            Lt = float(json.loads(cfgj.read_text())["Lt"])
        except Exception:
            Lt = None
    if Lt is None:
        # fallback: find reference wav
        ref_wav = ROOT / f"data/synthetic/vowels/{tag}/wav/{tag}.wav"
        assert ref_wav.exists(), f"[ERR] cannot infer Lt: {cfgj} missing & {ref_wav} not found"
        y, sr_ = sf.read(str(ref_wav)); assert sr_ == sr, f"SR mismatch {sr_} vs {sr}"
        Lt = len(y) / sr

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DualNet(Lx=float(Lx), Lt=float(Lt), use_ff=False).to(dev)

    # checkpoint may contain extra keys such as p_gain, t_shift_raw, and zeta; strict=False ignores extra keys
    state = torch.load(str(ckpt), map_location=dev)
    net.load_state_dict(state, strict=False)
    net.eval()

    xs = torch.linspace(0, float(Lx), 512, device=dev).view(-1,1)
    with torch.no_grad():
        A = net.A_from_x(xs).view(-1).detach().cpu().numpy().astype(np.float32)
    xnp = xs.detach().cpu().numpy().astype(np.float32)
    return xnp, A

def plot_A_curve(x, A, title, out_png):
    plt.figure(figsize=(5,3))
    plt.plot(x, A)
    plt.xlabel("x (m)"); plt.ylabel("Area A(x) (a.u.)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rid", default=os.environ.get("RID","r0"))
    ap.add_argument("--tag", default="a", choices=["a","i","u"])
    ap.add_argument("--Ahat", default=None, help="path to *_Ahat.npy; use it directly for plotting if provided")
    ap.add_argument("--from_ckpt", action="store_true", help="ignore Ahat and sample A(x) from a checkpoint")
    ap.add_argument("--ckp_dir", default=None, help="default=exp/A_static/ckps/{rid}/{tag}")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    tag = args.tag
    rid = args.rid
    ckp_dir = pathlib.Path(args.ckp_dir) if args.ckp_dir else pathlib.Path(f"exp/A_static/ckps/{rid}/{tag}")
    ensure_dir(ckp_dir)

    # - Choose the source of A - #
    A_vec = None
    x_grid = None
    title  = f"A(x) - {tag} [{rid}]"

    if (args.Ahat is not None) and (not args.from_ckpt):
        A_path = pathlib.Path(args.Ahat)
        assert A_path.exists(), f"[ERR] Ahat not found: {A_path}"
        A_vec = np.load(A_path).astype(np.float32).squeeze()
        # x Use [0,Lx] uniform 512-point grid for plotting only; data are not changed
        x_grid = np.linspace(0, float(Lx), 512, dtype=np.float32)
        print(f"[A] from Ahat: {A_path}")

    if (A_vec is None) and (not args.from_ckpt):
        # Try to find Ahat in the ckps directory
        p1 = ckp_dir / f"{tag}_Ahat.npy"
        p2 = ckp_dir / f"{tag}_A_curve.npy"  # Allowsave "sampled curve"
        if p1.exists():
            A_vec = np.load(p1).astype(np.float32).squeeze()
            x_grid = np.linspace(0, float(Lx), 512, dtype=np.float32)
            print(f"[A] from {p1.name}")
        elif p2.exists():
            # This is usually the same format we write and may have a paired x_grid
            A_vec = np.load(p2).astype(np.float32).squeeze()
            xg = ckp_dir / f"{tag}_x_grid.npy"
            if xg.exists():
                x_grid = np.load(xg).astype(np.float32).squeeze()
                print(f"[A] from {p2.name} + {xg.name}")
            else:
                x_grid = np.linspace(0, float(Lx), len(A_vec), dtype=np.float32)
                print(f"[A] from {p2.name} (no x_grid, assume uniform)")
        else:
            print("[A] no local Ahat found; will sample from ckpt…")

    if (A_vec is None) or args.from_ckpt:
        x_grid, A_vec = load_A_from_ckpt(ckp_dir, tag, sr=args.sr)
        print(f"[A] sampled from ckpt: {ckp_dir / (tag+'_best_sA.pt')}")

    # - Save figure and data - #
    out_png = ckp_dir / f"{tag}_A_curve.png"
    out_npy = ckp_dir / f"{tag}_A_curve.npy"
    out_xnp = ckp_dir / f"{tag}_x_grid.npy"

    plot_A_curve(x_grid, A_vec, title, out_png)
    np.save(out_npy, A_vec.astype(np.float32))
    np.save(out_xnp, x_grid.astype(np.float32))

    print(f"[OK] wrote: {out_png}")
    print(f"[OK] data : {out_npy}, {out_xnp}")

if __name__ == "__main__":
    main()
