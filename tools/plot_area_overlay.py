# tools/plot_area_overlay.py
import numpy as np, matplotlib.pyplot as plt, pathlib, sys
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
from exp.A_static.synthesize_ref import area_a, area_i, area_u
from exp.common.singer import get_profile

AREA_FNS = {"a": area_a, "i": area_i, "u": area_u}
SPEAKER_L = float(get_profile()["L"])  # ~0.16 m

def load_A_pair(rid, tag):
    """Return (x_norm, Ahat).supports three cases:
1) Ahat.npy (N,) -> x=linspace(0,1,N)
2) Ahat.npy (N,2) -> first column is x, either normalized or physical
       3) separate x_grid.npy     -> use it for x
    """
    ckp_dir = pathlib.Path(f"exp/A_static/ckps/{rid}/{tag}")
    p_ahat = ckp_dir / f"{tag}_Ahat.npy"
    assert p_ahat.exists(), f"missing {p_ahat}"
    A = np.load(p_ahat)

    # Prefer a separate x_grid.npy
    p_x = ckp_dir / f"{tag}_x_grid.npy"
    if p_x.exists():
        x = np.load(p_x).astype(np.float32).reshape(-1)
        if x.max() > 0.5:  # physical coordinates()
            x = x / float(SPEAKER_L)
        # A may be one-dimensional or (N,2)
        if A.ndim == 2 and A.shape[1] == 2:
            A = A[:,1]
        return x, A.astype(np.float32).reshape(-1)

    # No x_grid.npy: A
    if A.ndim == 1:  # area values only
        N = A.shape[0]
        x = np.linspace(0, 1.0, N, dtype=np.float32)
        return x, A.astype(np.float32)
    if A.ndim == 2 and A.shape[1] == 2:  # (x, A)
        x = A[:,0].astype(np.float32); A = A[:,1].astype(np.float32)
        if x.max() > 0.5:  # physical coordinates -> Normalize
            x = x / float(SPEAKER_L)
        return x, A

    raise ValueError(f"Unrecognized Ahat shape: {A.shape}")

def plot_three(rid="main", out="exp/A_static/figs/area_overlay_all.png"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, tag in zip(axes, ["a","i","u"]):
        xs = np.linspace(0, 1.0, 1000, dtype=np.float32)
        A_true = AREA_FNS[tag](xs)
        ax.plot(xs, A_true, "k--", lw=2, label="A_true (synth)")

        xh, Ah = load_A_pair(rid, tag)
        ax.plot(xh, Ah, lw=2, label="learned (main)")

        ax.set_title(f"/{tag}/", fontsize=14, pad=6)
        ax.set_xlabel("x (normalized)")
        ax.grid(alpha=0.25)
        if tag == "a":
            ax.set_ylabel("Area A(x) (a.u.)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    outp = pathlib.Path(out); outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=220, bbox_inches="tight")
    print(f"[ok] -> {outp}")

if __name__ == "__main__":
    plot_three("main")
