# tools/plot_robust_figs.py
import numpy as np, matplotlib.pyplot as plt, librosa, librosa.display, soundfile as sf, pathlib

ROOT = pathlib.Path(".")
TAGS = ["a","i","u"]
# Adjust these names to match the actual out_dir
SCENES = {
    "post":      "post",
    "grid_lo":   "post_mis_grid",
    "grid_hi":   "post_mis_grid_hi",
    "f0m":       "post_f0_90p",
    "f0p":       "post_f0_110p",
}
def ref_wav(tag): return ROOT / f"data/synthetic/vowels/{tag}/wav/{tag}.wav"
def hat_wav(tag, scene): return ROOT / f"exp/A_static/ckps/main/{tag}/{SCENES[scene]}/hat_robin.wav"

def spec(ax, y, sr, title):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    S_db = 20*np.log10(S/ (S.max()+1e-12) + 1e-12)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='linear', ax=ax)
    ax.set_title(title); return img

def mean_envelope(y, sr):
    # Simple average envelope: take the median over time in the STFT magnitude
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256)) + 1e-9
    med = np.median(S, axis=1)
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    med_db = 20*np.log10(med/med.max())
    return f, med_db

def plot_one_tag(tag):
    sr = 16000
    y_ref, _ = sf.read(str(ref_wav(tag)))

    # ---- (Fig A) Grid mismatch ----
    fig, axes = plt.subplots(1, 3, figsize=(12,3.2))
    spec(axes[0], y_ref, sr, f"{tag}: REF")
    for k,sc in enumerate(["post","grid_lo","grid_hi"], start=1):
        y_hat, _ = sf.read(str(hat_wav(tag, sc)))
        f, e_ref = mean_envelope(y_ref, sr); _, e_hat = mean_envelope(y_hat, sr)
        axes[k-1].plot([],[])  # no-op

    # Envelope overlays are more readable here
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.8,3.2))
    f_ref, e_ref = mean_envelope(y_ref, sr); ax.plot(f_ref, e_ref, 'k--', label="REF")  # Keep the reference line dashed
    for sc, lab, ls in [("post","Post","-."), ("grid_lo","Grid↓",":"), ("grid_hi","Grid↑","--")]:
        y_hat, _ = sf.read(str(hat_wav(tag, sc)))
        f, e = mean_envelope(y_hat, sr); ax.plot(f, e, label=lab, linestyle=ls)  # Add line-style parameter
    ax.set_xlim(0, 5000); ax.set_ylim(-60, 0); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
    ax.set_title(f"{tag}: envelope under grid mismatch"); ax.legend()
    out = ROOT / f"exp/A_static/figs/robust"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{tag}_env_grid.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- (Fig B) f0 +/-10% ----
    fig, ax = plt.subplots(figsize=(5.8,3.2))
    ax.plot(f_ref, e_ref, 'k--', label="REF")  # Keep the reference line dashed
    for sc, lab, ls in [("post","Post","-."), ("f0m","f0-10%",":"), ("f0p","f0+10%","--")]:
        y_hat, _ = sf.read(str(hat_wav(tag, sc)))
        f, e = mean_envelope(y_hat, sr); ax.plot(f, e, label=lab, linestyle=ls)  # Add line-style parameter
    ax.set_xlim(0, 5000); ax.set_ylim(-60, 0); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
    ax.set_title(f"{tag}: envelope under $f_0$ perturbation"); ax.legend()
    fig.savefig(out / f"{tag}_env_f0.png", dpi=200, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    for t in TAGS: plot_one_tag(t)
    print("[ok] figs -> exp/A_static/figs/robust/")