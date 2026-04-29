import os
import sys
import pathlib

# ---- package path setup ----
if __name__ == "__main__" and not __package__:
    current_file = pathlib.Path(__file__).resolve()
    repo_root = current_file.parent.parent.parent  # exp/A_static -> exp -> repo
    sys.path.insert(0, str(repo_root))
    __package__ = "exp.A_static"

import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project-local modules
from .train_with_audio import FourierFeatures1D, train_one_epoch, DualNet
from .audio_losses import MelBank, multi_stft_loss, lsd_db, logmel_envelope_loss
from .audio_forward import eval_p_lip_series_full, _get_tshift, pick_window
from .phys_consts import Lx, c
from exp.B_probes.formant_probe import probe_all
from exp.C_ddsp.ddsp_synth import HEnvMapper, synth_harmonic
from exp.C_ddsp.ddsp_baselines import HEnvPredictor
from exp.common.singer import get_profile, band_scale_for_L
from exp.A_static.synthesize_ref import webster_1d_fd 

# ============ Global configuration ============
PROFILE = get_profile()
SPEAKER_L = PROFILE["L"]
BAND_SCALE = band_scale_for_L(SPEAKER_L)
print(f"[PROFILE] name={os.environ.get('SPEAKER_PROFILE','female').lower()}, L={SPEAKER_L}, band_scale={BAND_SCALE:.3f}")

# save:best/final/none/every_eval
PLOT_POLICY = os.environ.get("PLOT", "best").lower()

# ============ Environment-variable switches for reproducibility and quick trials ============
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

# Training epochs and tag selection
EPOCHS_ENV = _env_int("EPOCHS", 4000)
TAGS_ENV = os.environ.get("TAGS", "") or os.environ.get("ONLY_TAG", "")
if TAGS_ENV:
    TAG_LIST = [t.strip().lower() for t in TAGS_ENV.split(",") if t.strip()]
    if not TAG_LIST:
        TAG_LIST = ["a", "i", "u"]
else:
    TAG_LIST = ["a", "i", "u"]

# ramp-inweight
AUDIO_RAMP_START = _env_int("AUDIO_RAMP_START", 50)
AUDIO_RAMP_END   = _env_int("AUDIO_RAMP_END",   650)  # original setting 50 -> 650
AUDIO_W_MSTFT_MAX = _env_float("AUDIO_W_MSTFT_MAX", 0.2)
AUDIO_W_ENV_MAX   = _env_float("AUDIO_W_ENV_MAX",   0.1)

# DDSP start epoch; a very large value effectively disables it, or use MODE=no_ddsp
DDSP_START_EPOCH = _env_int("DDSP_START", 400)
MODE_NO_DDSP_ENV = os.environ.get("NO_DDSP", "0").strip() in ("1", "true", "True")
DISABLE_GLOBAL = os.environ.get("DISABLE_GLOBAL", "0").strip().lower() in ("1","true","yes")

# regularization
RAD_START_EPOCH = _env_int("RAD_START", 50)
RAD_RAMP_EPOCHS = _env_int("RAD_RAMP", 400)
RAD_MAX         = _env_float("RAD_MAX", 0.15)

# A'' regularization(default)
W_AXX_OVERRIDE = os.environ.get("W_AXX", "").strip()

# source
W_SOURCE_ENV = _env_float("W_SOURCE", 0.02)
SOURCE_DECAY_EPOCHS = _env_int("SOURCE_DECAY", 800)
# glottalgain( U_g):default c; FDTD,for rho*c~=1.2*343~=411.6
GLOT_GAIN = _env_float("GLOT_GAIN", float(c))
GLOT_GAIN0 = _env_float("GLOT_GAIN0", 60.0)
GLOT_GAIN_RAMP = _env_int("GLOT_GAIN_RAMP", 300)
W_GLOT = _env_float("W_GLOT", 0.02)

# cold start 1:fixwindow epoch(helps establish initial sound)
FIXED_WIN_EPOCHS = _env_int("FIXED_WIN_EPOCHS", 0)
FIXED_WIN_CENTER_FRAC = _env_float("FIXED_WIN_CENTER_FRAC", 0.5)

# cold start 2:disableregularization epoch
RAD_FREEZE_EPOCHS = _env_int("RAD_FREEZE_EPOCHS", 0)

# periodicity(encourage sinusoidal structure synchronized with the reference f0)
W_PERIOD = _env_float("W_PERIOD", 0.0)
PERIOD_RAMP = _env_int("PERIOD_RAMP", 0)

# zeta regularization(Robin regularization)
W_ZETA_REG = _env_float("W_ZETA_REG", 1.0)

# zeta learning rate
RAD_ZETA_LR_ENV = os.environ.get("RAD_ZETA_LR", "").strip()
P_GAIN_LR_ENV = os.environ.get("P_GAIN_LR", "").strip()
T_SHIFT_LR_ENV = os.environ.get("T_SHIFT_LR", "").strip()

# audio warmup head(t->p auxiliary)mixing-ratio schedule
AUX_GAMMA0 = _env_float("AUX_GAMMA0", 0.0)
AUX_GAMMA_RAMP = _env_int("AUX_GAMMA_RAMP", 0)
AUX_LR_ENV = os.environ.get("AUX_LR", "").strip()
AUX_USE_TEACHER = os.environ.get("AUX_USE_TEACHER", "0").strip().lower() in ("1","true","yes")
# Option B:auxiliary
AUX_USE_HARM = os.environ.get("AUX_USE_HARM", "0").strip().lower() in ("1","true","yes")
HARM_K = _env_int("HARM_K", 24)

# teacher()
W_TEACHER = _env_float("W_TEACHER", 0.0)
TEACHER_K = _env_int("TEACHER_K", 12)
TEACHER_TILT = _env_float("TEACHER_TILT", 1.2)

# ============ Vowel presets ============
PRESETS = {
    "a": dict(probe=dict(n_fft=1024, hop=256, temp=6.0, K=22, sigma_scale=0.16, band_scale=BAND_SCALE),
              form_sched=dict(start=900, end=1800, to=0.45)),
    "i": dict(probe=dict(n_fft=4096, hop=256, temp=6.0, K=26, sigma_scale=0.14, band_scale=BAND_SCALE),
              form_sched=dict(start=1000, end=2000, to=0.35)),
    "u": dict(probe=dict(n_fft=2048, hop=256, temp=6.5, K=18, sigma_scale=0.20, band_scale=BAND_SCALE),
              form_sched=dict(start=1200, end=2400, to=0.35)),
}

# ============ Experiment mode and run_id ============
RUN_MODE = os.environ.get("MODE", "main")
RUN_ID   = os.environ.get("RID",  "r0")
BC_TYPE  = os.environ.get("BC", "robin").lower()
print(f"[BC] Using boundary = {BC_TYPE}   [MODE={RUN_MODE}, RID={RUN_ID}]")

# ============ Training hyperparameter safeguards ============
EVAL_PERIOD = 200
PATIENCE_EVALS = 6
MIN_DELTA = 1e-3
LR_MIN = 1e-6
LR_FACTOR = 0.5
LR_SCHED_PATIENCE_EVALS = 3

FSMOOTH_EMA_BETA = 0.95
FSMOOTH_HI = 4.0
FSMOOTH_LO = 3.0
FORMANT_DOWNSCALE = 0.4

F2_IMPROVE_EPS = 0.5
sA_CSV_NAME = "full_eval_all_sA.csv"

# ============ Small utilities ============
def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True); return p

def hnr_db(x, sr=16000):
    x = x.view(-1)
    x = (x - x.mean()) / (x.std().clamp_min(1e-8))
    n = int(1 << (x.numel()-1).bit_length())
    pad = F.pad(x, (0, n - x.numel()))
    X = torch.fft.rfft(pad)
    r = torch.fft.irfft(X * torch.conj(X))
    r = r[:min(int(sr*0.03), r.numel())]
    r0 = r[0].clamp_min(1e-8)
    rp = r[1:].max()
    return 10.0 * torch.log10((rp / (r0 - rp + 1e-8)).clamp_min(1e-8)).item()

def align_by_xcorr(x_hat, x_ref, sr=16000, hop=256, ignore_ms=300, max_shift_sec=0.30):
    ig = int(ignore_ms * 1e-3 * sr)
    m  = min(len(x_hat), len(x_ref))
    a, b = ig, max(ig, m - ig)
    xh0 = x_hat[a:b].view(-1)
    xr0 = x_ref[a:b].view(-1)
    n = int(1 << (max(len(xh0), len(xr0)).bit_length()))
    def _norm(v):
        v = v - v.mean(); s = v.std().clamp_min(1e-8); return v / s
    xh = torch.zeros(n*2, device=x_hat.device); xr = torch.zeros(n*2, device=x_ref.device)
    xh[:len(xh0)] = _norm(xh0); xr[:len(xr0)] = _norm(xr0)
    XH = torch.fft.rfft(xh); XR = torch.fft.rfft(xr)
    r  = torch.fft.irfft(XH * torch.conj(XR), n=len(xh))
    r = torch.roll(r, shifts=len(xh0)-1, dims=0)

    # correlation peak(rough unnormalized strength, sufficient for relative reliability checks)
    rho_peak = float(r.max().detach().cpu().item())

    k = int(torch.argmax(r).item())
    shift = k - (len(xh0)-1)
    max_shift = int(max_shift_sec * sr)
    hit_limit = False
    if shift >  max_shift: shift =  max_shift; hit_limit = True
    if shift < -max_shift: shift = -max_shift; hit_limit = True

    if shift > 0:
        xh_aln = x_hat[shift:]; xr_aln = x_ref[:len(xh_aln)]
    elif shift < 0:
        xr_aln = x_ref[-shift:]; xh_aln = x_hat[:len(xr_aln)]
    else:
        xh_aln, xr_aln = x_hat, x_ref

    L = min(len(xh_aln), len(xr_aln))
    if L <= 0:
        L = min(len(x_hat), len(x_ref)); xh_aln = x_hat[:L]; xr_aln = x_ref[:L]; shift = 0; hit_limit = True

    return xh_aln, xr_aln, shift, rho_peak, hit_limit


def global_mstft_scale(epoch: int) -> float:
    if epoch < 300:   return 0.0
    if epoch < 900:   return 0.5 * (epoch - 300) / 600
    if epoch < 1600:  return 0.5 + 0.2 * (epoch - 900) / 700
    return 0.7

def ramp(val0, val1, ep, s, e):
    if ep <= s: return val0
    if ep >= e: return val1
    a = (ep - s) / float(e - s); return val0 + (val1 - val0) * a

# ------------------ plotting( CSV ) ------------------

def _read_csv_rows(path: pathlib.Path):
    if (not path.exists()) or (path.stat().st_size == 0):
        return []
    rows = []
    with open(path, "r") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows

def _to_float_list(rows, key, filt=None):
    out = []
    for r in rows:
        if (filt is None) or filt(r):
            v = r.get(key, "")
            try:
                out.append(float(v))
            except Exception:
                # bool/int strings
                if str(v).strip().lower() in ("true","false"):
                    out.append(1.0 if str(v).strip().lower()=="true" else 0.0)
                else:
                    out.append(np.nan)
    return out

def _plot_train_curves(fig_path: pathlib.Path, train_rows):
    if not train_rows:
        return
    eps = _to_float_list(train_rows, "epoch")
    # atomic loss
    L = _to_float_list(train_rows, "L")
    L_mstft = _to_float_list(train_rows, "L_mstft")
    L_env = _to_float_list(train_rows, "L_env")
    L_full = _to_float_list(train_rows, "L_mstft_full")
    L_pde = _to_float_list(train_rows, "L_pde")
    L_rad = _to_float_list(train_rows, "L_rad")
    L_glot = _to_float_list(train_rows, "L_glot")
    L_smh = _to_float_list(train_rows, "L_smh")
    L_geom= _to_float_list(train_rows, "L_geom")
    L_Aend= _to_float_list(train_rows, "L_Aend")
    L_form= _to_float_list(train_rows, "L_form")
    L_henv= _to_float_list(train_rows, "L_henv")
    L_fsmo= _to_float_list(train_rows, "L_form_smooth")
    L_time= _to_float_list(train_rows, "L_time")
    L_amp = _to_float_list(train_rows, "L_amp")
    L_tv = _to_float_list(train_rows, "L_logA_TV")

    # scalarlearning rate
    p_gain = _to_float_list(train_rows, "p_gain")
    tau_ms = [1e3*x for x in _to_float_list(train_rows, "tau_sec")]
    zeta   = _to_float_list(train_rows, "zeta")
    lr_main= _to_float_list(train_rows, "lr_main")
    lr_p   = _to_float_list(train_rows, "lr_p_gain")
    lr_tau = _to_float_list(train_rows, "lr_tau")
    lr_zeta= _to_float_list(train_rows, "lr_zeta")
    lr_ddsp= _to_float_list(train_rows, "lr_ddsp_mapper")

    plt.figure(figsize=(14,10))

    ax = plt.subplot(2,3,1)
    ax.semilogy(eps, np.maximum(L,1e-9), label="L")
    ax.semilogy(eps, np.maximum(L_mstft,1e-9), label="L_mstft")
    ax.semilogy(eps, np.maximum(L_env,1e-9), label="L_env")
    ax.semilogy(eps, np.maximum(L_full,1e-9), label="L_mstft_full")
    ax.set_title("Audio losses"); ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    ax = plt.subplot(2,3,2)
    ax.semilogy(eps, np.maximum(L_pde,1e-12), label="L_pde")
    ax.semilogy(eps, np.maximum(L_rad,1e-12), label="L_rad")
    ax.semilogy(eps, np.maximum(L_glot,1e-12), label="L_glot")
    ax.semilogy(eps, np.maximum(L_smh,1e-12), label="L_smh")
    ax.set_title("PDE/BC losses"); ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    ax = plt.subplot(2,3,3)
    ax.semilogy(eps, np.maximum(L_geom,1e-12), label="L_geom")
    ax.semilogy(eps, np.maximum(L_Aend,1e-12), label="L_Aend")
    ax.semilogy(eps, np.maximum(L_time,1e-12), label="L_time")
    L_period = _to_float_list(train_rows, "L_period")
    L_teacher = _to_float_list(train_rows, "L_teacher")
    if any(np.isfinite(L_period)):
        ax.semilogy(eps, np.maximum(L_period,1e-12), label="L_period")
    if any(np.isfinite(L_teacher)):
        ax.semilogy(eps, np.maximum(L_teacher,1e-12), label="L_teacher")
    ax.semilogy(eps, np.maximum(L_amp,1e-12), label="L_amp")
    ax.semilogy(eps, np.maximum(L_tv,1e-12), label="L_logA_TV")
    ax.set_title("Geometry/regularizers"); ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    ax = plt.subplot(2,3,4)
    ax.semilogy(eps, np.maximum(L_form,1e-12), label="L_form")
    ax.semilogy(eps, np.maximum(L_henv,1e-12), label="L_henv")
    ax.semilogy(eps, np.maximum(L_fsmo,1e-12), label="L_form_smooth")
    ax.set_title("Probe losses"); ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    ax = plt.subplot(2,3,5)
    ax.plot(eps, p_gain, label="p_gain")
    ax.plot(eps, tau_ms, label="tau_ms")
    ax.plot(eps, zeta,   label="zeta")
    ax.set_title("Learnable scalars"); ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    ax = plt.subplot(2,3,6)
    ax.semilogy(eps, np.maximum(lr_main,1e-12), label="lr_main")
    ax.semilogy(eps, np.maximum(lr_p,1e-12),    label="lr_p_gain")
    ax.semilogy(eps, np.maximum(lr_tau,1e-12),  label="lr_tau")
    ax.semilogy(eps, np.maximum(lr_zeta,1e-12), label="lr_zeta")
    ax.semilogy(eps, np.maximum(lr_ddsp,1e-12), label="lr_ddsp")
    ax.set_title("Learning rates"); ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_eval_curves(fig_path: pathlib.Path, eval_rows, run_id: str, tag: str):
    if not eval_rows:
        return
    rows = [r for r in eval_rows if (r.get("run_id")==run_id and r.get("tag")==tag)]
    if not rows:
        return
    eps = _to_float_list(rows, "epoch")
    m_raw  = _to_float_list(rows, "mSTFT_raw")
    m_aln  = _to_float_list(rows, "mSTFT_aln")
    l_rawdB= _to_float_list(rows, "LSD_raw_dB")
    l_alndB= _to_float_list(rows, "LSD_aln_dB")
    shift  = _to_float_list(rows, "shift_samples")
    rho    = _to_float_list(rows, "rho_peak")
    hit    = _to_float_list(rows, "aln_hit_limit")

    plt.figure(figsize=(12,8))

    ax = plt.subplot(2,2,1)
    ax.semilogy(eps, np.maximum(m_raw,1e-9), label="mSTFT_raw")
    ax.semilogy(eps, np.maximum(m_aln,1e-9), label="mSTFT_aln")
    ax.set_title("Eval mSTFT"); ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    ax = plt.subplot(2,2,2)
    ax.plot(eps, l_rawdB, label="LSD_raw_dB")
    ax.plot(eps, l_alndB, label="LSD_aln_dB")
    ax.set_title("Eval LSD (dB)"); ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    ax = plt.subplot(2,2,3)
    ax.plot(eps, shift, label="shift_samples")
    ax.set_title("XCorr shift (samples)"); ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    ax = plt.subplot(2,2,4)
    ax.plot(eps, rho, label="xcorr_peak")
    ax.plot(eps, hit, label="hit_limit")
    ax.set_title("XCorr diagnostics"); ax.legend(); ax.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def update_pngs(figs_dir: pathlib.Path, tag: str, run_id: str, train_csv: pathlib.Path, eval_csv: pathlib.Path):
    # Read each CSV and plot
    train_rows = _read_csv_rows(train_csv)
    _plot_train_curves(figs_dir / f"train_curves_{tag}.png", train_rows)

    eval_rows  = _read_csv_rows(eval_csv)
    _plot_eval_curves(figs_dir / f"eval_curves_{tag}.png", eval_rows, run_id, tag)


def plot_spec_tensor(x, sr, title, fpath):
    try:
        import librosa, librosa.display
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) + 1e-8
        S_db = 20*np.log10(S / S.max())
        plt.figure(figsize=(6,3))
        librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis="time", y_axis="linear")
        plt.title(title); plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()
    except Exception:
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        plt.specgram(x, NFFT=1024, Fs=sr, noverlap=768, cmap="magma")
        plt.title(title); plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()

def plot_formant_and_env(p_win, y_win, f0_win, sr, out_dir, tag, ep, probe_kwargs=None):
    if probe_kwargs is None: probe_kwargs = {}
    with torch.no_grad():
        pr = probe_all(y_win, sr=sr, f0_samples=f0_win, **probe_kwargs)
    ph = probe_all(p_win, sr=sr, f0_samples=f0_win, **probe_kwargs)
    plt.figure(figsize=(6,4))
    for i, name in enumerate(["F1","F2","F3"]):
        plt.plot(pr["F"][i].cpu(), '--', label=f'{name} ref')
        plt.plot(ph["F"][i].cpu(),      label=f'{name} hat')
    plt.legend(); plt.title(f'Formants {tag} @ep{ep}')
    plt.tight_layout(); plt.savefig(out_dir / f'formants_{tag}_ep{ep}.png', dpi=150); plt.close()
    plt.figure(figsize=(6,3))
    plt.plot(pr["H_env"].cpu(), '--', label='ref')
    plt.plot(ph["H_env"].cpu(),      label='hat')
    plt.title(f'Harmonic envelope {tag} @ep{ep}')
    plt.tight_layout(); plt.savefig(out_dir / f'henv_{tag}_ep{ep}.png', dpi=150); plt.close()

def get_wF(tag: str, ep: int, device: torch.device):
    base = torch.tensor([1.0, 1.0, 1.0], device=device).view(3,1)
    if tag == "i":
        target = torch.tensor([0.6, 1.3, 1.1], device=device).view(3,1)
        if ep <= 1000: a = 0.0
        elif ep >= 1600: a = 1.0
        else: a = (ep - 1000) / 600.0
        return (1.0 - a) * base + a * target
    return base

# ============ save(Normalize) ============
def _save_wav_norm(path, x, sr):
    import soundfile as sf
    if torch.is_tensor(x):
        x = (x / (x.std().clamp_min(1e-6))).detach().cpu().numpy()
    x = x.astype("float32")
    sf.write(path, x, sr)

def _make_Ax_fn(Avec, L):
    """Build Ax(x) interpolation from a uniform-grid area vector."""
    xp = np.linspace(0.0, float(L), len(Avec))
    def Ax_fn(x_phys):
        return np.interp(x_phys, xp, np.asarray(Avec),
                         left=Avec[0], right=Avec[-1]).astype(np.float32)
    return Ax_fn

def _zeta_eff_from_net(net, bc_type: str) -> float:
    """Convert the learned radiation parameter to the single zeta used by FDTD."""
    bc = bc_type.lower()
    dev = next(net.parameters()).device
    if bc == "neumann":
        return 0.0
    if bc == "dirichlet":
        return 1e6
    if bc == "robin_fd":
        z0_raw = getattr(net, "rad_zeta0_raw", torch.tensor(-2.0, device=dev))
        z0 = torch.nn.functional.softplus(z0_raw)
        return float(z0.detach().cpu().item())
    # default robin
    z_raw = getattr(net, "rad_zeta_raw", torch.tensor(-2.0, device=dev))
    z = torch.nn.functional.softplus(z_raw)
    return float(z.detach().cpu().item())


# ============ main function:single vowel ============
def run_one_vowel(tag: str, epochs=4000, sr=16000, seed=1234):
    torch.manual_seed(seed); np.random.seed(seed)

    wav_path = f"data/synthetic/vowels/{tag}/wav/{tag}.wav"
    wav, sr_ = sf.read(wav_path); assert sr_ == sr, f"SR mismatch: {sr_} vs {sr}"
    t_grid = np.linspace(0, len(wav)/sr, len(wav), endpoint=False).astype(np.float32)
    Lt_local = float(len(wav) / sr)
    assert abs(Lt_local - (t_grid[-1] + 1.0/sr)) < 0.05 * Lt_local, "Lt and t_grid mismatch"

    f0_path = f"data/synthetic/vowels/{tag}/f0/{tag}.npy"
    f0 = np.load(f0_path).astype(np.float32)
    assert len(f0) == len(wav), "F0 and audio length mismatch"

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    melbank  = MelBank(sr=sr, n_fft=512, n_mels=64, device=device)

    # Output directory( RID save)
    ckpt_root = ensure_dir(pathlib.Path("exp/A_static/ckps") / RUN_ID)
    tag_dir   = ensure_dir(ckpt_root / tag)
    figs_dir  = ensure_dir(pathlib.Path("exp/A_static/figs") / RUN_ID / tag)

    # Unified CSV for Stage A
    csv_path  = pathlib.Path("exp/A_static/ckps") / sA_CSV_NAME

    # ========== Baseline 1: DDSP-only ==========
    if RUN_MODE == "b1":
        print(f"[MODE=b1] DDSP-only baseline (no PINN/probe/PDE). RID={RUN_ID}")
        ps = PRESETS.get(tag, PRESETS["a"])
        probe_kwargs = ps["probe"].copy()
        K = int(probe_kwargs.get("K", 26))
        hop = int(probe_kwargs.get("hop", 256))

        predictor = HEnvPredictor(K=K, hidden=128).to(device)
        opt = torch.optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-6)

        best_rec = {"epoch": -1, "mSTFT_raw": 1e9, "LSD_raw_dB": 1e9}
        train_csv = tag_dir / "train_log.csv"
        train_header_written = train_csv.exists()

        for ep in range(1, epochs+1):
            if ep <= 50:
                w_mstft_win, w_env = 0.0, 0.0
            else:
                am = min(1.0, (ep - 50) / 600.0)
                w_mstft_win = 0.2 * am
                w_env = 0.1 * am

            i0, i1 = pick_window(t_grid, 4096 if tag=="i" else 2048)
            y_win = torch.tensor(wav[i0:i1], dtype=torch.float32, device=device)
            f0_win = torch.tensor(f0[i0:i1], dtype=torch.float32, device=device)

            f0_frames = F.avg_pool1d(f0_win.view(1,1,-1), hop, hop).view(-1).clamp_min(40.0)
            rms = torch.sqrt(F.avg_pool1d((y_win.view(1,1,-1)**2), hop, hop) + 1e-8).view(-1)
            loud_frames = (rms / rms.max().clamp_min(1e-6)).detach()

            H_env = predictor(f0_frames, loud_frames)
            audio_hat = synth_harmonic(f0_frames, H_env, loud_frames, sr=sr, hop=hop, K=K)

            T = min(len(audio_hat), len(y_win))
            a = (audio_hat[:T] / audio_hat[:T].std().clamp_min(1e-6)).clamp(-10,10)
            y = (y_win[:T]     / y_win[:T].std().clamp_min(1e-6)).clamp(-10,10)

            L = w_mstft_win * multi_stft_loss(a, y) + w_env * logmel_envelope_loss(a, y, melbank)
            opt.zero_grad(); L.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=5.0)
            opt.step()

            # Compute additional atomic losses(unweighted), train_log.csv(write only the available columns)
            l_mst = float(multi_stft_loss(a, y).item())
            l_env = float(logmel_envelope_loss(a, y, melbank).item())
            with open(train_csv, "a", newline="") as f:
                w = csv.writer(f)
                if not train_header_written:
                    w.writerow(["epoch","L","L_mstft","L_env"])
                    train_header_written = True
                w.writerow([ep, w_mstft_win * l_mst + w_env * l_env, l_mst, l_env])


            if ep % EVAL_PERIOD == 0:
                f0_full = torch.tensor(f0, dtype=torch.float32, device=device)
                f0_frames = F.avg_pool1d(f0_full.view(1,1,-1), hop, hop).view(-1).clamp_min(40.0)
                y_full = torch.tensor(wav, dtype=torch.float32, device=device)
                rms = torch.sqrt(F.avg_pool1d((y_full.view(1,1,-1)**2), hop, hop) + 1e-8).view(-1)
                loud_frames = (rms / rms.max().clamp_min(1e-6)).detach()

                with torch.no_grad():
                    H_env = predictor(f0_frames, loud_frames)
                    p_full = synth_harmonic(f0_frames, H_env, loud_frames, sr=sr, hop=hop, K=K)

                p_full = p_full / p_full.std().clamp_min(1e-6)
                y_full = y_full / y_full.std().clamp_min(1e-6)

                p_aln, y_aln, shift_samp, rho_peak, hit_limit = align_by_xcorr(
                    p_full, y_full, sr=sr, hop=hop, ignore_ms=300, max_shift_sec=0.30
                )
                m_full_raw = float(multi_stft_loss(p_full, y_full).item())
                l_full_raw = float(lsd_db(p_full, y_full).item())
                m_full_aln = float(multi_stft_loss(p_aln, y_aln).item())
                l_full_aln = float(lsd_db(p_aln, y_aln).item())

                hnr_db_hat = hnr_db(p_full, sr=sr)
                hnr_db_ref = hnr_db(y_full, sr=sr)

                # save DDSP (Normalize -0.5 dBFS )
                wav = p_full.detach().cpu().numpy().astype("float32")
                wav = 0.95 * (wav / (np.max(np.abs(wav)) + 1e-8))

                # Maintain a best checkpoint( m_full_raw for)
                if m_full_raw < best_rec["mSTFT_raw"] - 1e-6:
                    best_rec.update({"epoch": ep, "mSTFT_raw": m_full_raw, "LSD_raw_D": l_full_raw})
                    sf.write(tag_dir / f"{tag}_ddsp_best.wav", wav, sr)
                    print(f"[BEST] {tag} ep{ep}: mSTFT_raw={m_full_raw:.4f}  -> wrote ddsp_best.wav")

                # Write CSV( bc )
                write_header = (not csv_path.exists())
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow([
                            "mode","run_id","bc","tag","epoch",
                            "mSTFT_raw","LSD_raw_dB","mSTFT_aln","LSD_aln_dB","shift_samples",
                            "rho_peak","aln_hit_limit",
                            "p_gain","tau_sec","zeta",
                            "F1_MAE_Hz","F2_MAE_Hz","F3_MAE_Hz",
                            "hnr_db_hat","hnr_db_ref",
                            "w_form","w_henv","w_form_smooth","formant_scale","fsmooth_eval",
                            "lr_main","lr_p_gain","lr_tau","lr_zeta","lr_ddsp_mapper",
                            "use_ddsp_audio","ddsp_train_mapper"
                        ])
                    w.writerow([
                        RUN_MODE, RUN_ID, BC_TYPE, tag, ep,
                        m_full_raw, l_full_raw, m_full_aln, l_full_aln, shift_samp,
                        rho_peak, int(hit_limit),
                        1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,    # F1/F2/F3 not evaluated in b1
                        hnr_db_hat, hnr_db_ref,
                        0.0, 0.0, 0.0, 0.0, 0.0,
                        1e-3, 0.0, 0.0, 0.0, 0.0,
                        1, 0
                    ])

                # evaluation
                update_pngs(figs_dir, tag, RUN_ID, tag_dir / "train_log.csv", csv_path)
        return

    # ========== PINN ==========
    USE_FF = os.environ.get("USE_FF", "0").strip().lower() in ("1","true","yes")
    ONLY_T = os.environ.get("ONLY_T", "0").strip().lower() in ("1","true","yes")
    FF_M = _env_int("FF_M", 16)
    FF_SIG = _env_float("FF_SIGMA", 8.0)
    SIREN_W0 = _env_float("SIREN_W0", 30.0)
    net = DualNet(Lx=float(Lx), Lt=float(Lt_local), use_ff=USE_FF, only_t=ONLY_T,
                  ff_m=FF_M, ff_sigma=FF_SIG, siren_w0=SIREN_W0).to(device)

    # base lr(can be overridden by environment variables:BASE_LR)
    try:
        base_lr_env = os.environ.get("BASE_LR", "").strip()
        base_lr = float(base_lr_env) if base_lr_env else (2e-5 if RUN_MODE == "main" else 1e-4)
    except Exception:
        base_lr = 2e-5 if RUN_MODE == "main" else 1e-4
    opt = torch.optim.Adam(net.parameters(), lr=base_lr)

    best_rec = {"epoch": -1, "mSTFT_raw": 1e9, "LSD_raw_D": 1e9}
    best_metric = float('inf'); best_F2_mae_i = float('inf')
    no_improve_evals = 0; evals_since_improve = 0

    fsmooth_ema = None; formant_downscale_on = False; formant_scale = 1.0

    ddsp_mapper = None
    # DDSP C1 start epoch(can be overridden by environment variables)
    c1_start = int(DDSP_START_EPOCH)
    c2_start = 10**9
    w_f0_align_c2   = 0.05
    w_loud_align_c2 = 0.07

    MODE_NO_PDE   = (RUN_MODE == "no_pde")
    MODE_B2_PINN  = (RUN_MODE == "b2")
    MODE_NO_PROBE = (RUN_MODE == "no_probe")
    MODE_NO_TREG  = (RUN_MODE == "no_treg")
    MODE_NO_DDSP  = (RUN_MODE == "no_ddsp") or MODE_NO_DDSP_ENV

    ps = PRESETS.get(tag, PRESETS["a"])
    probe_kwargs = ps["probe"].copy()
    form_sched   = ps["form_sched"]

    cfg = dict(mode=RUN_MODE, run_id=RUN_ID, bc=BC_TYPE,
               tag=tag, sr=sr, base_lr=base_lr,
               Lx=float(Lx), Lt=float(Lt_local), c=float(c),
               use_ff=bool(USE_FF), ff_m=int(FF_M), only_t=bool(ONLY_T))
    (tag_dir / "config_sA.json").write_text(json.dumps(cfg, indent=2))

    for ep in range(1, epochs+1):
        # windowweight
        # windowramp-in(supports environment-variable override)
        if ep <= AUDIO_RAMP_START:
            pde_only = True
            w_mstft_win = 0.0
            w_env = 0.0
        else:
            pde_only = False
            denom = max(1, (AUDIO_RAMP_END - AUDIO_RAMP_START))
            am = min(1.0, max(0.0, (ep - AUDIO_RAMP_START) / denom))
            w_mstft_win = AUDIO_W_MSTFT_MAX * am
            w_env = AUDIO_W_ENV_MAX * am

        # PDE & global(regularizationusing a configurable schedule)
        if ep <= RAD_START_EPOCH:
            w_rad = 0.0
        else:
            rad_alpha = min(1.0, (ep - RAD_START_EPOCH) / max(1, RAD_RAMP_EPOCHS))
            w_rad = RAD_MAX * rad_alpha
        w_mstft_global = 0.0 if DISABLE_GLOBAL else global_mstft_scale(ep)
        use_global_grad = (ep >= 120)

        # probe weight
        fs = form_sched
        w_form = ramp(0.0, fs["to"], ep, fs["start"], fs["end"])
        w_henv = ramp(0.0, 0.2, ep, 800, 1400)
        w_form_smooth = 0.0 if ep < 1400 else ramp(0.02, 0.010, ep, 1400, 2000)

        if tag == "i":
            w_form *= 0.8; w_form_smooth *= 0.6

        if MODE_B2_PINN:
            use_ddsp_audio = False
            zero_probe = True
        else:
            use_ddsp_audio = (ep >= c1_start)
            zero_probe = MODE_NO_PROBE
        if MODE_NO_DDSP:
            use_ddsp_audio = False
        if MODE_NO_TREG: w_form_smooth = 0.0
        if zero_probe:   w_form = 0.0; w_henv = 0.0; w_form_smooth = 0.0

        if MODE_NO_PDE:
            w_pde = 0.0; w_axx = 0.0; w_geom = 0.0; w_Aend = 0.0; w_rad_now = 0.0
        else:
            # default(can be overridden by environment variables)
            w_pde = 1.0
            w_axx = 6e-8 if tag == "i" else 1e-7
            w_geom = 2e-3
            w_Aend = 5e-3
            w_rad_now = w_rad
            # cold start epoch()
            if ep <= RAD_FREEZE_EPOCHS:
                w_rad_now = 0.0
            # === b2 regularization ===
            if MODE_B2_PINN:
                w_axx = 1e-4

        # If set W_AXX environment variable, force override w_axx
        if W_AXX_OVERRIDE:
            try:
                w_axx = float(W_AXX_OVERRIDE)
            except Exception:
                pass

        ddsp_train_mapper = False
        if ep >= c2_start:
            if ddsp_mapper is None:
                K = (probe_kwargs or {}).get("K", 20)
                ddsp_mapper = HEnvMapper(K=K).to(device)
            ddsp_train_mapper = True

        wF_now = get_wF(tag, ep, device)
        zeta_cap = 0.18 if (tag == "i" and BC_TYPE == "robin") else None
        L_amp_factor = 1e-2 if tag in ("a","u") else 8e-3

        w_logA_TV = (1e-3 if MODE_B2_PINN else 0.0)

        # glot_gain ramp-in(cold start)
        if GLOT_GAIN_RAMP > 0 and ep <= GLOT_GAIN_RAMP:
            a = ep / float(GLOT_GAIN_RAMP)
            glot_gain_now = (1.0 - a) * GLOT_GAIN0 + a * GLOT_GAIN
        else:
            glot_gain_now = GLOT_GAIN

        # periodicityramp-in()
        if PERIOD_RAMP > 0 and ep <= PERIOD_RAMP:
            w_period_now = (ep / float(PERIOD_RAMP)) * W_PERIOD
        else:
            w_period_now = W_PERIOD

        # optionalwindow()
        T_WIN_ENV = _env_int("T_WIN", 0)

        cfg = dict(
            audio_ref=wav, t_grid=t_grid, f0_ref=f0, sr=sr,
            Lx=Lx, Lt=Lt_local, c=c, device=device, melbank=melbank,
            bc_type=BC_TYPE,
            # PDE
            w_pde=w_pde, w_axx=w_axx, w_geom=w_geom,
            w_Aend=1e-2,
            w_rad=w_rad_now,
            # Audio
            w_mstft=w_mstft_win, w_env=w_env, w_mstft_global=w_mstft_global,
            # Probe
            w_form=w_form, w_henv=w_henv, w_form_smooth=w_form_smooth,
            n_domain=4096, T_win=(T_WIN_ENV if T_WIN_ENV>0 else (4096 if tag=="i" else 2048)),
            pde_only=pde_only, use_global_grad=use_global_grad,
            probe_kwargs=probe_kwargs, formant_scale=formant_scale, formant_wF=wF_now,
            # Guards
            fsmooth_clip=(3.0 if tag=="i" else 3.5),
            p_gain_range=(1.0, 1.6) if tag=="i" else None,
            gain_soft_w=5e-4,
            zeta_soft_cap=zeta_cap, zeta_cap_w=1e-3,
            # DDSP
            use_ddsp_audio=use_ddsp_audio,
            ddsp_hop=probe_kwargs.get("hop", 256),
            ddsp_mapper=ddsp_mapper, ddsp_train_mapper=ddsp_train_mapper,
            w_f0_align=(0.0 if ep < c2_start else w_f0_align_c2),
            w_loud_align=(0.0 if ep < c2_start else w_loud_align_c2),
            # LRs
            p_gain_lr=(float(P_GAIN_LR_ENV) if P_GAIN_LR_ENV else (1e-5 if RUN_MODE=="main" else 5e-4)),
            t_shift_lr=(float(T_SHIFT_LR_ENV) if T_SHIFT_LR_ENV else (5e-5 if RUN_MODE=="main" else 3e-5)),
            rad_zeta_lr=(float(RAD_ZETA_LR_ENV) if RAD_ZETA_LR_ENV else (1e-5 if RUN_MODE=="main" else 5e-4)),
            ddsp_mapper_lr=1e-3,
            L_amp_factor=L_amp_factor,
            rar_pool=4096, 
            rar_frac=0.25, 
            rar_edge_frac=0.30,
            # source:;g_source
            w_source = float(W_SOURCE_ENV),
            g_source = max(0.0, 1.0 - ep / max(1, SOURCE_DECAY_EPOCHS)),
            w_logA_TV = w_logA_TV,
            w_glot = float(W_GLOT),
            # ramp-in gain
            glot_gain = float(glot_gain_now),
            # periodicity & zeta regularization
            w_period = float(w_period_now),
            w_zeta_reg = float(W_ZETA_REG),
            # teacher
            w_teacher = float(W_TEACHER),
            teacher_K = int(TEACHER_K),
            teacher_tilt = float(TEACHER_TILT),
            # auxiliary(t->p)
            aux_gamma0 = float(AUX_GAMMA0),
            aux_gamma_ramp = int(AUX_GAMMA_RAMP),
            aux_lr = (float(AUX_LR_ENV) if AUX_LR_ENV else None),
            aux_use_teacher = bool(AUX_USE_TEACHER),
            aux_use_harmonic = bool(AUX_USE_HARM),
            aux_harm_k = int(HARM_K),
            # cold start:fixwindow & current epoch
            ep = ep,
            fixed_win_epochs = int(FIXED_WIN_EPOCHS),
            fixed_win_center_frac = float(FIXED_WIN_CENTER_FRAC),
        )
        logs = train_one_epoch(net, opt, **cfg)

        # - Write per-epoch training log(train CSV)-
        train_csv = tag_dir / "train_log.csv"
        train_write_header = (not train_csv.exists())

        # current loss weight,""
        w_now = dict(
            w_pde=w_pde, w_axx=w_axx, w_geom=w_geom, w_Aend=5e-3,
            w_rad=w_rad_now, w_mstft=w_mstft_win, w_env=w_env,
            w_mstft_global=w_mstft_global, w_form=w_form, w_henv=w_henv,
            w_form_smooth=w_form_smooth
        )

        # Weighted contributions, used to see which term dominates total loss
        contrib = dict(
            c_pde   = w_now["w_pde"]          * logs["L_pde"],
            c_axx   = w_now["w_axx"]          * logs["L_smh"],
            c_geom  = w_now["w_geom"]         * logs["L_geom"],
            c_Aend  = w_now["w_Aend"]         * logs["L_Aend"],
            c_rad   = w_now["w_rad"]          * logs["L_rad"],
            c_mstft = w_now["w_mstft"]        * logs["L_mstft"],
            c_env   = w_now["w_env"]          * logs["L_env"],
            c_mfull = w_now["w_mstft_global"] * logs["L_mstft_full"],
            c_form  = w_now["w_form"]         * logs["L_form"],
            c_henv  = w_now["w_henv"]         * logs["L_henv"],
            c_fsmo  = w_now["w_form_smooth"]  * logs["L_form_smooth"],
        )

        # learning rate( param_group tag )
        lr_main = lr_p = lr_tau = lr_zeta = lr_ddsp = 0.0
        for g in opt.param_groups:
            tag_pg = g.get("tag", None)
            if tag_pg is None:           lr_main = g["lr"]
            elif tag_pg == "p_gain":     lr_p    = g["lr"]
            elif tag_pg == "t_shift":    lr_tau  = g["lr"]
            elif tag_pg == "rad_zeta":   lr_zeta = g["lr"]
            elif tag_pg == "ddsp_mapper":lr_ddsp = g["lr"]

        with open(train_csv, "a", newline="") as f:
            w = csv.writer(f)
            if train_write_header:
                w.writerow([
                    "mode","run_id","bc","tag","epoch",
                    # unweightedatomic loss
                    "L","L_mstft","L_env","L_mstft_full","L_pde","L_rad","L_glot","L_smh","L_geom","L_Aend","L_logA_TV",
                    "L_time","L_period","L_teacher","L_ic","L_amp","L_tau","L_form","L_henv","L_form_smooth","aux_gamma",
                    # ()
                    "c_mstft","c_env","c_mfull","c_pde","c_rad","c_glot","c_axx","c_geom","c_Aend","c_form","c_henv","c_fsmo",
                    # scalar
                    "p_gain","tau_sec","zeta","g_tau","g_zeta",
                    # weight(for reproducibility)
                    "w_mstft","w_env","w_mstft_global","w_pde","w_axx","w_geom","w_Aend","w_rad","w_form","w_henv","w_form_smooth",
                    # learning rate
                    "lr_main","lr_p_gain","lr_tau","lr_zeta","lr_ddsp_mapper",
                    # runtime stage information
                    "use_ddsp_audio","ddsp_train_mapper","fsmooth_ema"
                ])
            # :write order must match the header exactly; an earlier aux_gamma position caused column misalignment
            w.writerow([
                RUN_MODE, RUN_ID, BC_TYPE, tag, ep,
                # unweightedatomic loss( header )
                logs["L"], logs["L_mstft"], logs["L_env"], logs["L_mstft_full"], logs["L_pde"], logs["L_rad"], logs["L_glot"],
                logs["L_smh"], logs["L_geom"], logs["L_Aend"], logs["L_logA_TV"],
                logs.get("L_time", 0.0), logs.get("L_period", 0.0), logs.get("L_teacher", 0.0),
                logs["L_ic"], logs["L_amp"], logs["L_tau"], logs["L_form"], logs["L_henv"], logs["L_form_smooth"],
                logs.get("aux_gamma", 0.0),
                # Implementation note.
                contrib["c_mstft"], contrib["c_env"], contrib["c_mfull"], contrib["c_pde"], contrib["c_rad"],
                float(W_GLOT)*logs["L_glot"], contrib["c_axx"], contrib["c_geom"], contrib["c_Aend"], contrib["c_form"], contrib["c_henv"], contrib["c_fsmo"],
                # scalar
                logs["p_gain"], logs["tau_sec"], logs["zeta"], logs.get("g_tau",0.0), logs.get("g_zeta",0.0),
                # weight
                w_mstft_win, w_env, w_mstft_global, w_pde, w_axx, w_geom, 5e-3, w_rad_now, w_form, w_henv, w_form_smooth,
                # learning rate
                lr_main, lr_p, lr_tau, lr_zeta, lr_ddsp,
                int(use_ddsp_audio), int(ddsp_train_mapper), (fsmooth_ema if fsmooth_ema is not None else 0.0)
            ])

        # Compact one-line terminal output(printed every N steps)
        if (ep % 20) == 0:
            print(f"[{tag} ep{ep}] L={logs['L']:.3f} | win(mstft/env)={logs['L_mstft']:.3f}/{logs['L_env']:.3f} "
                f"| PDE={logs['L_pde']:.2e} rad={logs['L_rad']:.2e} Axx={logs['L_smh']:.2e} "
                f"| full={logs['L_mstft_full']:.3f} | gain={logs['p_gain']:.3f} "
                f"| tau={1e3*logs['tau_sec']:.2f}ms zeta={logs['zeta']:.3f} "
                f"| per={logs.get('L_period',0.0):.3f} teach={logs.get('L_teacher',0.0):.3f} "
                f"| aux_g={logs.get('aux_gamma',0.0):.2f}")

        # - Refresh training-curve PNG every fixed number of steps - #
        if (ep % 20) == 0:
            update_pngs(figs_dir, tag, RUN_ID, train_csv, csv_path)

        # fsmooth EMA & automatic down-weighting
        fs_val = float(logs.get("L_form_smooth", 0.0))
        if np.isfinite(fs_val):
            if fsmooth_ema is None: fsmooth_ema = fs_val
            else: fsmooth_ema = FSMOOTH_EMA_BETA * fsmooth_ema + (1 - FSMOOTH_EMA_BETA) * fs_val
            if (not formant_downscale_on) and fsmooth_ema > FSMOOTH_HI:
                print(f"[SAFEGUARD] fsmooth EMA={fsmooth_ema:.2f} > {FSMOOTH_HI}, downscale x{FORMANT_DOWNSCALE}")
                formant_downscale_on = True; formant_scale = FORMANT_DOWNSCALE
            elif formant_downscale_on and fsmooth_ema < FSMOOTH_LO:
                print(f"[SAFEGUARD] fsmooth EMA back {fsmooth_ema:.2f} < {FSMOOTH_LO}, restore=1.0")
                formant_downscale_on = False; formant_scale = 1.0

        if ep % EVAL_PERIOD == 0:
            torch.set_grad_enabled(False)

            p_full = eval_p_lip_series_full(net, Lx, t_grid, device=device, requires_grad=False)
            if MODE_B2_PINN:
                # fixgain:evaluation does not multiply by p_gain
                pass
            else:
                if hasattr(net, "p_gain"):
                    p_full = net.p_gain * p_full

            p_full = p_full / (p_full.std().clamp_min(1e-6))
            y_full = torch.tensor(wav, dtype=torch.float32, device=device)
            y_full = y_full / (y_full.std().clamp_min(1e-6))

            if MODE_B2_PINN:
                # fix:no alignment
                p_aln, y_aln = p_full, y_full
                shift_samples, rho_peak, hit_limit = 0, 0.0, False
            else:
                p_aln, y_aln, shift_samples, rho_peak, hit_limit = align_by_xcorr(
                    p_full, y_full, sr=sr, hop=256, ignore_ms=300, max_shift_sec=0.30
                )

            m_full_raw = float(multi_stft_loss(p_full, y_full).item())
            l_full_raw = float(lsd_db(p_full, y_full).item())
            m_full_aln = float(multi_stft_loss(p_aln, y_aln).item())
            l_full_aln = float(lsd_db(p_aln, y_aln).item())

            tau = float(_get_tshift(net).detach().cpu().item()) if hasattr(net, "t_shift_raw") else 0.0

            # ====== new:post-render evaluation(independent FDTD)======
            # 1) sample A(x)
            xs = torch.linspace(0, float(Lx), 512).view(-1,1).to(device)
            with torch.no_grad():
                A_vec = net.A_from_x(xs).view(-1).detach().cpu().numpy().astype(np.float32)

            # 2) get effective zeta( robin_fd for zeta)
            z_eff = _zeta_eff_from_net(net, BC_TYPE)

            # 3) generate with FDTD y_hat_post(same length and sample rate as reference)
            dur = len(wav) / sr
            Ax_fn = _make_Ax_fn(A_vec, float(Lx))
            f0_for_post = float(np.median(f0))  # use the median for simplicity; an existing f0_ref can also be used
            y_hat_post, _ = webster_1d_fd(
                Ax_fn, f0=f0_for_post, dur=dur, sr_out=sr, c=float(c), L=float(Lx),
                zeta_ref=z_eff, Oq=0.6, Cq=0.3, beta=10.0
            )
            # Metrics(Normalize)
            y_hat_post_t = torch.tensor(y_hat_post, dtype=torch.float32, device=device)
            y_hat_post_t = y_hat_post_t / (y_hat_post_t.std().clamp_min(1e-6))
            m_post = float(multi_stft_loss(y_hat_post_t, y_full).item())
            l_post = float(lsd_db(y_hat_post_t, y_full).item())


            # - optional:evaluation(controlled by environment variable)-
            if os.environ.get("SAVE_EVAL_WAV", "0").strip().lower() in ("1","true","yes"):
                _save_wav_norm(tag_dir / f"{tag}_hat_ep{ep}.wav", p_full, sr)


            # /summarize zeta
            if BC_TYPE == "robin":
                zeta = float(torch.nn.functional.softplus(
                    getattr(net, "rad_zeta_raw", torch.tensor(-2.0, device=device))
                ).detach().cpu().item())
            elif BC_TYPE == "robin_fd":
                z0 = torch.nn.functional.softplus(getattr(net, "rad_zeta0_raw", torch.tensor(-2.0, device=device)))
                z1 = torch.nn.functional.softplus(getattr(net, "rad_zeta1_raw", torch.tensor(-3.0, device=device)))
                zeta = float(torch.stack([z0, z1]).mean().detach().cpu().item())
            else:
                zeta = 0.0

            # === export Ahat.npy & zeta.txt(512 points)===
            try:
                xg = torch.linspace(0, float(Lx), steps=512, device=device).view(-1,1)
                Ahat = net.A_from_x(xg).detach().cpu().numpy()
                np.save(tag_dir / f"{tag}_Ahat.npy", Ahat.squeeze())
                with open(tag_dir / f"{tag}_zeta.txt","w") as fz:
                    fz.write(f"{zeta:.8f}\n")
            except Exception as e:
                print(f"[WARN] export Ahat/zeta failed: {e}")

            gain_v = (1.0 if MODE_B2_PINN else (float(net.p_gain.detach().cpu().item()) if hasattr(net,"p_gain") else 1.0))

            # ===== compute formant MAE first to decide whether to plot =====
            nfft_probe = int((probe_kwargs or {}).get("n_fft", 1024))
            W = max(nfft_probe, (nfft_probe // 2) + 1, 2048)
            W = min(W, len(p_full));  W = (W-1) if (W % 2) else W
            mid = len(p_full) // 2
            i0 = max(0, mid - W // 2); i1 = i0 + W
            p_win  = p_full[i0:i1].detach()
            y_win  = y_full[i0:i1].detach()
            f0_win = torch.tensor(f0[i0:i1], dtype=torch.float32, device=device)

            with torch.no_grad():
                pr = probe_all(y_win, sr=sr, f0_samples=f0_win, **(probe_kwargs or {}))
            ph = probe_all(p_win, sr=sr, f0_samples=f0_win, **(probe_kwargs or {}))
            F_mae = (ph["F"] - pr["F"]).abs().mean(dim=1)
            F1_mae = float(F_mae[0].detach().cpu().item())
            F2_mae = float(F_mae[1].detach().cpu().item())
            F3_mae = float(F_mae[2].detach().cpu().item())

            hnr_db_hat = hnr_db(p_full, sr=sr)
            hnr_db_ref = hnr_db(y_full, sr=sr)

            # ====== save ======
            # 1) save
            ref_png = figs_dir / "spec_ref.png"
            if PLOT_POLICY != "none" and (not ref_png.exists()):
                plot_spec_tensor(y_full, sr, f"REF {tag}", ref_png)

            # 2) Decide whether to plot
            is_final_eval = (ep >= epochs)
            if PLOT_POLICY == "every_eval":
                do_plot = True
            elif PLOT_POLICY == "final":
                do_plot = is_final_eval
            elif PLOT_POLICY == "best":
                improved_for_plot = (m_full_raw < best_metric - MIN_DELTA) or \
                                    (tag == "i" and (F2_mae + 1e-6 < best_F2_mae_i - F2_IMPROVE_EPS))
                do_plot = improved_for_plot
            else:  # "none" unknown
                do_plot = False

            if do_plot:
                plot_spec_tensor(
                    p_full, sr, f"HAT {tag}",
                    figs_dir / ("spec_hat_final.png" if is_final_eval else "spec_hat_best.png")
                )
                # formants / henv requires
                plot_formant_and_env(p_win, y_win, f0_win, sr, figs_dir, tag,
                                     (ep if not is_final_eval else epochs), probe_kwargs)

            # ====== Write CSV ======
            lr_main = lr_p = lr_tau = lr_zeta = lr_ddsp = 0.0
            for g in opt.param_groups:
                tag_pg = g.get("tag", None)
                if tag_pg is None:
                    lr_main = g["lr"]
                elif tag_pg == "p_gain":
                    lr_p = g["lr"]
                elif tag_pg == "t_shift":
                    lr_tau = g["lr"]
                elif (tag_pg is not None) and tag_pg.startswith("rad_zeta"):
                    lr_zeta = max(lr_zeta, g["lr"])
                elif tag_pg == "ddsp_mapper":
                    lr_ddsp = g["lr"]

            write_header = (not csv_path.exists())
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow([
                        "mode","run_id","bc","tag","epoch","m_post","l_post",
                        "mSTFT_raw","LSD_raw_dB","mSTFT_aln","LSD_aln_dB","shift_samples",
                        "rho_peak","aln_hit_limit",
                        "p_gain","tau_sec","zeta",
                        "F1_MAE_Hz","F2_MAE_Hz","F3_MAE_Hz",
                        "hnr_db_hat","hnr_db_ref",
                        "w_form","w_henv","w_form_smooth","formant_scale","fsmooth_eval",
                        "lr_main","lr_p_gain","lr_tau","lr_zeta","lr_ddsp_mapper",
                        "use_ddsp_audio","ddsp_train_mapper"
                    ])
                w.writerow([
                    RUN_MODE, RUN_ID, BC_TYPE, tag, ep, m_post, l_post,
                    m_full_raw, l_full_raw, m_full_aln, l_full_aln, shift_samples,
                    rho_peak, int(hit_limit),
                    gain_v, tau, zeta,
                    F1_mae, F2_mae, F3_mae,
                    hnr_db_hat, hnr_db_ref,
                    w_form, w_henv, w_form_smooth, formant_scale, (fsmooth_ema if fsmooth_ema is not None else 0.0),
                    lr_main, lr_p, lr_tau, lr_zeta, lr_ddsp,
                    int(use_ddsp_audio), int(ddsp_train_mapper)
                ])
            print(f"[CSV] {tag} ep{ep} -> {csv_path.resolve()}")

            # evaluation
            update_pngs(figs_dir, tag, RUN_ID, train_csv, csv_path)

            # ====== post-render Metrics best/early-stop ======
            improved = False
            # m_post forMetrics(smaller is better);for a more stable criterion, use (m_post, l_post) lexicographic order
            if m_post < best_metric - MIN_DELTA:
                best_metric = m_post
                improved = True
            # /i/ F2 special case, optional to keep(optional)
            if tag == "i" and (F2_mae + 1e-6 < best_F2_mae_i - F2_IMPROVE_EPS):
                best_F2_mae_i = F2_mae
                improved = True

            if improved:
                no_improve_evals = 0
                evals_since_improve = 0
                best_rec.update(dict(
                    epoch=ep,
                    mSTFT_post=m_post, LSD_post=l_post,  # :recordspost-render
                    mSTFT_raw=m_full_raw, LSD_raw_D=l_full_raw,  # compatible with
                    p_gain=gain_v, tau_sec=tau, zeta=zeta
                ))
                torch.save(net.state_dict(), tag_dir / f"{tag}_best_sA.pt")
                _save_wav_norm(tag_dir / f"{tag}_hat_best.wav", p_full, sr)              # direct-output backup
                _save_wav_norm(tag_dir / f"{tag}_hat_post_best.wav", y_hat_post, sr)  # :postRender
            else:
                no_improve_evals += 1
                evals_since_improve += 1

            if evals_since_improve >= LR_SCHED_PATIENCE_EVALS:
                for pg in opt.param_groups:
                    new_lr = max(pg['lr'] * LR_FACTOR, LR_MIN)
                    if new_lr < pg['lr']:
                        print(f"[LR] {pg.get('tag','main')} {pg['lr']:.2e} -> {new_lr:.2e}")
                        pg['lr'] = new_lr
                evals_since_improve = 0

            if no_improve_evals >= PATIENCE_EVALS:
                print(f"[EARLY-STOP] tag={tag} ep={ep} "
                    f"no_improve_evals={no_improve_evals} >= {PATIENCE_EVALS}")
                torch.save(net.state_dict(), tag_dir / f"{tag}_last_sA.pt")
                _save_wav_norm(tag_dir / f"{tag}_hat_last.wav", p_full, sr)
                _save_wav_norm(tag_dir / f"{tag}_hat_post_last.wav", y_hat_post, sr)  # :postRender
                (tag_dir / "EARLY_STOPPED").write_text(f"epoch={ep}\n")
                break


            # # ====== continue original"best"(unchanged) ======
            # improved = False
            # if m_full_raw < best_metric - MIN_DELTA: best_metric = m_full_raw; improved = True
            # if tag == "i" and (F2_mae + 1e-6 < best_F2_mae_i - F2_IMPROVE_EPS): best_F2_mae_i = F2_mae; improved = True
            # if improved:
            #     no_improve_evals = 0; evals_since_improve = 0
            #     best_rec.update(dict(epoch=ep, mSTFT_raw=m_full_raw, LSD_raw_D=l_full_raw,
            #                          p_gain=gain_v, tau_sec=tau, zeta=zeta))
            #     torch.save(net.state_dict(), tag_dir / f"{tag}_best_sA.pt")
            #     _save_wav_norm(tag_dir / f"{tag}_hat_best.wav", p_full, sr)
            # else:
            #     no_improve_evals += 1; evals_since_improve += 1
            # if evals_since_improve >= LR_SCHED_PATIENCE_EVALS:
            #     for pg in opt.param_groups:
            #         new_lr = max(pg['lr'] * LR_FACTOR, LR_MIN)
            #         if new_lr < pg['lr']:
            #             print(f"[LR] {pg.get('tag','main')} {pg['lr']:.2e} -> {new_lr:.2e}")
            #             pg['lr'] = new_lr
            #     evals_since_improve = 0

            # if no_improve_evals >= PATIENCE_EVALS:
            #     print(f"[EARLY-STOP] tag={tag} ep={ep} "
            #         f"no_improve_evals={no_improve_evals} >= {PATIENCE_EVALS}")
            # # optional:save""current,
            #     torch.save(net.state_dict(), tag_dir / f"{tag}_last_sA.pt")
            #     _save_wav_norm(tag_dir / f"{tag}_hat_last.wav", p_full, sr)
            #     # optional:write a marker file for script detection
            #     (tag_dir / "EARLY_STOPPED").write_text(f"epoch={ep}\n")
            #     break

        if ep % 500 == 0:
            torch.save(net.state_dict(), tag_dir / f"{tag}_last_sA.pt")
    (tag_dir / f"{tag}_best_sA.json").write_text(json.dumps(best_rec, indent=2))

    print(f"[BEST] {tag}: {best_rec}")
    # finalexportcurrent(final)
    try:
        with torch.no_grad():
            p_final = eval_p_lip_series_full(net, Lx, t_grid, device=device, requires_grad=False)
            if (not MODE_B2_PINN) and hasattr(net, "p_gain"):
                p_final = net.p_gain * p_final
            p_final = p_final / (p_final.std().clamp_min(1e-6))
        _save_wav_norm(tag_dir / f"{tag}_hat_final.wav", p_final, sr)
    except Exception as e:
        print(f"[WARN] export final wav failed: {e}")   

# ============ main ============
if __name__ == "__main__":
    print("[CWD]", os.getcwd())
    for tag in TAG_LIST:
        print(f"\n========== RUN TAG: {tag} ==========")
        run_one_vowel(tag, epochs=EPOCHS_ENV, sr=16000, seed=_env_int("SEED", 1234))
    print("\nAll vowels done.")
