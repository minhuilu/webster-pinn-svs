# exp/B_probes/formant_probe.py
import sys
import pathlib

# Automatically detect and fix package path issues
if __name__ == "__main__" and not __package__:
    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root))
    __package__ = "exp.B_probes"

import torch
import torch.nn.functional as F
from ..A_static.audio_losses import stft_mag  # stable STFT magnitude implementation

def _freq_grid(n_fft: int, sr: int, device):
    n_bins = n_fft // 2 + 1
    # [F, 1] for broadcasting
    return torch.linspace(0.0, sr / 2.0, n_bins, device=device).view(-1, 1)

def _soft_peak(freqs, spec_db, f_lo, f_hi, temp=9.0):
    """
    Softmax-based differentiable peak estimate:
      - freqs: [F,1]
      - spec_db: [F, T] (log magnitude; the temperature controls peak sharpness)
      - return f_peak[T], bw_fwhm[T], weights[F,T]
    """
    F_bins, T = spec_db.shape
    mask = (freqs >= float(f_lo)) & (freqs <= float(f_hi))  # [F,1]
    logits = spec_db.clone()
    logits[~mask.expand(-1, T)] = -1e9
    logits = logits - logits.max(dim=0, keepdim=True).values
    w = F.softmax(logits * (1.0 / temp), dim=0)  # [F,T]
    f = (w * freqs).sum(dim=0)                   # [T]
    var = (w * (freqs - f.view(1, -1))**2).sum(dim=0)
    bw = 2.355 * torch.sqrt(var + 1e-9)          # approximate bandwidth with FWHM
    return f, bw, w

def _interp_f0_to_frames(f0_samples, n_frames):
    """Resample the sample-level f0 sequence to the STFT frame grid with linear interpolation."""
    if f0_samples is None:
        return None
    x = f0_samples.view(1, 1, -1).float()
    y = torch.nn.functional.interpolate(x, size=n_frames, mode="linear", align_corners=False)
    return y.view(-1)

@torch.no_grad()
def _unit_norm(v, eps=1e-6):
    return (v - v.mean()) / (v.std() + eps)

def probe_all(
    x,                         # [T] Waveform (torch)
    sr: int,
    f0_samples=None,           # [T] sample-level f0, torch tensor or None
    n_fft: int = 1024,
    hop: int = 256,
    bands=((200,1000), (700,3000), (2000,5000)),
    temp: float = 6.0,
    K: int = 20,
    sigma_scale: float = 0.18,   # <- harmonic kernel width coefficient
    band_scale: float = 1.0,
):
    # Scale each formant search band by band_scale.
    if band_scale != 1.0:
        bands = tuple((band_scale*lo, band_scale*hi) for (lo, hi) in bands)
    """
    Returns:
      F123: [3, Frames]   (F1, F2, F3 trajectories)
B123: [3, Frames] (trajectories)
      H_env: [K]          (harmonic envelope, frame-averaged)
      noise: scalar         ("non-harmonic"average energy)
      frames: int
    """
    # Spectrum computed with the stable stft_mag helper.
    M = stft_mag(x, n_fft, hop)             # [1,F,Frames]
    S = torch.log(M.squeeze(0) + 1e-8)  # [F,Frames] log
    freqs = _freq_grid(n_fft, sr, x.device) # [F,1]
    F_list, B_list = [], []
    for (lo, hi) in bands:
        f, b, _ = _soft_peak(freqs, S, lo, hi, temp=temp)
        F_list.append(f); B_list.append(b)
    F123 = torch.stack(F_list, dim=0)       # [3,Frames]
    B123 = torch.stack(B_list, dim=0)       # [3,Frames]

    # Frame-wise harmonic envelope: integrate the spectrum around each k*F0 with Gaussian kernels.
    if f0_samples is None:
        f0_frames = torch.full((S.shape[1],), 160.0, device=x.device)
    else:
        f0_frames = _interp_f0_to_frames(f0_samples.to(x.device), S.shape[1])

    K = int(K)
    centers = torch.stack([ (k+1) * f0_frames for k in range(K) ], dim=0)  # [K,Frames]
    sigma = (sigma_scale * f0_frames.clamp_min(50.0)).view(1, -1)          # [1,Frames]
    freq_col = freqs.view(-1, 1)                                           # [F,1]
    # [F,K,Frames]
    w = torch.exp(-0.5 * ((freq_col.view(-1,1,1) - centers.view(1,K,-1)) / (sigma.view(1,1,-1)+1e-9))**2)
    w = w / (w.sum(dim=0, keepdim=True) + 1e-12)                           # normalize over frequency
    E_fk = (S.unsqueeze(1) * w).sum(dim=0)                                 # [K,Frames] weighted log magnitude
    # Average over frames to obtain a steady envelope; the time dimension can be retained for time-varying regularization.
    H_env = E_fk.mean(dim=-1)                                              # [K]
    H_env = _unit_norm(H_env)                                              # normalize to zero mean and unit variance

    harm_mask = w.sum(dim=1).clamp(0, 1)                                   # [F,Frames]
    noise = (S * (1 - harm_mask)).mean()  # "non-harmonic"Metrics

    return dict(F=F123, B=B123, H_env=H_env, noise=noise, frames=S.shape[1])
