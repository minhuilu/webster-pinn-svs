# exp/C_ddsp/ddsp_synth.py
# A PyTorch re-implementation of DDSP Harmonic (and a minimal noise stub)
# Interface aligned with magenta/ddsp.synths.Harmonic:
#   audio = Harmonic()(amplitudes, harmonic_distribution, f0_hz)
# where tensors are frame-rate; we upsample to sample-rate internally.
# References:
# - Engel et al. "DDSP: Differentiable Digital Signal Processing", ICLR 2020.
# - magenta/ddsp README: ddsp.synths.Harmonic(amplitudes, harmonic_distribution, f0_hz)

from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F

def _interp_frames_to_samples(x_frames: torch.Tensor, T_samples: int):
    """Linear upsampling from [T_frames, C] -> [T_samples, C]."""
    x = x_frames.transpose(0, 1).unsqueeze(0)  # [1, C, T_frames]
    x_up = F.interpolate(x, size=T_samples, mode="linear", align_corners=False)
    return x_up.squeeze(0).transpose(0, 1)     # [T_samples, C]

@torch.no_grad()
def _cumsum_phase(f0_hz_samples: torch.Tensor, sr: int, phi0: float = 0.0):
    """Instantaneous phase from per-sample f0 (Hz): φ[n] = sum(2π f0 / sr)."""
    omega = 2.0 * pi * f0_hz_samples.clamp_min(0.0) / float(sr)  # [T]
    phi = torch.cumsum(omega, dim=0)
    if phi0 != 0.0:
        phi = phi + phi0
    return torch.remainder(phi, 2.0 * pi)

class HarmonicSynth(nn.Module):
    """
    PyTorch version of DDSP Harmonic synthesizer.
    Inputs are frame-rate controls:
      amplitudes: [T_frames, 1]  (linear amplitude; you may map loudness->amplitude before calling)
      harmonic_distribution: [T_frames, K]  (non-negative; we'll renormalize per frame)
      f0_hz: [T_frames] or [T_frames, 1]
    Arguments:
      sr: sample rate; hop: samples per frame (analysis hop)
      max_harmonics: K if not inferable from input
      normalize_per_frame: re-normalize harmonic_distribution to sum=1 per frame
      anti_alias: zero-out harmonics above Nyquist per *sample*
    Returns:
      audio_hat: [T_samples]
    """
    def __init__(self, sr: int = 16000, hop: int = 256,
                 max_harmonics: int = None,
                 normalize_per_frame: bool = True,
                 anti_alias: bool = True):
        super().__init__()
        self.sr = int(sr)
        self.hop = int(hop)
        self.max_harmonics = max_harmonics
        self.normalize_per_frame = normalize_per_frame
        self.anti_alias = anti_alias

    def forward(self,
                amplitudes: torch.Tensor,           # [T_f, 1]
                harmonic_distribution: torch.Tensor,# [T_f, K]
                f0_hz: torch.Tensor                 # [T_f] or [T_f,1]
                ) -> torch.Tensor:
        assert amplitudes.dim() == 2 and amplitudes.size(1) == 1
        assert harmonic_distribution.dim() == 2
        T_f, K = harmonic_distribution.shape
        if self.max_harmonics is None:
            self.max_harmonics = K

        f0_f = f0_hz.view(-1, 1) if f0_hz.dim() == 1 else f0_hz
        assert f0_f.shape[0] == T_f

        # Target samples (overlap-save with hop)
        T = int(T_f * self.hop)

        # Frame -> sample upsampling
        A = _interp_frames_to_samples(amplitudes, T).view(T)            # [T]
        H = _interp_frames_to_samples(harmonic_distribution, T)         # [T, K]
        f0 = _interp_frames_to_samples(f0_f, T).view(T)                 # [T]

        # Optional per-frame normalization of harmonic distribution
        if self.normalize_per_frame:
            s = H.sum(dim=1, keepdim=True).clamp_min(1e-8)
            H = H / s

        # Anti-alias: zero harmonics above Nyquist for each sample
        if self.anti_alias:
            k_ids = torch.arange(1, K + 1, device=H.device).view(1, K)  # [1, K]
            nyq = 0.5 * float(self.sr)
            k_max = torch.clamp((nyq / f0.clamp_min(1e-8)).floor(), min=0.0, max=float(K))
            mask = (k_ids <= k_max.view(-1, 1)).to(H.dtype)             # [T, K]
            H = H * mask
            # renormalize if at least one remains
            s = H.sum(dim=1, keepdim=True)
            H = torch.where(s > 1e-8, H / s, H)

        # Per-sample phase of fundamental and its integer multiples
        phi = _cumsum_phase(f0, self.sr)                                # [T]
        k_ids = torch.arange(1, K + 1, device=H.device).view(1, K)
        phi_k = phi.view(T, 1) * k_ids                                  # [T, K]

        # Sum_k  A[n] * H[n,k] * sin(phi_k[n])
        audio = (H * torch.sin(phi_k)).sum(dim=1)                       # [T]
        audio = (A * audio).contiguous()
        # Clamp to avoid rare blow-ups (safety)
        return audio.clamp(min=-10.0, max=10.0)

def synth_harmonic(f0_frames: torch.Tensor,        # [T_f] or [T_f,1], Hz
                   H_env_frames: torch.Tensor,     # [T_f, K], non-negative
                   loud_frames: torch.Tensor,      # [T_f] or [T_f,1], linear (e.g., RMS/normalized)
                   sr: int = 16000, hop: int = 256, K: int = None,
                   amp_map: str = "linear"         # or "db": treat loud_frames as loudness dB
                   ) -> torch.Tensor:
    """Convenience wrapper: map loudness->amplitude then call HarmonicSynth."""
    if loud_frames.dim() == 1:
        loud_frames = loud_frames.view(-1, 1)
    if amp_map == "db":
        # loudness in dB -> linear amplitude (DDSP uses exp with log scaling internally)
        amplitudes = (10.0 ** (loud_frames / 20.0)).clamp_min(1e-5)
    else:
        amplitudes = loud_frames.clamp_min(0.0)
    synth = HarmonicSynth(sr=sr, hop=hop, max_harmonics=K, normalize_per_frame=True, anti_alias=True)
    return synth(amplitudes=amplitudes, harmonic_distribution=H_env_frames, f0_hz=f0_frames)

# --- (Optional) Minimal noise stub to extend later ---
class FilteredNoiseStub(nn.Module):
    """
    Minimal colored-noise placeholder (frame-wise gain on white noise).
    Replace with a true DDSP FilteredNoise if needed.
    """
    def __init__(self, sr=16000, hop=256):
        super().__init__()
        self.sr = sr; self.hop = hop
    def forward(self, magnitudes: torch.Tensor, T_samples: int) -> torch.Tensor:
        # magnitudes: [T_f, 1] -> upsample -> multiply white noise
        g = _interp_frames_to_samples(magnitudes, T_samples).view(-1)
        noise = torch.randn(T_samples, device=g.device)
        return g * noise

class HEnvMapper(torch.nn.Module):
    """
    C2 small mapper used to lightly reshape the probed harmonic envelope, with only a small number of parameters
    """
    def __init__(self, K: int, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(K, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, K), torch.nn.Softplus()
        )
    def forward(self, H_env_frames: torch.Tensor):
        # compatible with [K] / [T_frames, K]
        if H_env_frames.ndim == 1:
            return self.net(H_env_frames.view(1, -1)).view(-1)
        return self.net(H_env_frames)


