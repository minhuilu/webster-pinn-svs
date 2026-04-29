# file: exp/A_static/audio_losses.py
import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np


def stft_mag(x, n_fft, hop, win=None):
    """
    Numerically stable STFT magnitude:
      1) compute power, clamp it, then take sqrt
      2) clamp magnitude again to avoid log(0)
    """
    if win is None:
        win = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(
        x.view(1, -1), n_fft=n_fft, hop_length=hop, window=win,
        return_complex=True, center=True, pad_mode="reflect"
    )
    # power should be non-negative, but numerical noise can appear; clamp before sqrt
    power = X.real.pow(2) + X.imag.pow(2)
    mag = torch.sqrt(torch.clamp(power, min=1e-12))
    mag = mag.clamp_min(1e-8)
    return mag

def multi_stft_loss(x_hat, x_ref, cfg=((256,64,0.2),(512,128,0.3),(1024,256,0.5))):
    # Use a smaller maximum n_fft to reduce memory use
    loss = 0.0
    for n_fft, hop, w in cfg:
        M_hat = stft_mag(x_hat, n_fft, hop)
        M_ref = stft_mag(x_ref, n_fft, hop)
        # Magnitude is already clamped before log, so -inf is avoided
        loss = loss + w * ((M_hat.log() - M_ref.log()).abs().mean())
    return loss
    
class MelBank(torch.nn.Module):
    def __init__(self, sr=16000, n_fft=512, n_mels=64, device="cpu"):
        super().__init__()
        import librosa
        mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).astype(np.float32)
        self.register_buffer("mel", torch.tensor(mel, dtype=torch.float32, device=device))
        self.n_fft = n_fft

    def forward(self, x, hop=128):
        M = stft_mag(x, self.n_fft, hop)              # [1, F, T], numerically stable
        E = torch.matmul(self.mel, M.squeeze(0)).clamp_min(1e-8).log()
        return E

def logmel_envelope_loss(x_hat, x_ref, melbank: MelBank, hop=128):
    E_hat = melbank(x_hat, hop=hop)
    E_ref = melbank(x_ref, hop=hop)
    return (E_hat - E_ref).abs().mean()


def lsd_db(x_hat: torch.Tensor, x_ref: torch.Tensor, n_fft=1024, hop=256):
    """
    Log-Spectral Distance (dB): sqrt(mean( (log10|X| - log10|Y|)^2 )) over all bins.
    Return a scalar torch.Tensor.
    """
    M_hat = stft_mag(x_hat, n_fft, hop)  # [1, freq, frames]
    M_ref = stft_mag(x_ref, n_fft, hop)
    L_hat = (M_hat).log10()              # already lower-bounded
    L_ref = (M_ref).log10()
    d2 = (L_hat - L_ref).pow(2).mean()
    return (10.0 * d2.sqrt()).squeeze()  # dB

@torch.no_grad()
def _best_shift_indices(max_shift_frames, hop):
    # max_shift_frames=4, hop=256 -> shifts in samples: [-1024,...,0,...,+1024]
    return [s*hop for s in range(-max_shift_frames, max_shift_frames+1)]

def aligned_multi_stft_loss(x_hat, x_ref, cfg, max_shift_frames=4):
    """
    Enumerate shifts of +/- K frames under the multi-scale cfg and take the minimum mSTFT.
    Use one fixed hop as the alignment stride; here it is the hop from the largest scale.
    """
    # Use the largest hop in cfg as the alignment stride
    base_hop = max(h for _, h, _ in cfg)
    shifts = _best_shift_indices(max_shift_frames, base_hop)

    # Normalize x_hat and x_ref so amplitude differences do not choose the shift
    x_hat_n = x_hat / (x_hat.std().clamp_min(1e-6))
    x_ref_n = x_ref / (x_ref.std().clamp_min(1e-6))

    best = None
    best_val = None
    for s in shifts:
        if s >= 0:
            xh = F.pad(x_hat_n.unsqueeze(0), (s, 0)).squeeze(0)[:-s or None]
            xr = x_ref_n
        else:
            s2 = -s
            xh = x_hat_n
            xr = F.pad(x_ref_n.unsqueeze(0), (s2, 0)).squeeze(0)[:-s2 or None]

        # After alignment, use the original multi_stft_loss
        val = multi_stft_loss(xh, xr, cfg=cfg)
        if (best_val is None) or (val < best_val):
            best_val = val
            best = (s, val)

    # Return the minimum while preserving gradients via a second roll rather than the no_grad path
    s_best = best[0]
    if s_best >= 0:
        xh = F.pad(x_hat.unsqueeze(0), (s_best, 0)).squeeze(0)[:-s_best or None]
        xr = x_ref
    else:
        s2 = -s_best
        xh = x_hat
        xr = F.pad(x_ref.unsqueeze(0), (s2, 0)).squeeze(0)[:-s2 or None]

    return multi_stft_loss(xh, xr, cfg=cfg)