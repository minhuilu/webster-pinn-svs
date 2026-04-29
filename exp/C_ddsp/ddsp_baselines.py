# file: exp/C_ddsp/ddsp_baselines.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# C3-1|Baseline 1:NN -> θ_DDSP(Without PDE)

class HEnvPredictor(nn.Module):
    """
Frame-wise mapping from [log f0, loud] -> H_env[K] minimal MLP
    Use softmax to distribute energy over K harmonics, matching the synth_harmonic interface.
    """
    def __init__(self, K: int = 26, hidden: int = 128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, K),
        )

    def forward(self, f0_frames: torch.Tensor, loud_frames: torch.Tensor) -> torch.Tensor:
        # f0_frames, loud_frames: [T]
        x = torch.stack([
            torch.log(f0_frames.clamp_min(1e-2)),
            loud_frames.clamp_min(1e-6)
        ], dim=-1)  # [T,2]
        h = self.net(x)             # [T,K]
        return torch.softmax(h, dim=-1)  # Normalize
