# --- add to: exp/A_static/audio_forward.py ---

import torch
import numpy as np

def _get_tshift(net, max_shift_s: float = 0.05):
    """
Return time-shift τ(), tanh :|τ| <= max_shift_s.
    """
    if not hasattr(net, "t_shift_raw"):
        net.t_shift_raw = torch.nn.Parameter(torch.tensor(0.0, device=next(net.parameters()).device))
    return max_shift_s * torch.tanh(net.t_shift_raw)

@torch.no_grad()
def pick_window(t_grid: np.ndarray, T_win: int):
    n = len(t_grid)
    if T_win >= n: return 0, n
    i0 = np.random.randint(0, n - T_win + 1)
    return i0, i0 + T_win

def eval_p_lip_series_window(net, Lx: float, t_win: np.ndarray, device="cpu"):
    # Allow
    with torch.enable_grad():
        tau = _get_tshift(net)  # tensor
        # Convert NumPy values to tensors before addition to keep the path in torch
        t = torch.tensor(t_win, dtype=torch.float32, device=device).view(-1, 1)
        t = t + tau
        t.requires_grad_(True)

        x = torch.full((len(t_win), 1), float(Lx), device=device)
        X = torch.cat([x, t], dim=1)
        psi = net(X)[:, :1]
        dpsi_dt = torch.autograd.grad(psi, t, grad_outputs=torch.ones_like(psi),
                                      retain_graph=True, create_graph=True)[0]
        return (-dpsi_dt).view(-1)


def eval_p_lip_series_full(net, Lx: float, t_grid: np.ndarray, device="cpu", requires_grad: bool = True):
    """
    Full lip-pressure time series.
    - requires_grad=True  for training/backpropagation, participates in gradients
- requires_grad=False evaluation(), grad dψ/dt, detach Return
    """
    if requires_grad:
        # Training path: build the full graph
        with torch.enable_grad():
            tau = _get_tshift(net)  # torch tensor, differentiable
            t = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
            t = t + tau
            t.requires_grad_(True)

            x = torch.full((len(t_grid), 1), float(Lx), device=device)
            X = torch.cat([x, t], dim=1)
            psi = net(X)[:, :1]
            dpsi_dt = torch.autograd.grad(
                psi, t, grad_outputs=torch.ones_like(psi),
                retain_graph=True, create_graph=True
            )[0]
            return (-dpsi_dt).view(-1)
    else:
        # evaluation:, grad , detach
        with torch.enable_grad():
            tau = _get_tshift(net)  # tensor
            t = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
            t = t + tau
            t.requires_grad_(True)

            x = torch.full((len(t_grid), 1), float(Lx), device=device)
            X = torch.cat([x, t], dim=1)
            psi = net(X)[:, :1]
            dpsi_dt = torch.autograd.grad(
                psi, t, grad_outputs=torch.ones_like(psi),
                retain_graph=False, create_graph=False
            )[0]
            p = (-dpsi_dt).view(-1)
            return p.detach()  # <- evaluationReturn
