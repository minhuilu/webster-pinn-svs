# file: exp/A_static/audio_bc.py Mouth-End
# Fixed mouth-end time grid plus lip-pressure operator
"""Pre-generate evenly spaced time points at the mouth end x=L and use an OperatorBC for the p_lip(t)-audio_ref(t) residual."""
import numpy as np
import torch
import deepxde as dde

def _torch_interp1d(x_query, x_ref, y_ref):
    """
    Linear interpolation from y_ref on x_ref to x_query.
    Requires x_ref to be increasing. Returns shape (N, 1).
    """
    # Ensure shapes
    xq = x_query.view(-1)
    xr = x_ref.view(-1)
    yr = y_ref.view(-1)

    # Clamp to range
    xq = torch.clamp(xq, xr[0], xr[-1])

    # Find intervals
    idx = torch.bucketize(xq, xr)  # in [0, len(xr)]
    idx = torch.clamp(idx, 1, len(xr)-1)

    x0 = xr[idx-1]; x1 = xr[idx]
    y0 = yr[idx-1]; y1 = yr[idx]

    w = (xq - x0) / (x1 - x0 + 1e-12)
    yq = y0 * (1 - w) + y1 * w
    return yq.view(-1,1)

class LipAudioBC(dde.icbc.OperatorBC):
    """
    Apply a time-domain audio constraint at the mouth end x=L.:
      p_lip(t) = -rho * ∂ψ/∂t  ~=  audio_ref(t)
    Key point: interpolate against the batch times X[:, 1], so no fixed t_grid is needed.
    """
    def __init__(self, geomtime, Lx: float, t_ref: np.ndarray, audio_ref: np.ndarray,
                 rho: float = 1.0, component_psi: int = 0, lip_tol: float = 1e-6):
        self.Lx = float(Lx)
        self.rho = float(rho)
        self.comp = int(component_psi)
        self.t_ref_t = torch.tensor(t_ref.astype(np.float32))  # (T,)
        self.y_ref_t = torch.tensor(audio_ref.astype(np.float32))  # (T,)
        self.lip_tol = float(lip_tol)

        def on_lip(X, on_boundary):
            # DeepXDE passes X here as a NumPy coordinate
            # Enable only boundary points near x=Lx
            return on_boundary and (abs(X[0] - self.Lx) <= self.lip_tol)

        super().__init__(geomtime, self._op, on_lip)

    def _op(self, X, Y, _):
        # Move X to torch.float32 on the correct device
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=Y.device)
        else:
            X = X.to(Y.device).float()
        X.requires_grad_(True)

        # Take t_batch directly from X
        t_batch = X[:, 1:2]

        psi = Y[:, self.comp:self.comp+1]
        dpsi_dt = dde.grad.jacobian(psi, X, i=0, j=1)
        p_hat = - self.rho * dpsi_dt

        # Interpolate the reference audio on the same device
        yq = _torch_interp1d(
            t_batch.view(-1),
            self.t_ref_t.to(Y.device),
            self.y_ref_t.to(Y.device),
        )
        return p_hat - yq

