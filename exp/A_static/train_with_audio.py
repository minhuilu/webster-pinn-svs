# train_with_audio.py
import sys
import pathlib
if __name__ == "__main__" and not __package__:
    current_file = pathlib.Path(__file__).resolve()
    repo_root = current_file.parent.parent.parent  # exp/A_static -> exp -> repo
    sys.path.insert(0, str(repo_root))
    __package__ = "exp.A_static"

import torch
import os
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
import math
import deepxde as dde
import numpy as np

from typing import Optional
from .audio_forward import (
    eval_p_lip_series_window,
    eval_p_lip_series_full,
    pick_window,
    _get_tshift,
)
from .audio_losses import multi_stft_loss, logmel_envelope_loss
from .phys_consts import Lx, c
from exp.B_probes.formant_probe import probe_all
from exp.C_ddsp.ddsp_synth import synth_harmonic, HEnvMapper

# =========================
# Constant safeguards
# =========================
FSMOOTH_CLIP = 5.0  # "formant "default,
MAX_GRAD_NORM = 5.0


# -----------------------------
# : param_group
# -----------------------------
def ensure_param_group(opt, params, lr, wd=0.0, tag="extra"):
    existing = {id(p) for g in opt.param_groups for p in g["params"]}
    new = [p for p in params if id(p) not in existing]
    if new:
        opt.add_param_group({"params": new, "lr": lr, "weight_decay": wd, "tag": tag})


# -----------------------------
# :points points
# -----------------------------
def sample_collocation_rect(
    Lx, Lt, N_domain, device,
    edge_frac: float = 0.30,  # 30% points
    edge_width: float = 0.10  # points:0.1*Lx
):
    """
[0, Lx]×[0, Lt] :
- edge_frac :x∈[0, w*Lx] ∪ [ (1-w)*Lx, Lx ]
- :x∈[w*Lx, (1-w)*Lx]
t [0, Lt]
    """
    w = float(edge_width)
    Ne = int(N_domain * edge_frac)
    Ne = Ne + (Ne % 2)  # ,
    Nm = max(0, N_domain - Ne)

    # /
    xL = torch.rand(Ne // 2, 1, device=device) * (w * Lx)
    xR = (1.0 - w) * Lx + torch.rand(Ne // 2, 1, device=device) * (w * Lx)
    # Implementation note.
    if Nm > 0:
        xM = w * Lx + torch.rand(Nm, 1, device=device) * ((1 - 2 * w) * Lx)
        x = torch.cat([xL, xM, xR], dim=0)
    else:
        x = torch.cat([xL, xR], dim=0)

    t = torch.rand(x.shape[0], 1, device=device) * float(Lt)
    Xd = torch.cat([x, t], dim=1).requires_grad_(True)
    return Xd


# -----------------------------
# autograd auxiliary
# -----------------------------
def _grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


# =========================
# Fourier ( ψ )
# =========================
class FourierFeatures1D(torch.nn.Module):
    """
x t m /:
      phi(x,t) = [sin(2π*b_x*x/Lx), cos(...)]_{m} ⊕ [sin(2π*b_t*t/Lt), cos(...)]_{m}
b_x, b_t ~ N(0, sigma^2)
    """
    def __init__(self, m: int = 16, sigma: float = 8.0, Lx: float = 1.0, Lt: float = 1.0):
        super().__init__()
        self.m = m
        self.Lx = float(Lx)
        self.Lt = float(Lt)
        Bx = torch.randn(m) * sigma
        Bt = torch.randn(m) * sigma
        self.register_buffer("Bx", Bx)  # [m]
        self.register_buffer("Bt", Bt)  # [m]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (N,2) = [x,t], [0,1] 2π*b
        dev = self.Bx.device
        X = X.to(dev)
        x = X[:, 0:1] / self.Lx
        t = X[:, 1:2] / self.Lt
        # (N, m)
        Zx = 2.0 * math.pi * x @ self.Bx.view(1, -1)
        Zt = 2.0 * math.pi * t @ self.Bt.view(1, -1)
        fx = torch.cat([torch.sin(Zx), torch.cos(Zx)], dim=1)
        ft = torch.cat([torch.sin(Zt), torch.cos(Zt)], dim=1)
        return torch.cat([fx, ft], dim=1)  # (N, 4m)

    def t_features(self, X: torch.Tensor) -> torch.Tensor:
        """Only temporal Fourier features based on t component of X."""
        dev = self.Bt.device
        X = X.to(dev)
        t = X[:, 1:2] / self.Lt
        Zt = 2.0 * math.pi * t @ self.Bt.view(1, -1)
        ft = torch.cat([torch.sin(Zt), torch.cos(Zt)], dim=1)
        return ft  # (N, 2m)


# =========================
# SIREN( ψ )
# =========================
class SineLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, is_first=False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.is_first, self.w0 = is_first, w0
        self.lin = torch.nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Xavier-like for first layer
                self.lin.weight.uniform_(-1/self.in_features, 1/self.in_features)
            else:
                # SIREN
                bound = math.sqrt(6 / self.in_features) / self.w0
                self.lin.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.lin(x))


# =========================
# DualNet:ψ SIREN(+optional Fourier),A tanh FNN
# =========================
class DualNet(torch.nn.Module):
    """Separated heads: psi_net(x,t), A_net(x) -> A(x)>0 via softplus."""
    def __init__(
        self,
        # ψ
        use_ff: bool = True,
        ff_m: int = 16,
        ff_sigma: float = 8.0,
        siren_w0: float = 30.0,
        Lx: float = 1.0,
        Lt: float = 1.0,
        only_t: bool = False,
        # A
        A_layers=(1, 64, 64, 64, 1),
        A_activ="tanh",
        A_init="He normal",
    ):
        super().__init__()
        self.use_ff = use_ff
        self.only_t = only_t
        if use_ff:
            self.ff = FourierFeatures1D(m=ff_m, sigma=ff_sigma, Lx=Lx, Lt=Lt)
            psi_in = (1 + 2*ff_m) if only_t else (2 + 4*ff_m)
        else:
            psi_in = 1 if only_t else 2

        # --- SIREN psi_net ---
        hidden = 128
        self.psi_first = SineLayer(psi_in, hidden, w0=siren_w0, is_first=True)
        self.psi_h1    = SineLayer(hidden, hidden, w0=siren_w0)
        self.psi_h2    = SineLayer(hidden, hidden, w0=siren_w0)
        self.psi_out   = torch.nn.Linear(hidden, 1)

        # --- A_net tanh ---
        self.A_net = dde.nn.pytorch.FNN(list(A_layers), A_activ, A_init)

    def forward(self, x):
        dev = next(self.parameters()).device
        x = x.to(dev)
        if self.only_t:
            t = x[:, 1:2]
            if self.use_ff:
                psi_in = torch.cat([t, self.ff.t_features(x)], dim=1)
            else:
                psi_in = t
        else:
            psi_in = torch.cat([x, self.ff(x)], 1) if self.use_ff else x
        h = self.psi_first(psi_in)
        h = self.psi_h1(h)
        h = self.psi_h2(h)
        psi = self.psi_out(h)

        Ax  = self.A_net(x[:, 0:1])
        A   = F.softplus(Ax) + 1e-3
        return torch.cat([psi, A], dim=1)

    def A_from_x(self, x_only):
        dev = next(self.parameters()).device
        Ax = self.A_net(x_only.to(dev))
        return F.softplus(Ax) + 1e-3


# =========================
# (new) F0 glottal U_g(t)
# =========================
def synth_glottal_flow_from_f0(f0_1d: torch.Tensor, sr: int, K: int = 12, tilt: float = 1.2) -> torch.Tensor:
    """
F0 trajectories U_g(t):(1/k^tilt).·LF,,.
/for 1D tensor,.
    """
    f0 = f0_1d.clamp_min(40.0)
    phi = 2.0 * math.pi * torch.cumsum(f0 / sr, dim=0)
    g = torch.zeros_like(phi)
    for k in range(1, K + 1):
        g = g + (k ** (-tilt)) * torch.sin(k * phi)
    g = g / g.std().clamp_min(1e-6)
    return g


# =========================
# PDE (Webster)+ A'' + + / BC +()glottal
# =========================
def pde_residuals(
    net,
    Xd,
    c,
    Lx: float = 1.0,
    t_grid = None,
    bc_type: str = "robin",   # robin | dirichlet | neumann | robin_fd
    A_min: float = 1e-2,
    A_max: float = 2.2,
    zeta_soft_cap: Optional[float] = None,
    zeta_cap_w: float = 1e-3,
    # ---- new:glottal(optional) ----
    glot_flow: Optional[torch.Tensor] = None,
    sr: int = 16000,
    # ---- new:glottalgain( FDTD rho*c;default c) ----
    glot_gain: float = None,
):
    """
    Returns:res_web, A_xx, geom_barrier, res_bc, zeta_out, L_zeta, res_glot
- res_bc:()
- zeta_out: robin/robin_fd for tensor(scalar),for None
- L_zeta: robin/robin_fd forregularization,for 0
- res_glot:glottal(for"")
    """
    bc_type = bc_type.lower()
    assert bc_type in ("robin", "dirichlet", "neumann", "robin_fd"), f"Unknown bc_type={bc_type}"

    dev = next(net.parameters()).device
    X = Xd.to(dev).detach().clone().requires_grad_(True)  # (N,2)=[x,t]
    Y = net(X)
    psi = Y[:, :1]
    A   = Y[:, 1:2]

    # - :psi X -
    if (not X.requires_grad) or (not psi.requires_grad) or (psi.grad_fn is None):
        raise RuntimeError(
"psi (x,t) : DualNet.forward no_grad / detach / numpy ."
        )

    # (forward softplus)+
    A_safe = A
    under  = F.relu(A_min - A_safe)
    over   = F.relu(A_safe - A_max)
    geom_barrier = (under**2 + over**2).mean()

    # /
    g_psi   = _grad(psi, X)        # [ψ_x, ψ_t]
    dpsi_x  = g_psi[:, 0:1]
    dpsi_t  = g_psi[:, 1:2]
    dpsi_tt = _grad(dpsi_t, X)[:, 1:2]
    flux    = A_safe * dpsi_x
    flux_x  = _grad(flux, X)[:, 0:1]

    g_A   = _grad(A_safe, X)[:, 0:1]
    A_xx  = _grad(g_A, X)[:, 0:1]

    res_web = dpsi_tt / (c**2) - flux_x / A_safe

    # --- (x=L)points ---
    if t_grid is None:
        t_s = X[:, 1:2].detach()
    else:
        idx = np.random.randint(0, len(t_grid), size=256)
        t_s = torch.tensor(t_grid[idx], dtype=torch.float32, device=X.device).view(-1, 1)

    XL = torch.cat([torch.full_like(t_s, float(Lx)), t_s], dim=1).requires_grad_(True)
    psiL = net(XL)[:, :1]
    gL   = _grad(psiL, XL)
    dpsi_dx_L = gL[:, 0:1]
    dpsi_dt_L = gL[:, 1:2]

    # Implementation note.
    dpsi_dx_L = torch.nan_to_num(dpsi_dx_L, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e3, 1e3)
    dpsi_dt_L = torch.nan_to_num(dpsi_dt_L, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e3, 1e3)

    # glottal (x=0) points( t_s )
    X0_b = torch.cat([torch.zeros_like(t_s), t_s], dim=1).requires_grad_(True)
    psi0_b = net(X0_b)[:, :1]
    g0 = _grad(psi0_b, X0_b)
    dpsi_dx_0 = g0[:, 0:1]     # u(0,t)=psi_x(0,t)

    # ---- : psi_x(0,t) = (gain * u_scale * U_g(t)) / A(0,t) ----
    if glot_flow is not None:
        # glot_flow t_s
        t_idx = (t_s.view(-1) * sr).long().clamp(min=0, max=len(glot_flow)-1)
        U_batch = glot_flow[t_idx].view(-1, 1)  # [B,1]
        # A(0,t)
        A0_t = net(X0_b)[:, 1:2].detach()
        if not hasattr(net, "u_scale"):
            net.u_scale = torch.nn.Parameter(torch.tensor(1.0, device=X.device))
        # defaultgain: c; FDTD, rho*c
        gain = glot_gain if (glot_gain is not None) else float(c)
        res_glot = dpsi_dx_0 - (gain * net.u_scale * U_batch) / (A0_t + 1e-6)
    else:
        # compatible with:No U_g for""()
        res_glot = dpsi_dx_0

    # --- ---
    if bc_type == "robin":
        if not hasattr(net, "rad_zeta_raw"):
            net.rad_zeta_raw = torch.nn.Parameter(torch.tensor(-2.0, device=X.device))  # softplus~=0.12
        zeta = F.softplus(net.rad_zeta_raw)  # >0
        res_bc = dpsi_dx_L + zeta * dpsi_dt_L
        L_zeta = 1e-7*(torch.log1p(zeta)**2) + 1e-4*(zeta - 0.25)**2
        if (zeta_soft_cap is not None):
            L_zeta = L_zeta + zeta_cap_w * (F.relu(zeta - zeta_soft_cap) ** 2)
        zeta_out = zeta

    elif bc_type == "robin_fd":  # :dpsi_dx + ζ0*psi_t + ζ1*psi_tt
        if not hasattr(net, "rad_zeta0_raw"):
            net.rad_zeta0_raw = torch.nn.Parameter(torch.tensor(-2.0, device=X.device))
        if not hasattr(net, "rad_zeta1_raw"):
            net.rad_zeta1_raw = torch.nn.Parameter(torch.tensor(-3.0, device=X.device))
        zeta0 = F.softplus(net.rad_zeta0_raw)   # > 0
        zeta1 = F.softplus(net.rad_zeta1_raw)   # ≥ 0

        dpsi_tt_L = _grad(dpsi_dt_L, XL)[:, 1:2]
        dpsi_tt_L = torch.nan_to_num(dpsi_tt_L, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)

        res_bc = dpsi_dx_L + zeta0 * dpsi_dt_L + zeta1 * dpsi_tt_L
        L_zeta = 1e-7*(torch.log1p(zeta0)**2 + torch.log1p(zeta1)**2)
        if (zeta_soft_cap is not None):
            L_zeta = L_zeta + zeta_cap_w * (F.relu(zeta0 - zeta_soft_cap) ** 2 + F.relu(zeta1 - zeta_soft_cap) ** 2)
        zeta_out = torch.stack([zeta0, zeta1]).mean()

    elif bc_type == "dirichlet":  # :p(L,t)=0 -> psi_t(L,t)=0
        res_bc = dpsi_dt_L
        L_zeta = torch.tensor(0.0, device=X.device)
        zeta_out = None
    else:  # neumann::u(L,t)=0 -> psi_x(L,t)=0
        res_bc = dpsi_dx_L
        L_zeta = torch.tensor(0.0, device=X.device)
        zeta_out = None

    # Implementation note.
    for t in (psi, A, A_safe, dpsi_tt, flux_x, A_xx):
        if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
            nan = torch.nan
            return nan, nan, nan, nan, zeta_out, L_zeta, res_glot

    return res_web, A_xx, geom_barrier, res_bc, zeta_out, L_zeta, res_glot, A_safe.detach(), g_A


# -----------------------------
# (global)mSTFT for
# -----------------------------
def _global_audio_loss(net, Lx, t_grid, audio_ref, device):
    p = eval_p_lip_series_full(net, Lx, t_grid, device=device)
    if hasattr(net, "p_gain"):
        p = net.p_gain * p
    p = p / (p.std().clamp_min(1e-6))
    y = torch.tensor(audio_ref, dtype=torch.float32, device=device)
    y = y / (y.std().clamp_min(1e-6))
    return multi_stft_loss(p, y)

# ----------------( net )-------------
def _ema_update(buf, key, val, beta=0.98):
    if key not in buf:
        buf[key] = float(val)
    else:
        buf[key] = beta*buf[key] + (1-beta)*float(val)
    return buf[key]


# -----------------------------
# epoch
# -----------------------------
def train_one_epoch(
    net,
    opt,
    audio_ref,
    t_grid,
    Lx,
    Lt,
    c,
    device,
    melbank,
    f0_ref=None,
    sr=16000,
    w_pde=1.0,
    w_axx=1e-7,
    w_mstft=0.0,
    w_env=0.0,
    w_geom=2e-3,
    w_Aend=1e-2,
    w_rad=0.0,
    n_domain=4096,
    T_win=2048,
    pde_only=False,
    use_global_grad=False,
    w_mstft_global: float = 0.0,
    w_form: float = 0.0,
    w_henv: float = 0.0,
    w_form_smooth: float = 0.0,
    probe_kwargs: Optional[dict] = None,
    formant_scale: float = 1.0,
    formant_wF: Optional[torch.Tensor] = None,
    fsmooth_clip: Optional[float] = None,
    p_gain_range: Optional[tuple] = None,
    gain_soft_w: float = 5e-4,
    zeta_soft_cap: Optional[float] = None,
    zeta_cap_w: float = 1e-3,
    use_ddsp_audio: bool = False,
    ddsp_hop: Optional[int] = None,
    ddsp_mapper: Optional[torch.nn.Module] = None,
    ddsp_train_mapper: bool = False,
    w_f0_align: float = 0.0,
    w_loud_align: float = 0.0,
    p_gain_lr: float = 5e-4,
    t_shift_lr: float = 5e-4,
    rad_zeta_lr: float = 5e-4,
    ddsp_mapper_lr: float = 1e-3,
    L_amp_factor: float = 5e-3,
    bc_type: str = "robin",
    rar_pool: int = 4096,
    rar_frac: float = 0.25,
    rar_edge_frac: float = 0.30,
    # new:"source"(optional)
    w_source: float = 0.0,  # 0
    g_source: float = 0.5,  # ,
    # logA TV regularizationweight
    w_logA_TV: float = 0.1,
    glot_gain: Optional[float] = None,
    # cold start:fixwindow epoch
    ep: Optional[int] = None,
    fixed_win_epochs: int = 0,
    fixed_win_center_frac: float = 0.5,
    # glottalweight(default,cold start"source")
    w_glot: float = 0.02,
    # Robin regularizationweight(default 1.0,cold start 0)
    w_zeta_reg: float = 1.0,
    # cold start"periodicity": p_win F0 (default 0 )
    w_period: float = 0.0,
    # cold start"teacher": p_win f0 (default 0 )
    w_teacher: float = 0.0,
    teacher_K: int = 12,
    teacher_tilt: float = 1.2,
    # audio warmup head:t->p auxiliary
    aux_gamma0: float = 0.0,
    aux_gamma_ramp: int = 0,
    aux_lr: Optional[float] = None,
    aux_use_teacher: bool = False,
    # Option B:auxiliary(sin/cos(k·phi) )
    aux_use_harmonic: bool = False,
    aux_harm_k: int = 24,
):
    """
epoch:
- :Webster PDE + A''(x) + + (robin/robin_fd/dirichlet/neumann)+ points +
- :window( τ)+ Normalize + mSTFT + log-mel + mSTFT()
- scalar:p_gain,t_shift_raw,(robin ) rad_zeta_raw,(robin_fd ) rad_zeta0_raw/1_raw
- Stage C:PINN->probe->DDSP->audio(C1 ;C2 mapper)
    """
    torch.set_grad_enabled(True)
    net.train()

    # 1) ( LR )
    if not hasattr(net, "p_gain"):
        net.p_gain = torch.nn.Parameter(torch.tensor(1.0, device=device))
    ensure_param_group(opt, [net.p_gain],     lr=p_gain_lr,  wd=0.0, tag="p_gain")

    if not hasattr(net, "t_shift_raw"):
        net.t_shift_raw = torch.nn.Parameter(torch.tensor(0.0, device=device))
    ensure_param_group(opt, [net.t_shift_raw], lr=t_shift_lr, wd=0.0, tag="t_shift")

    opt.zero_grad()

    # (new) f0,glottal U_g(t)
    U_g_full = None
    if f0_ref is not None:
        f0_full = torch.tensor(f0_ref, dtype=torch.float32, device=device).view(-1)
        U_g_full = synth_glottal_flow_from_f0(f0_full, sr=sr)

    # 2) PDE ()
    with torch.no_grad():
        Xcand = sample_collocation_rect(Lx, Lt, rar_pool, device,
                                        edge_frac=rar_edge_frac, edge_width=0.10)
    # evaluation Webster forweight - requires
    with torch.enable_grad():
        (res_web_cand, *_), = [(pde_residuals(
            net, Xcand, c, Lx=Lx, t_grid=t_grid, bc_type=bc_type,
            zeta_soft_cap=zeta_soft_cap, zeta_cap_w=zeta_cap_w,
            glot_flow=U_g_full, sr=sr
        )[:1])]
        score = res_web_cand.abs().view(-1)

    # score k
    k = int(n_domain * rar_frac)
    k = max(0, min(k, Xcand.shape[0]))
    topk = torch.topk(score, k, largest=True).indices
    Xhard = Xcand[topk]

    # (1 - rar_frac) points
    Xeasy = sample_collocation_rect(Lx, Lt, n_domain - k, device,
                                    edge_frac=0.30, edge_width=0.10)

    # for epoch collocation
    Xd = torch.cat([Xhard, Xeasy], dim=0).detach().requires_grad_(True)

    res_web, A_xx, L_geom, res_bc, zeta, L_zeta, res_glot, A_val, A_x = pde_residuals(
        net, Xd, c, Lx=Lx, t_grid=t_grid, bc_type=bc_type,
        zeta_soft_cap=zeta_soft_cap, zeta_cap_w=zeta_cap_w,
        glot_flow=U_g_full, sr=sr, glot_gain=(float(glot_gain) if glot_gain is not None else None)
    )

    # --- new:TV(logA) ---
    epsA = 1e-6
    logA_x = A_x / (A_val + epsA)
    L_logA_TV = logA_x.abs().mean()

    # robin / robin_fd
    if (bc_type.lower() == "robin") and hasattr(net, "rad_zeta_raw"):
        ensure_param_group(opt, [net.rad_zeta_raw], lr=rad_zeta_lr, wd=0.0, tag="rad_zeta")
    if (bc_type.lower() == "robin_fd"):
        for name in ["rad_zeta0_raw", "rad_zeta1_raw"]:
            if hasattr(net, name):
                ensure_param_group(opt, [getattr(net, name)], lr=rad_zeta_lr, wd=0.0, tag=name)
    # ()glottal u_scale:default,source"";, LEARN_USCALE=1
    learn_uscale = (str(os.environ.get("LEARN_USCALE", "0")).lower() in ("1","true","yes"))
    if hasattr(net, "u_scale") and learn_uscale:
        ensure_param_group(opt, [net.u_scale], lr=5e-4, wd=0.0, tag="u_scale")

    L_pde = (res_web**2).mean()
    L_smh = (A_xx**2).mean()
    L_bc  = (res_bc**2).mean()
    L_rad = L_bc  # compatible with
    L_glot = (res_glot**2).mean()

    # points A(0)=A(1)=1()
    with torch.no_grad():
        idx = np.random.randint(0, len(t_grid), size=64)
        t_s = torch.tensor(t_grid[idx], dtype=torch.float32, device=device).view(-1, 1)
    X0 = torch.cat([torch.zeros_like(t_s), t_s], dim=1).requires_grad_(True)
    XL = torch.cat([torch.full_like(t_s, float(Lx)), t_s], dim=1).requires_grad_(True)
    A0 = net(X0)[:, 1:2]
    AL = net(XL)[:, 1:2]
    L_Aend = ((A0 - 1.0)**2).mean() + ((AL - 1.0)**2).mean()

    # 3) (window):,,Normalize,
    # cold start: fixed_win_epochs epoch "window",
    if (ep is not None) and (fixed_win_epochs > 0) and (ep <= fixed_win_epochs):
        n = len(t_grid)
        if T_win >= n:
            i0, i1 = 0, n
        else:
            center = int(round((n - 1) * max(0.0, min(1.0, fixed_win_center_frac))))
            i0 = max(0, min(center - T_win // 2, n - T_win))
            i1 = i0 + T_win
    else:
        i0, i1 = pick_window(t_grid, T_win)
    t_win = t_grid[i0:i1]
    y_win = torch.tensor(audio_ref[i0:i1], dtype=torch.float32, device=device).view(-1)

    p_win = eval_p_lip_series_window(net, Lx, t_win, device=device)  # τ
    p_win = torch.nan_to_num(p_win, nan=0.0, posinf=0.0, neginf=0.0)
    p_win = net.p_gain * p_win
    # records,teacher( aux_gamma=1 /)
    p_net_for_teacher = p_win

    # - Warmup: auxiliary audio ( t), PINN γ -
    aux_gamma = 0.0
    if aux_gamma0 > 0.0:
        if (aux_gamma_ramp is not None) and (aux_gamma_ramp > 0) and (ep is not None):
            a = min(1.0, max(0.0, ep / float(aux_gamma_ramp)))
            aux_gamma = float((1.0 - a) * aux_gamma0)
        else:
            aux_gamma = float(aux_gamma0)

    if aux_gamma > 0.0:
        if aux_use_teacher and (f0_ref is not None):
            # teacherforauxiliary()
            f0_win_local = torch.tensor(f0_ref[i0:i1], dtype=torch.float32, device=device)
            g_aux = synth_glottal_flow_from_f0(f0_win_local.view(-1), sr=sr, K=int(teacher_K), tilt=float(teacher_tilt))
            g_aux = (g_aux - g_aux.mean()) / (g_aux.std().clamp_min(1e-6))
            p_aux = g_aux.view_as(p_win)
        elif aux_use_harmonic and (f0_ref is not None):
            # f0 :B=[sin(kphi), cos(kphi)]_{k=1..K}
            f0_win_local = torch.tensor(f0_ref[i0:i1], dtype=torch.float32, device=device).view(-1)
            dt = 1.0 / float(sr)
            phi = 2.0 * math.pi * torch.cumsum(f0_win_local.clamp_min(0.0) * dt, dim=0)
            K = int(max(1, aux_harm_k))
            # [T, 2K]
            Bs = []
            for k in range(1, K+1):
                Bs.append(torch.sin(k*phi))
                Bs.append(torch.cos(k*phi))
            B = torch.stack(Bs, dim=1)  # [T, 2K]
            # :2K->1
            if not hasattr(net, "aux_harm_head"):
                net.aux_harm_head = torch.nn.Linear(2*K, 1, bias=True).to(device)
                lr_aux = float(aux_lr) if (aux_lr is not None) else 5e-4
                ensure_param_group(opt, net.aux_harm_head.parameters(), lr=lr_aux, wd=0.0, tag="aux_harm_head")
            p_aux = net.aux_harm_head(B).view(-1)
            # ,
            p_aux = p_aux - p_aux.mean()
        else:
            # MLP(t)->p forauxiliary
            if not hasattr(net, "aux_head"):
                net.aux_head = torch.nn.Sequential(
                    torch.nn.Linear(1, 64), torch.nn.Tanh(),
                    torch.nn.Linear(64, 64), torch.nn.Tanh(),
                    torch.nn.Linear(64, 1)
                ).to(device)
                lr_aux = float(aux_lr) if (aux_lr is not None) else 5e-4
                ensure_param_group(opt, net.aux_head.parameters(), lr=lr_aux, wd=0.0, tag="aux_head")

            t_torch = torch.tensor(t_win, dtype=torch.float32, device=device).view(-1, 1)
            t0 = float(t_win[0]); t1 = float(t_win[-1])
            tn = (t_torch - 0.5*(t0+t1)) / (0.5*(t1 - t0) + 1e-8)
            p_aux = net.aux_head(tn).view(-1)

        p_win = (1.0 - aux_gamma) * p_win + aux_gamma * p_aux

    p_std = p_win.std().clamp_min(1e-6)
    y_std = y_win.std().clamp_min(1e-6)
    scale = (p_std / 1e-2).clamp(min=0.1, max=1.0)

    p_win = (p_win / p_std).clamp(-10.0, 10.0)
    y_win = (y_win / y_std).clamp(-10.0, 10.0)

    # : p_win (""),;fix scale=1
    scale = torch.tensor(1.0, device=device)
    L_mstft = scale * multi_stft_loss(p_win, y_win)
    L_env   = scale * logmel_envelope_loss(p_win, y_win, melbank)
    L_amp   = ((p_std - y_std).detach() * (p_std - y_std)) * L_amp_factor

    # NEW: (/); L_TIME_W default 0.2
    try:
        _ltw = float(os.environ.get("L_TIME_W", "0.2"))
    except Exception:
        _ltw = 0.2
    L_time  = _ltw * (p_win - y_win).abs().mean()

    # ========= Probe / DDSP =========
    f0_win = None
    if f0_ref is not None:
        f0_win = torch.tensor(f0_ref[i0:i1], dtype=torch.float32, device=device)

    # === ()===
    use_probe_losses = (w_form > 0.0) or (w_henv > 0.0) or (w_form_smooth > 0.0) or use_ddsp_audio

    if use_probe_losses:
        pk = probe_kwargs or {}
        probe_hat = probe_all(p_win, sr=sr, f0_samples=f0_win, **pk)  # p_win
        with torch.no_grad():
            probe_ref = probe_all(y_win, sr=sr, f0_samples=f0_win, **pk)
    else:
        probe_hat = {"F": None, "H_env": None}
        probe_ref = {"F": None, "H_env": None}

    # - Formant trajectories/ -
    if use_probe_losses:
        if formant_wF is None:
            wF = torch.ones(3, 1, device=device)
        else:
            wF = formant_wF.view(3, 1) if formant_wF.dim() == 1 else formant_wF
        L_form = (wF * (probe_hat["F"] - probe_ref["F"]).abs()).mean()
        L_henv = (probe_hat["H_env"] - probe_ref["H_env"]).abs().mean()
        if probe_hat["F"] is not None and probe_hat["F"].shape[1] > 1:
            dF = probe_hat["F"][:, 1:] - probe_hat["F"][:, :-1]
            L_form_smooth = dF.abs().mean()
            L_form_smooth_raw = float(L_form_smooth.detach().cpu())
            _clip_max = fsmooth_clip if (fsmooth_clip is not None) else FSMOOTH_CLIP
            L_form_smooth = torch.clamp(L_form_smooth, max=_clip_max)
        else:
            L_form_smooth = torch.tensor(0.0, device=device)
            L_form_smooth_raw = 0.0
    else:
        L_form = torch.tensor(0.0, device=device)
        L_henv = torch.tensor(0.0, device=device)
        L_form_smooth = torch.tensor(0.0, device=device)
        L_form_smooth_raw = 0.0

    # 4) IC:psi(x,0) ~ sin(pi x)
    xb  = torch.rand(256, 1, device=device)
    tb0 = torch.zeros_like(xb)
    X_ic = torch.cat([xb, tb0], dim=1)
    psi0 = net(X_ic)[:, :1]
    L_ic = ((psi0 - torch.sin(torch.pi * xb))**2).mean() * 1e-4

    # 5) τ regularization(+/-200ms)
    tau = _get_tshift(net)
    L_tau = 3e-5 * (tau / 0.2)**2

    # 6) (global)mSTFT(weight 0 )
    if w_mstft_global > 0.0:
        if use_global_grad:
            L_mstft_full = _global_audio_loss(net, Lx, t_grid, audio_ref, device)
        else:
            L_mstft_full = _global_audio_loss(net, Lx, t_grid, audio_ref, device).detach()
    else:
        L_mstft_full = torch.tensor(0.0, device=device)

    # === Stage-C: PINN -> probe -> DDSP -> audio_hat_ddsp ===
    L_mstft_ddsp = None
    L_env_ddsp   = None
    L_f0_align   = torch.tensor(0.0, device=device)
    L_loud_align = torch.tensor(0.0, device=device)

    if use_ddsp_audio:
        hop = int((probe_kwargs or {}).get("hop", ddsp_hop if ddsp_hop is not None else 256))
        if f0_win is None:
            f0_frames = torch.full_like(y_win[::hop], 140.0)
        else:
            f0_frames = F.avg_pool1d(f0_win.view(1,1,-1), kernel_size=hop, stride=hop).view(-1)

        H_env_frames = probe_hat["H_env"]
        if H_env_frames.ndim == 1:
            H_env_frames = H_env_frames.view(1, -1).repeat(int(f0_frames.shape[0]), 1)

        if ddsp_mapper is not None:
            if ddsp_train_mapper:
                ensure_param_group(opt, ddsp_mapper.parameters(), lr=ddsp_mapper_lr, wd=0.0, tag="ddsp_mapper")
                H_env_frames = ddsp_mapper(H_env_frames)
            else:
                with torch.no_grad():
                    H_env_frames = ddsp_mapper(H_env_frames)

        rms = torch.sqrt(F.avg_pool1d((y_win.view(1,1,-1)**2), hop, hop) + 1e-8).view(-1)
        loud_frames = (rms / (rms.max().clamp_min(1e-6)))

        f0_in = f0_frames.detach() if w_f0_align == 0.0 else f0_frames
        audio_hat_ddsp = synth_harmonic(
            f0_frames=f0_in, H_env_frames=H_env_frames, loud_frames=loud_frames,
            sr=sr, hop=hop, K=H_env_frames.shape[1]
        )

        T_hat = int(audio_hat_ddsp.shape[0])
        T_ref = int(y_win.shape[0])
        T_cmp = min(T_hat, T_ref)
        audio_hat_ddsp = audio_hat_ddsp[:T_cmp]
        y_win_ddsp = y_win[:T_cmp]

        a_std = audio_hat_ddsp.std().clamp_min(1e-6)
        audio_hat_ddsp = (audio_hat_ddsp / a_std).clamp(-10.0, 10.0)

        L_mstft_ddsp = multi_stft_loss(audio_hat_ddsp, y_win_ddsp)
        L_env_ddsp   = logmel_envelope_loss(audio_hat_ddsp, y_win_ddsp, melbank)

        if w_f0_align > 0.0 and f0_win is not None:
            f0_ref_frames = F.avg_pool1d(f0_win.view(1,1,-1), hop, hop).view(-1)
            L_f0_align = (f0_frames - f0_ref_frames).abs().mean()
        if w_loud_align > 0.0:
            p_rms = torch.sqrt(F.avg_pool1d((p_win.view(1,1,-1)**2), hop, hop) + 1e-8).view(-1)
            p_rms = p_rms / (p_rms.max().clamp_min(1e-6))
            L_loud_align = (loud_frames - p_rms).abs().mean()

    # ========= (EMA Normalize) =========
    if not hasattr(net, "_ema_loss"):
        net._ema_loss = {}
    ema = net._ema_loss
    eps = 1e-12
    with torch.no_grad():
        n_pde = _ema_update(ema, "pde", L_pde.item()) + eps
        n_bc = _ema_update(ema, "bc", L_rad.item()) + eps
        n_geo = _ema_update(ema, "geo", (L_geom.item() + L_Aend.item())) + eps
        n_wst = _ema_update(ema, "wst", (L_mstft.item() + L_env.item())) + eps
        n_gst = _ema_update(
            ema, "gst", (L_mstft_full.item() if torch.is_tensor(L_mstft_full) else float(L_mstft_full))
        ) + eps
        n_probe = _ema_update(
            ema, "probe", (L_form.item() + L_henv.item() + L_form_smooth.item())
        ) + eps

    # ========= =========
    if use_ddsp_audio and (L_mstft_ddsp is not None):
        L_audio_main = w_mstft * (L_mstft_ddsp / n_wst) + w_env * (L_env_ddsp / n_wst)
        L_align = w_f0_align * L_f0_align + w_loud_align * L_loud_align
    else:
        L_audio_main = w_mstft * (L_mstft / n_wst) + w_env * (L_env / n_wst)
        L_align = 0.0

    # - periodicity( f0_win )-
    L_period = torch.tensor(0.0, device=device)
    if (w_period > 0.0) and (f0_win is not None):
        # sin/cos
        dt = 1.0 / float(sr)
        phi = 2.0 * math.pi * torch.cumsum(f0_win.clamp_min(0.0) * dt, dim=0)
        s = torch.sin(phi); c = torch.cos(phi)
        # Normalize p_win ~0 ~1
        a_s = (p_win * s).mean(); a_c = (p_win * c).mean()
        proj = a_s*a_s + a_c*a_c
        L_period = (1.0 - proj.clamp(0.0, 1.0))  # 1

    # - teacher: f0 L1 -
    L_teacher = torch.tensor(0.0, device=device)
    if (w_teacher > 0.0) and (f0_win is not None):
        # f0 ,"", aux L_teacher 0
        g = synth_glottal_flow_from_f0(f0_win.view(-1), sr=sr, K=int(teacher_K), tilt=float(teacher_tilt))
        g = (g - g.mean()) / (g.std().clamp_min(1e-6))
        g = g.view_as(p_win)
        # Normalize( p_win )
        p_net_std = p_net_for_teacher.std().clamp_min(1e-6)
        p_net_norm = (p_net_for_teacher / p_net_std).clamp(-10.0, 10.0)
        L_teacher = (p_net_norm - g).abs().mean()

    L_gain_soft = 0.0
    if p_gain_range is not None:
        lo, hi = p_gain_range
        L_gain_soft = gain_soft_w * (F.relu(net.p_gain - hi) ** 2 + F.relu(lo - net.p_gain) ** 2)

    # Normalize()
    with torch.no_grad():
        n_pde = _ema_update(ema, "pde", L_pde.item()) + eps
        n_bc  = _ema_update(ema, "bc",  L_rad.item()) + eps
        n_geo = _ema_update(ema, "geo", L_geom.item()+L_Aend.item()) + eps
        n_wst = _ema_update(ema, "wst", L_mstft.item()+L_env.item()) + eps
        n_gst = _ema_update(ema, "gst", (L_mstft_full.item() if torch.is_tensor(L_mstft_full) else L_mstft_full)+0.0) + eps
        n_probe = _ema_update(ema, "probe", L_form.item()+L_henv.item()+L_form_smooth.item()) + eps

    # ---- "source"(optional;x=0 psi_t envelope alignment)----
    def _smooth_envelope(y, k=64):
        y_abs = y.abs().view(1,1,-1)
        env = F.avg_pool1d(y_abs, kernel_size=k, stride=1, padding=k//2).view(-1)
        return env / env.max().clamp_min(1e-6)

    L_source_weak = torch.tensor(0.0, device=device)
    if w_source > 0.0:
        env_ref = _smooth_envelope(y_win, k=64)
        t_seq = torch.tensor(t_win, dtype=torch.float32, device=device).view(-1,1)
        X0_seq = torch.cat([torch.zeros_like(t_seq), t_seq], dim=1).requires_grad_(True)
        psi0_t = _grad(net(X0_seq)[:, :1], X0_seq)[:, 1:2].view(-1)
        env_hat = _smooth_envelope(psi0_t, k=64)
        L_source_weak = (env_hat - env_ref).abs().mean() * (w_source * float(g_source))

    # u_scale regularization(default;for 0)
    L_uscale = torch.tensor(0.0, device=device)
    if hasattr(net, "u_scale"):
        L_uscale = 1e-3 * (net.u_scale - 1.0) ** 2

    L = (
        w_pde * (L_pde / n_pde)
        + w_axx * L_smh
        + w_geom * ((L_geom + L_Aend) / n_geo)
        + w_rad * (L_rad / n_bc)
        + float(w_zeta_reg) * L_zeta
        + L_audio_main
        + L_time
        + w_mstft_global * (L_mstft_full / n_gst if use_global_grad else (L_mstft_full.detach() / n_gst))
        + L_ic
        + L_tau
        + L_amp
        + 1e-5 * (net.p_gain - 1.0) ** 2
        + (L_gain_soft if isinstance(L_gain_soft, torch.Tensor) else torch.tensor(L_gain_soft, device=device))
        + (w_form * formant_scale) * (L_form / n_probe)
        + w_henv * (L_henv / n_probe)
        + w_form_smooth * (L_form_smooth / n_probe)
        + (L_align if isinstance(L_align, torch.Tensor) else torch.tensor(L_align, device=device))
        + float(w_glot) * L_glot
        + L_uscale
        + L_source_weak
        + w_logA_TV * L_logA_TV 
        + float(w_period) * L_period
        + float(w_teacher) * L_teacher
    )

    # 8)
    for val in (
        L,
        L_pde,
        L_smh,
        L_geom,
        L_Aend,
        L_rad,
        L_mstft,
        L_mstft_full if torch.is_tensor(L_mstft_full) else torch.tensor(float(L_mstft_full), device=device),
        L_env,
        L_ic,
        L_tau,
        L_amp,
        L_time,
    ):
        if not torch.isfinite(val):
            zeta_val = float(zeta.detach().cpu().item()) if (isinstance(zeta, torch.Tensor)) else 0.0
            return {
                "L": float("nan"), "L_pde": float("nan"), "L_smh": float("nan"),
                "L_geom": float("nan"), "L_Aend": float("nan"),
                "L_rad": float("nan"), "L_zeta": float("nan"),
                "L_mstft": float("nan"), "L_mstft_full": float("nan"),
                "L_env": float("nan"), "L_ic": float("nan"), "L_tau": float("nan"),
                "L_amp": float("nan"), "L_time": float("nan"),
                "L_form": float(L_form.detach().cpu().item()) if torch.is_tensor(L_form) else float(L_form),
                "L_henv": float(L_henv.detach().cpu().item()) if torch.is_tensor(L_henv) else float(L_henv),
                "L_form_smooth": float(L_form_smooth),
                "L_form_smooth_raw": float(L_form_smooth_raw),
                "L_gain_soft": float(L_gain_soft) if not torch.is_tensor(L_gain_soft) else float(L_gain_soft.detach().cpu().item()),
                "tau_sec": float(tau.detach().cpu().item()),
                "p_gain": float(net.p_gain.detach().cpu().item()),
                "zeta": zeta_val,
                "L_mstft_ddsp": float("nan"),
                "L_env_ddsp": float("nan"),
                "L_glot": float("nan"),
                "g_tau": 0.0,
                "g_zeta": 0.0,
                "aux_gamma": float(aux_gamma),
            }

    # 9)
    L.backward()
    g_tau = float(getattr(net, "t_shift_raw").grad.detach().abs().item()) if getattr(net, "t_shift_raw").grad is not None else 0.0
    g_zeta = 0.0
    if hasattr(net, "rad_zeta_raw") and getattr(net, "rad_zeta_raw").grad is not None:
        g_zeta = float(getattr(net, "rad_zeta_raw").grad.detach().abs().item())
    if hasattr(net, "rad_zeta0_raw") and getattr(net, "rad_zeta0_raw").grad is not None:
        g_zeta += float(getattr(net, "rad_zeta0_raw").grad.detach().abs().item())
    if hasattr(net, "rad_zeta1_raw") and getattr(net, "rad_zeta1_raw").grad is not None:
        g_zeta += float(getattr(net, "rad_zeta1_raw").grad.detach().abs().item())

    # /
    finite = True
    for g in opt.param_groups:
        for p in g["params"]:
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                finite = False
                break
        if not finite:
            break
    if not finite:
        opt.zero_grad(set_to_none=True)
        zeta_val = float(zeta.detach().cpu().item()) if (isinstance(zeta, torch.Tensor)) else 0.0
        return {
            "L": float("nan"), "L_pde": float("nan"), "L_smh": float("nan"),
            "L_geom": float("nan"), "L_Aend": float("nan"),
            "L_rad": float("nan"), "L_zeta": float("nan"),
            "L_mstft": float("nan"), "L_mstft_full": float("nan"),
            "L_env": float("nan"), "L_ic": float("nan"), "L_tau": float("nan"),
            "L_amp": float("nan"), "L_time": float("nan"),
            "tau_sec": float(tau.detach().cpu().item()),
            "p_gain": float(net.p_gain.detach().cpu().item()),
            "zeta": zeta_val,
            "L_mstft_ddsp": float("nan"),
            "L_env_ddsp": float("nan"),
            "L_gain_soft": float(L_gain_soft) if not torch.is_tensor(L_gain_soft) else float(L_gain_soft.detach().cpu().item()),
            "L_form": 0.0, "L_henv": 0.0, "L_form_smooth": 0.0, "L_form_smooth_raw": 0.0,
            "L_glot": float("nan"),
            "g_tau": 0.0, "g_zeta": 0.0,
            "aux_gamma": float(aux_gamma),
        }

    # Implementation note.
    _all_params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    if _all_params:
        torch.nn.utils.clip_grad_norm_(_all_params, max_norm=MAX_GRAD_NORM)

    opt.step()

    zeta_val = float(zeta.detach().cpu().item()) if (isinstance(zeta, torch.Tensor)) else 0.0

    return {
        "L": float(L.item()),
        "L_pde": float(L_pde.item()),
        "L_smh": float(L_smh.item()),
        "L_geom": float(L_geom.item()),
        "L_Aend": float(L_Aend.item()),
        "L_rad": float(L_rad.item()), 
        "L_zeta": float(L_zeta.item()) if torch.is_tensor(L_zeta) else float(L_zeta),
        "L_mstft": float(L_mstft.item()),
        "L_mstft_full": (float(L_mstft_full.item()) if torch.is_tensor(L_mstft_full) else float(L_mstft_full)),
        "L_env": float(L_env.item()),
        "L_ic": float(L_ic.item()),
        "L_tau": float(L_tau.item()),
        "L_amp": float(L_amp.item()),
        "L_time": float(L_time.item()),
        "L_period": float(L_period.item() if torch.is_tensor(L_period) else L_period),
        "L_teacher": float(L_teacher.item() if torch.is_tensor(L_teacher) else L_teacher),
        "tau_sec": float(tau.detach().cpu().item()),
        "p_gain": float(net.p_gain.detach().cpu().item()),
        "zeta": zeta_val,
        "g_tau": g_tau,
        "g_zeta": g_zeta,
        "L_form": float(L_form.detach().cpu().item()) if torch.is_tensor(L_form) else float(L_form),
        "L_henv": float(L_henv.detach().cpu().item()) if torch.is_tensor(L_henv) else float(L_henv),
        "L_form_smooth": float(L_form_smooth),
        "L_form_smooth_raw": float(L_form_smooth_raw),
        "L_mstft_ddsp": (float(L_mstft_ddsp.item()) if L_mstft_ddsp is not None else float("nan")),
        "L_env_ddsp":   (float(L_env_ddsp.item())   if L_env_ddsp   is not None else float("nan")),
        "L_glot": float(L_glot.item()),
        "L_source_weak": float(L_source_weak.item()) if torch.is_tensor(L_source_weak) else float(L_source_weak),
        "L_logA_TV": float(L_logA_TV.item()),
        "aux_gamma": float(aux_gamma),
    }
