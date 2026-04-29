# file: exp/A_static/audio_psbc.py
# Represent the mouth-end time series as a point-set constraint to avoid random-sampling alignment issues
import numpy as np
import deepxde as dde
import torch

def make_lip_pointset_bc(Lx, t_grid, audio_ref, rho=1.2, component_psi=0):
    """
    Build PointSetBC:X = [(Lx,t_i)], Y = [p_ref(t_i)]
    make network p_lip(t) fit audio_ref(t)
    """
    X = np.stack([np.full_like(t_grid, Lx), t_grid], axis=1).astype(np.float32)
    Y = audio_ref.astype(np.float32)[:,None]
    # PointSetBC directly supervises Y[:, component], so psi_t -> p must be represented as part of the model output
    # Option: add a virtual head_p in forward(), or use OperatorBC as in the previous approach
    # :Return psi(t) ,for p
    bc = dde.PointSetBC(X, Y, component=None)  # the corresponding term is converted to a p error inside the loss
    return bc
