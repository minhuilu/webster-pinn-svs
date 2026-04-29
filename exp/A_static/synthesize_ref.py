# file: exp/A_static/synthesize_ref.py
# FDTD-Webster forward synthesizer
import numpy as np
from scipy.signal import spectrogram, resample_poly
import soundfile as sf
import os, sys, pathlib
import matplotlib.pyplot as plt

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from exp.common.singer import get_profile

# ---------------------------
# 1) Simple but plausible area functions
# ---------------------------
def area_a(x):  # /a/:mid tract is expanded and the mouth end is slightly narrowed
    return 1.0 + 0.6*np.exp(-((x-0.50)/0.20)**2) - 0.10*np.exp(-((x-0.90)/0.10)**2)

def area_i(x):  # /i/:strong front constriction near the mouth, with a slight posterior bulge
    base = 1.10
    constr_front = 0.55*np.exp(-((x-0.85)/0.06)**2)   # constriction is represented by subtraction
    bulge_back   = 0.20*np.exp(-((x-0.35)/0.20)**2)   # small posterior expansion
    return np.clip(base - constr_front + bulge_back, 0.3, 2.5)

def area_u(x):  # /u/:rounded lips with a narrow mouth end and mild posterior expansion
    base = 1.05
    lip_constr = 0.70*np.exp(-((x-0.97)/0.03)**2)
    back_bulge = 0.25*np.exp(-((x-0.45)/0.25)**2)
    return np.clip(base - lip_constr + back_bulge, 0.3, 2.5)


# ---------------------------
# 2) Rosenberg glottal(,)
# ---------------------------
def glottal_flow_rosenberg(f0, dur, sr, Oq=0.6, Cq=0.3, amp=1.0, noise_db=-30):
    """
Return U_g(t):
      - Oq: open quotient (0.4~0.7) open quotient
      - Cq: close quotient (0.2~0.5) closed-phase proportion of the open segment; controls closing speed
    Classical Rosenberg-C approximation: 0..Ta opening, Ta..Te closing, closed elsewhere.
    """
    T0 = 1.0 / float(f0)
    N  = int(round(dur * sr))
    t  = np.arange(N) / sr
    # Segment lengths within each period
    To = Oq * T0                 # open phase
    Ta = 0.6 * To                # opening time
    Te = To                      # end of open phase
    Tc = Cq * To                 # closing segment duration, controlling closing speed

    U = np.zeros(N, dtype=np.float32)
    for n in range(N):
        tp = t[n] % T0           # time within the period
        if tp < Ta:
            # 0..Ta: 0 -> 1 smooth rise
            U[n] = 0.5 * (1 - np.cos(np.pi * tp / Ta))
        elif tp < Te:
            # Ta..Te: smoothly decay from 1 to 0
            r = (tp - Ta) / (Te - Ta + 1e-12)
            U[n] = (np.cos(np.pi * r / 2))**2
        else:
            U[n] = 0.0

    # Faster closing phase: adjust the shape via Cq by shortening the final decay
    if Cq < 0.5:
        k = max(1, int((1.0 - 2.0*Cq) * 6))  # simple tail sharpening
        U = np.power(U, 1.0 + 0.25*k)

    # Add a little aspiration noise
    rms = np.sqrt(np.mean(U**2)) + 1e-8
    noise = np.random.randn(N).astype(np.float32)
    noise *= (10.0**(noise_db/20.0)) * rms / (np.sqrt(np.mean(noise**2)) + 1e-8)

    U = amp * (U + noise)
    # Remove DC and normalize
    U = U - U.mean()
    U = U / (np.max(np.abs(U)) + 1e-8)
    return U.astype(np.float32)

# ---------------------------
# 3) Webster Explicit + Robin
# ---------------------------
def webster_1d_fd(
    Ax_fn, f0=140.0, dur=1.0,
    sr_out=16000,
    c=340.0, rho=1.2,
    L=0.155, Nx=256,
    cfl_max=0.8,
    beta=10.0,            # linear damping; keep moderate
    zeta_ref=0.06,        # Robin radiation coefficient, matching the training form
    Oq=0.6, Cq=0.3,        # Rosenberg source shape
    noise_db=-28.0

):
    """
Explicit + Robin () + () + -beta*psi_t.
    Simulate internally at a high sample rate, then resample_poly to sr_out.
    """
    dx = L / (Nx - 1)
    sr_min = int(np.ceil(c / (cfl_max * dx)))
    sr_sim = max(sr_min, sr_out)     # simulation sample rate
    Nt = int(np.round(dur * sr_sim))
    dt = 1.0 / sr_sim

    x = np.linspace(0, L, Nx)
    A = Ax_fn(x).astype(np.float32)
    A_mid = 0.5*(A[1:] + A[:-1])

    # glottal()->
    Ug = glottal_flow_rosenberg(f0, dur, sr_sim, Oq=Oq, Cq=Cq, amp=1.0, noise_db=noise_db)

    psi   = np.zeros(Nx, dtype=np.float32)   # current
    psi_p = np.zeros(Nx, dtype=np.float32)   # previous frame
    p_lip = np.zeros(Nt, dtype=np.float32)

    # Smooth fade in/out to avoid transients
    ramp = int(0.01 * sr_sim)
    if ramp > 0:
        r = 0.5*(1 - np.cos(np.pi*np.arange(ramp)/max(1,ramp)))
        Ug[:ramp] *= r
        Ug[-ramp:] *= r[::-1]

    for n in range(Nt):
        # Interior first derivative and flux
        dpsi_dx = (psi[1:] - psi[:-1]) / dx
        F = A_mid * dpsi_dx

        # Left glottal boundary specifies volume flow:A(0)*psi_x = rho*c*Ug
        F[0] = rho * c * Ug[n]

        # Right mouth-end Robin boundary:psi_x + zeta_ref*psi_t = 0
        # Discretize as flux:F[-1] = A_mid[-1]*psi_x(L-) ~ A(L)*psi_x(L)
        # First update psi_n with the interior formula, then correct psi_n[-1] with Robin after the full-domain update
        dF_dx = np.zeros_like(psi)
        dF_dx[1:-1] = (F[1:] - F[:-1]) / dx

        psi_n = np.empty_like(psi)
        # Interior update(including damping term -beta*psi_t ~= -beta*(psi - psi_p)/dt)
        psi_n[1:-1] = (2*psi[1:-1] - psi_p[1:-1]
                       + (c**2 * dt**2) * (dF_dx[1:-1] / (A[1:-1] + 1e-8))
                       - beta*dt * (psi[1:-1] - psi_p[1:-1]))

        # Left boundary: approximate Neumann; flux is already set to avoid numerical jitter
        psi_n[0] = psi_n[1]

        # Right boundary: substitute the Robin boundary solution for psi_n[-1]
        # (psi_N - psi_{N-1})/dx + zeta*(psi_N^n - psi_N^{n-1})/dt = 0
        # => psi_N^n * (1/dx + zeta/dt) = psi_{N-1}^n/dx + zeta*psi_N^{n-1}/dt
        denom = (1.0/dx + zeta_ref/dt)
        rhs   = (psi_n[-2]/dx + zeta_ref * psi[-1]/dt)
        psi_n[-1] = rhs / (denom + 1e-12)

        # Lip pressure:p = -rho * psi_t(L,t) ~= -rho * (psi_n[-1] - psi[-1]) / dt
        p_lip[n] = -rho * (psi_n[-1] - psi[-1]) / dt

        psi_p, psi = psi, psi_n

    # Remove DC
    p_lip -= np.mean(p_lip)

    # Anti-aliased resampling to sr_out
    from math import gcd
    up, down = sr_out, sr_sim
    g = gcd(up, down); up//=g; down//=g
    p_out = resample_poly(p_lip, up, down)

    Nt_out = int(np.round(dur * sr_out))
    if len(p_out) != Nt_out:
        # Correct length
        idx = np.linspace(0, len(p_out)-1, Nt_out)
        p_out = np.interp(idx, np.arange(len(p_out)), p_out)

    t_out = np.arange(Nt_out) / sr_out
    return p_out.astype(np.float32), t_out

# ---------------------------
# 4) optional:Spectrum(requires)
# ---------------------------
def plot_spectrograms(tag, audio, sr, f0, out_root="data/synthetic/vowels"):
    img_dir = f"{out_root}/{tag}/img"
    os.makedirs(img_dir, exist_ok=True)

    start_idx = int(len(audio) * 0.3)
    end_idx   = int(len(audio) * 0.7)
    seg = audio[start_idx:end_idx]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    # Waveform
    t = np.arange(len(seg)) / sr
    axes[0].plot(t, seg); axes[0].set_title("Waveform"); axes[0].grid(True)
    # Spectrum
    n_fft = min(4096, len(seg))
    freq = np.fft.rfftfreq(n_fft, 1/sr)
    fftv = np.abs(np.fft.rfft(seg, n=n_fft)) + 1e-12
    axes[1].semilogy(freq, fftv)
    axes[1].set_xlim(0, 5000); axes[1].set_title("Spectrum (log)")
    for k in range(1, 12):
        h = k * f0
        if h < 5000: axes[1].axvline(h, ls='--', alpha=0.4)
    # Spectrogram
    f, tt, Sxx = spectrogram(seg, fs=sr, nperseg=512, noverlap=256)
    im = axes[2].pcolormesh(tt, f, 10*np.log10(Sxx + 1e-12), shading='gouraud')
    axes[2].set_ylim(0, 5000); axes[2].set_title("Spectrogram"); fig.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig(f"{img_dir}/{tag}_spectral_analysis.png", dpi=140)
    plt.close()

# ---------------------------
# 5) Write reference pair: wav plus f0.npy
# ---------------------------
def write_pair(tag, Ax_fn, f0=None, sr_out=16000, dur=0.8, out_root="data/synthetic/vowels", speaker="female",
               zeta_ref=0.06, Oq=0.6, Cq=0.3, beta=10.0, make_plot=True):
    prof = get_profile(speaker)
    if f0 is None:
        f0 = float(prof["f0_by_vowel"][tag])

    p, t = webster_1d_fd(
        Ax_fn, f0=f0, dur=dur, sr_out=sr_out, L=prof["L"],
        beta=beta, zeta_ref=zeta_ref, Oq=Oq, Cq=Cq
    )
    wav = 0.9 * p / (np.max(np.abs(p)) + 1e-8)

    os.makedirs(f"{out_root}/{tag}/wav", exist_ok=True)
    os.makedirs(f"{out_root}/{tag}/f0",  exist_ok=True)
    sf.write(f"{out_root}/{tag}/wav/{tag}.wav", wav.astype(np.float32), sr_out)
    np.save(f"{out_root}/{tag}/f0/{tag}.npy", np.full_like(wav, f0, dtype=np.float32))

    if make_plot:
        plot_spectrograms(tag, wav, sr_out, f0, out_root)

    print(f"[{tag}] saved -> {out_root}/{tag}/wav/{tag}.wav  (f0={f0} Hz, zeta={zeta_ref}, Oq={Oq}, Cq={Cq}, beta={beta})")
    return wav, sr_out

if __name__ == "__main__":
    # You can tune zeta_ref, Oq, and Cq to hear timbral changes
    write_pair("a", area_a, sr_out=16000, speaker="female", zeta_ref=0.06, Oq=0.58, Cq=0.35, beta=10.0)
    write_pair("i", area_i, sr_out=16000, speaker="female", zeta_ref=0.06, Oq=0.62, Cq=0.30, beta=10.0)
    write_pair("u", area_u, sr_out=16000, speaker="female", zeta_ref=0.06, Oq=0.60, Cq=0.35, beta=10.0)
    print("Done.")
