# PINN–DDSP Training Log Cheatsheet

| Log Key | Full Name | Meaning & Research Insight |
|---------|-----------|----------------------------|
| **L** | Total Loss | Overall training objective. Should steadily decrease. Spikes = NaN/grad issues. |
| **L_pde** | PDE Residual | Measures how well the network satisfies Webster"s equation. Low -> physics consistency. |
| **L_smh** | Cross-Section Smoothness | Penalizes oscillations in A(x,t). Low -> stable tract shape. |
| **L_geom** | Geometric Barrier | Keeps A(x,t) within valid range. Low -> physically plausible geometry. |
| **L_Aend** | Endpoint Anchor | Forces A(0)=A(1)=1. Ensures boundary stability. |
| **L_rad** | Robin Radiation Loss | Error at lip boundary (Robin condition). Low -> realistic mouth radiation. |
| **L_zeta** | Radiation Regularizer | Prevents ζ -> 0 or ∞. Keeps damping in a plausible range. |
| **L_mstft** | Windowed Multi-STFT Loss | Local timbre match. Low -> short segments sound correct. |
| **L_env** | Log-Mel Envelope Loss | Matches overall spectral envelope. Important for timbre quality. |
| **L_mstft_full** | Global Multi-STFT (weak) | Full utterance match. High values -> drift or time misalignment. |
| **L_form** | Formant Trajectory Loss | Penalizes mismatch in F1–F3 tracks. Low -> correct resonances. |
| **L_henv** | Harmonic Envelope Loss | Matches harmonic amplitudes. Low -> better spectral detail. |
| **L_form_smooth** | Formant Smoothing | Keeps formant trajectories smooth. High -> jittery formants. |
| **L_amp** | Amplitude Match | Controls loudness scaling. Helps reduce mSTFT raw error. |
| **L_ic** | Initial Condition Loss | Keeps initial waveform condition valid. Auxiliary stabilizer. |
| **L_tau** | Time-Shift Regularizer | Prevents τ (alignment shift) from drifting. |
| **p_gain** | Gain Factor | Should stay ~1.0–1.4. >1.6 -> model compensating with brute gain. |
| **tau_sec** | Time Shift (s) | Normally within +/-50 ms. Larger -> alignment problem. |
| **zeta** | Radiation Coefficient | Normal range 0.1–0.2. Too high/low -> unrealistic boundary damping. |
| **g_tau, g_zeta** | Gradient Magnitudes | Monitor training health. Exploding/vanishing gradients show here. |
| **L_mstft_ddsp** | DDSP mSTFT Loss | Timbre match of DDSP-synthesized audio. |
| **L_env_ddsp** | DDSP Envelope Loss | Envelope match of DDSP audio. |
| **mSTFT(raw)** | Global mSTFT (unaligned) | Core benchmark metric. Lower -> better timbre fidelity. |
| **LSD(raw)** | Log-Spectral Distance (unaligned) | Lower -> closer spectral match. |
| **mSTFT(aln)** | Global mSTFT (aligned) | Removes timing offset. Better for fair comparison. |
| **LSD(aln)** | Log-Spectral Distance (aligned) | Like above, but spectral error. |
| **F1/F2/F3_MAE** | Formant MAE (Hz) | Lower -> more accurate resonance control. Key interpretability metric. |
| **HNR** | Harmonics-to-Noise Ratio (dB) | Higher -> cleaner, harmonic-rich sound. Drop -> noisy artifacts. |

---

📌 **Usage Tips**
- **Early training** -> Watch `L_pde`, `L_rad`, `L_geom`, `L_mstft` to ensure physics + audio start converging.  
- **Later training** -> Focus on `mSTFT(raw)`, `LSD(raw)`, and `HNR` for benchmark reporting.  
- **Baselines vs Main** -> Compare primarily on `mSTFT(raw)` + `LSD(raw)`, check `F1–F3 MAE` for interpretability advantage.  
