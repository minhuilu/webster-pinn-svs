# ICASSP 2026 Public Reproducibility Selection

This list defines what should be committed to GitHub for the paper release and what should stay local.

## Public Core Code

### `exp/A_static/`
Core sustained-vowel Webster PINN implementation used by the paper.

- `synthesize_ref.py`: synthetic /a/, /i/, /u/ reference generation and independent FDTD-Webster solver.
- `train_with_audio.py`: DualNet, area/radiation parameterization, PDE/BC residuals, and one-epoch training logic.
- `run_train.py`: main sustained-vowel training/evaluation entrypoint.
- `audio_forward.py`: lip-pressure/audio forward evaluation helpers.
- `audio_losses.py`: multi-STFT, log-mel envelope, LSD, and alignment losses.
- `audio_bc.py`, `audio_psbc.py`: audio boundary-condition helpers.
- `phys_consts.py`: shared physical constants.
- `plot_train_csv.py`: training-curve utility.

### `exp/B_probes/`
Probe code for formant and harmonic-envelope measurements used in training/evaluation.

### `exp/C_ddsp/`
Minimal DDSP harmonic baseline and helper networks used for the paper baseline/stabilizer.

### `exp/common/`
Speaker-profile and vocal-tract length helpers shared by the paper code.

### `tools/`
Only paper-result reproduction/export tools are public:

- `pack_results.py`: exports paper audio, figures, and metrics.
- `post_render_eval.py`: independent post-render evaluation from recovered area/radiation.
- `export_artifacts.py`: exports area curves and related artifacts.
- `plot_area_overlay.py`, `plot_robust_figs.py`: paper-style figures.
- `hnr_voicing_eval.py`: HNR/voicing evaluation.
- `summarize_sA.py`, `summarize_c3.py`: summaries for full evaluation CSVs generated after rerunning the static PINN and DDSP experiments.

## Public Data/Outputs

- `data/synthetic/vowels/`: compact synthetic references and f0 trajectories for /a/, /i/, /u/.
- `audio/`: exported demo audio for DDSP, in-graph PINN, and post-render PINN.
- `figs/`: compact spectrogram and robustness figures.
- `metrics/`: final CSV summaries used to check paper tables.
- `doc/resources/figures/`: selected release figures used for visual context. Manuscript source, PDF drafts, bibliography files, and submission material stay local.


## Suggested Public Commands

Generate synthetic sustained-vowel references:

```bash
python -m exp.A_static.synthesize_ref
```

Run the main static-vowel training entrypoint:

```bash
TAGS=a,i,u MODE=main RID=main python -m exp.A_static.run_train
```

Full training creates local `ckps/` directories that are ignored by Git. Export compact paper results after checkpoints are available locally:

```bash
python tools/pack_results.py main --rid main
python tools/pack_results.py robust --rid main
```
