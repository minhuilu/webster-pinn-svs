# Learning Vocal-Tract Area and Radiation with a Physics-Informed Webster Model

This repository contains the public reproducibility code for the ICASSP 2026 paper.

The release focuses on the paper's sustained-vowel setting: synthetic /a/, /i/, /u/ references, a physics-informed Webster PINN with learned vocal-tract area and radiation, a compact DDSP baseline/stabilizer, independent FDTD-Webster post-rendering, and the metric/figure export scripts used to check the reported results.

## Audio Demo

Interactive listening examples are available at:

<https://minhuilu.github.io/webster-pinn-svs/docs/audio.html>

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

## Quick Orientation

- `exp/A_static/`: main Webster PINN and independent FDTD reference/post-render code.
- `exp/B_probes/`: formant and harmonic-envelope probes.
- `exp/C_ddsp/`: DDSP harmonic baseline/stabilizer.
- `tools/`: result export, post-render evaluation, and plotting scripts.
- `data/synthetic/vowels/`: compact synthetic references for /a/, /i/, /u/.
- `metrics/`, `audio/`, `figs/`: compact exported outputs for checking the paper results.


## Lightweight Checks

These commands check the public code path without running a full training job:

```bash
python -m compileall -q exp/A_static exp/B_probes exp/C_ddsp exp/common tools scripts
python tools/post_render_eval.py --help
python tools/pack_results.py --help
```

A minimal post-render check can be run by providing an explicit area vector and zeta file to `tools/post_render_eval.py`. Full training is intentionally left as a longer reproducibility run.

## Reproducing Results

The compact files in `metrics/`, `audio/`, and `figs/` provide the exported paper results. To regenerate them from scratch, first run the training entrypoint to create local checkpoints, then run:

```bash
python tools/pack_results.py main --rid main
python tools/pack_results.py robust --rid main
```
