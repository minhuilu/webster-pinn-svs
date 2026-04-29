# GitHub Pages Demo Manifest

The public repository should use two branches:

- `main`: clean reproducibility code for the ICASSP 2026 paper.
- `gh-pages`: static audio-demo website only.

## Files to Keep on `gh-pages`

The existing audio demo should remain on the `gh-pages` branch with this structure:

```text
README.md                         # optional short demo README
assets/IndependentFDTD.jpeg
assets/TrainingValidationWorkflow.jpeg
assets/finalsolver.jpeg
assets/a/ddsp.wav
assets/a/ddsp_spec.png
assets/a/ingraph.wav
assets/a/pinn_post_spec.png
assets/a/post.wav
assets/a/ref.wav
assets/a/spec_ref.png
assets/i/ddsp.wav
assets/i/ddsp_spec.png
assets/i/ingraph.wav
assets/i/pinn_post_spec.png
assets/i/post.wav
assets/i/ref.wav
assets/i/spec_ref.png
assets/u/ddsp.wav
assets/u/ddsp_spec.png
assets/u/ingraph.wav
assets/u/pinn_post_spec.png
assets/u/post.wav
assets/u/ref.wav
assets/u/spec_ref.png
docs/audio.html
```

## Pages Settings

In GitHub repository settings, configure Pages as:

```text
Source: Deploy from a branch
Branch: gh-pages
Folder: / (root)
```

The demo URL can remain:

```text
https://minhuilu.github.io/webster-pinn-svs/docs/audio.html
```

The `main` branch README should link to this URL instead of carrying the website source.
