# Diagnostics

Official-safe diagnostic reports live here. These reports summarize local
Training Set and Internal Validation Split behavior without adding external
weather images or treating manual review notes as authoritative labels.

Generated reports may depend on ignored Experiment Run artifacts under
`results/runs/`. Keep report inputs reproducible through documented `just`
recipes and keep bulky generated assets out of this directory unless they are
needed for durable review.

Image feature diagnostics require an active runtime with `opencv-python-headless`
installed, such as the platform or CUDA training dependency set.

Current reports:

- `efficientnet-b0-stable-error-pool.*`: fixed-split EfficientNet-B0 error
  overlap across three training seeds.
- `efficientnet-b0-val-image-features.*`: image statistics and dHash
  duplicate diagnostics for the fixed-split EfficientNet-B0 validation set.
- `train-image-features.*`: image statistics and dHash duplicate diagnostics
  for the full Training Set.
- `split42-duplicate-leakage.*`: exact and near-duplicate dHash group leakage
  across the fixed Internal Validation Split.
- `efficientnet-b0-logit-bias.*`: in-split class-wise logit bias diagnostic
  across fixed-split EfficientNet-B0 runs.
