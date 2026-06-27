# Diagnostics

Official-safe diagnostic reports live here. These reports summarize local
Training Set and Internal Validation Split behavior without adding external
weather images or treating manual review notes as authoritative labels.

Generated reports may depend on ignored Experiment Run artifacts under
`results/runs/`. Keep report inputs reproducible through documented `just`
recipes and keep bulky generated assets out of this directory unless they are
needed for durable review.

Use `just compare-runs` for ad hoc Experiment Run comparisons. Save generated
comparison reports here only when the comparison is durable evidence for a
decision.

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
- `efficientnet-b0-exactdhash-split42-duplicate-leakage.*`: duplicate leakage
  check for the B0 exact dHash group-aware split run.
- `efficientnet-b0-logit-bias.*`: in-split class-wise logit bias diagnostic
  across fixed-split EfficientNet-B0 runs.
- `efficientnet-b0-vs-b3-exactdhash-split42.*`: exact dHash group-aware
  EfficientNet-B0 vs EfficientNet-B3 validation comparison.
- `efficientnet-b0-b3-exactdhash-logit-bias.*`: in-split class-wise logit
  bias diagnostic for exact dHash group-aware B0/B3 predictions.
- `efficientnet-b0-exactdhash-hflip-tta.*`: horizontal-flip test-time
  augmentation diagnostic for the B0 exact dHash group-aware run.
