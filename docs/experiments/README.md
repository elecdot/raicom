# Experiments

Manual review log for notable Experiment Runs. Keep this file as a decision index; use each run's `metadata.json` and `metrics.json` for full machine-readable details.

## Review Workflow

Use `just audit-errors <run-id>` to create a compact high-confidence validation
error audit pack under `results/runs/<run-id>/audit/`. Add `--contact-sheet`
when visual review is useful in a runtime with OpenCV installed.

When comparing error overlap across runs, keep `--split-seed` fixed and vary
`--train-seed`. Runs that vary `--seed` alone usually change both the training
randomness and the Internal Validation Split, so low path overlap between their
audit packs is expected.

Keep this README as the durable decision index. Use notebooks only for ad hoc
visual exploration or plots that are hard to review in plain text, and pair any
durable notebook with Jupytext.

## EfficientNet-B0

Shared recipe unless noted: pretrained TorchVision `efficientnet_b0`, mild augmentation, stratified split, inverse-sqrt class weights, label smoothing 0.05, AdamW, cosine scheduler, CUDA AMP.

### Seed Sweep

Conservative loader shape: batch size 32, no DataLoader workers.

| Run ID | Seed | Best Macro F1 | Best Epoch | Artifact | Decision |
| --- | --- | --- | --- | --- | --- |
| `20260626-175735-efficientnet_b0-seed42` | 42 | 0.9302 | 24 | `results/runs/20260626-175735-efficientnet_b0-seed42/model.pth` | Keep as first strong baseline evidence. |
| `efficientnet-b0-seed2025` | 2025 | 0.9161 | 3 | `results/runs/efficientnet-b0-seed2025/model.pth` | Confirms lower-bound seed variance. |
| `efficientnet-b0-seed3407` | 3407 | 0.9326 | 23 | `results/runs/efficientnet-b0-seed3407/model.pth` | Strongest conservative-loader seed sweep run. |

### Loader And Batch Ablation

Seed 42 only. This group compares local throughput settings, not model design.

| Run ID | Batch | Workers | Persistent | Best Macro F1 | Best Epoch | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `efficientnet-b0-seed42-bs32-w4` | 32 | 4 | yes | 0.9392 | 14 | Preferred local throughput setting for B0. |
| `efficientnet-b0-seed42-bs64-w4` | 64 | 4 | yes | 0.9322 | 13 | Keep as throughput reference; do not promote over bs32/w4. |
| `efficientnet-b0-seed42-bs96-w4` | 96 | 4 | no | 0.9217 | 6 | Reject as current B0 training setting. |

### Split Strategy Check

Seed 42 only. This group compares Internal Validation Split construction, not
model design.

| Run ID | Split Strategy | Val Rows | Best Macro F1 | Best Epoch | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `efficientnet-b0-seed42-bs32-w4` | `stratified_shuffle` | 1000 | 0.9392 | 14 | Keep as historical B0 baseline. |
| `efficientnet-b0-exactdhash-split42-train42-bs32-w4` | `exact_dhash_group` | 991 | 0.9361 | 15 | Keep as the more duplicate-aware B0 validation check. |

The exact dHash group-aware split removed exact dHash duplicate leakage by
grouping 174 exact dHash collision groups before assigning train/validation
membership. Its lower Macro F1 suggests the historical split may be mildly
optimistic, but the result is close enough that the B0 baseline remains
credible.

### Augmentation Ablation

Seed 42 only. This group compares training augmentation recipes on the exact
dHash group-aware split.

| Run ID | Augmentation | Val Rows | Best Macro F1 | Best Epoch | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `efficientnet-b0-exactdhash-split42-train42-bs32-w4` | `mild` | 991 | 0.9361 | 15 | Keep as the duplicate-aware B0 reference. |
| `efficientnet-b0-exactdhash-targeted-split42-train42-bs32-w4` | `targeted` | 991 | 0.9177 | 6 | Reject current targeted recipe. |
| `efficientnet-b0-exactdhash-stronger-split42-train42-bs32-w4` | `stronger` | 991 | 0.9169 | 16 | Reject current stronger recipe. |

Both stronger recipes used the same validation paths as the mild exact-dHash
run and underperformed. `targeted` corrected 16 mild-run errors but introduced
27 new errors; `stronger` corrected 15 mild-run errors but introduced 25 new
errors. Stop increasing augmentation strength on B0 for now and keep `mild` as
the training reference.

### Objective And Weighting Ablation

Seed 42 only. This group compares loss-side settings on the exact dHash
group-aware split with `mild` augmentation.

| Run ID | Class Weights | Label Smoothing | Best Macro F1 | Best Epoch | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `efficientnet-b0-exactdhash-split42-train42-bs32-w4` | `inverse-sqrt` | 0.05 | 0.9361 | 15 | Keep as the duplicate-aware B0 reference. |
| `efficientnet-b0-exactdhash-ls0-split42-train42-bs32-w4` | `inverse-sqrt` | 0.00 | 0.9230 | 19 | Reject removing label smoothing. |
| `efficientnet-b0-exactdhash-cw-none-split42-train42-bs32-w4` | `none` | 0.05 | 0.9225 | 14 | Reject removing class weights. |

Removing label smoothing or class weights did not improve the minority Weather
Categories on this split. Keep inverse-sqrt class weights and label smoothing
0.05 as the B0 reference loss settings.

### Validation Error Review

The `efficientnet-b0-seed42-bs32-w4` high-confidence Validation Error Audit
was manually reviewed at `--min-confidence 0.8`. A structured snapshot is kept
at `docs/experiments/reviews/efficientnet-b0-seed42-bs32-w4-high-conf-errors.csv`
because generated audit files under `results/` are ignored and may be
regenerated.

Review label counts: `model-error` 10, `hard-example` 5, `source-artifact` 3,
`ambiguous` 2, `mislabel` 2, `hard-model` 1. Treat this Validation Error
Review as low-confidence qualitative evidence, not as Training Set relabeling
or an authoritative correction source.
