# TTA Diagnostic

Horizontal-flip test-time augmentation evaluated on an Experiment Run's Internal Validation Split.

- Run: `efficientnet-b0-exactdhash-split42-train42-bs32-w4`
- Artifact: `results/runs/efficientnet-b0-exactdhash-split42-train42-bs32-w4/model.pth`
- Validation predictions: `results/runs/efficientnet-b0-exactdhash-split42-train42-bs32-w4/val_predictions.csv`
- Summary CSV: `docs/diagnostics/efficientnet-b0-exactdhash-hflip-tta-summary.csv`
- Changes CSV: `docs/diagnostics/efficientnet-b0-exactdhash-hflip-tta-changes.csv`
- Device: `cpu`
- AMP enabled: `False`

## Summary

| Source | Macro F1 | Accuracy | Delta vs CSV | Seconds |
| --- | ---: | ---: | ---: | ---: |
| `csv_predictions` | 0.9361 | 0.9364 | 0.0000 | 0.00 |
| `single_pass_eval` | 0.9361 | 0.9364 | 0.0000 | 14.37 |
| `hflip_tta` | 0.9279 | 0.9314 | -0.0082 | 29.06 |

## TTA Changes vs CSV Predictions

- Changed: `5`
- Corrected: `0`
- Regressed: `5`
- Wrong-to-wrong: `0`

| Label | CSV F1 | TTA F1 | Delta |
| --- | ---: | ---: | ---: |
| `cloudy` | 0.9345 | 0.9312 | -0.0033 |
| `rainy` | 0.9017 | 0.8824 | -0.0194 |
| `snowy` | 0.9682 | 0.9620 | -0.0061 |
| `sunny` | 0.9399 | 0.9361 | -0.0038 |

## Interpretation Boundary

- This is an in-split diagnostic for a fixed validation split.
- Treat positive TTA deltas as candidates for cross-run validation, not as a guaranteed Submission Prediction Interface improvement.
