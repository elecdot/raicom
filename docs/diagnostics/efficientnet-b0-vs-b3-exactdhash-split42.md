# Experiment Run Comparison

Comparison generated from Experiment Run `metadata.json`, `metrics.json`, and `val_predictions.csv` files.

Input runs:
- `results/runs/efficientnet-b0-exactdhash-split42-train42-bs32-w4`
- `results/runs/efficientnet-b3-exactdhash-split42-train42-bs32-w4`

## Run Summary

| Run | Model | Split | Aug | Class Weights | Smoothing | Val Rows | Best Macro F1 | Best Acc | Best Epoch | Final Macro F1 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet-b0-exactdhash-split42-train42-bs32-w4` | `efficientnet_b0` | `exact_dhash_group` | `mild` | `inverse-sqrt` | 0.05 | 991 | 0.9361 | 0.9364 | 15 | 0.9229 |
| `efficientnet-b3-exactdhash-split42-train42-bs32-w4` | `efficientnet_b3` | `exact_dhash_group` | `mild` | `inverse-sqrt` | 0.05 | 991 | 0.9203 | 0.9273 | 9 | 0.9187 |

## Pairwise Against First Run

| Candidate | Same Val Paths | Common | Baseline Correct / Candidate Wrong | Baseline Wrong / Candidate Correct | Both Wrong | Different Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet-b3-exactdhash-split42-train42-bs32-w4` | 1 | 991 | 27 | 18 | 45 | 45 |

## Per-Class F1 Delta Against First Run

| Candidate | Label | Baseline F1 | Candidate F1 | Delta |
| --- | --- | ---: | ---: | ---: |
| `efficientnet-b3-exactdhash-split42-train42-bs32-w4` | `cloudy` | 0.9345 | 0.9252 | -0.0093 |
| `efficientnet-b3-exactdhash-split42-train42-bs32-w4` | `rainy` | 0.9017 | 0.8927 | -0.0091 |
| `efficientnet-b3-exactdhash-split42-train42-bs32-w4` | `snowy` | 0.9682 | 0.9250 | -0.0432 |
| `efficientnet-b3-exactdhash-split42-train42-bs32-w4` | `sunny` | 0.9399 | 0.9381 | -0.0018 |
