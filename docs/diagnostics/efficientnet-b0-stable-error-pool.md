# EfficientNet-B0 Stable Error Pool

Generated from comparable fixed-split EfficientNet-B0 Experiment Runs.

- CSV: `docs/diagnostics/efficientnet-b0-stable-error-pool.csv`
- Error samples: `87`

## Input Runs

| Short ID | Run | Predictions |
| --- | --- | --- |
| `seed42` | `results/runs/efficientnet-b0-seed42-bs32-w4` | `results/runs/efficientnet-b0-seed42-bs32-w4/val_predictions.csv` |
| `train2025` | `results/runs/efficientnet-b0-split42-train2025-bs32-w4` | `results/runs/efficientnet-b0-split42-train2025-bs32-w4/val_predictions.csv` |
| `train3407` | `results/runs/efficientnet-b0-split42-train3407-bs32-w4` | `results/runs/efficientnet-b0-split42-train3407-bs32-w4/val_predictions.csv` |

## Stability Tiers

| Tier | Samples |
| --- | ---: |
| `1/3` | 35 |
| `3/3` | 27 |
| `2/3` | 25 |

## Diagnostic Categories

| Category | Samples |
| --- | ---: |
| `random_sensitive` | 35 |
| `boundary_case` | 20 |
| `stable_high_conf_error` | 18 |
| `stable_model_error` | 9 |
| `low_priority` | 5 |

## Error Pairs

| Error Pair | Samples |
| --- | ---: |
| `cloudy->sunny` | 23 |
| `sunny->cloudy` | 21 |
| `rainy->cloudy` | 10 |
| `cloudy->rainy` | 9 |
| `cloudy->snowy` | 8 |
| `snowy->cloudy` | 5 |
| `snowy->sunny` | 4 |
| `snowy->rainy` | 3 |
| `rainy->sunny` | 2 |
| `rainy->snowy` | 2 |
| `sunny->rainy` | 2 |
| `sunny->snowy` | 1 |

## Stable High-Confidence Errors

| Mean Confidence | Error Count | Error Pairs | Path |
| ---: | ---: | --- | --- |
| 0.9578 | 3 | `cloudy->sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00393.jpg` |
| 0.9443 | 3 | `rainy->cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00408.jpg` |
| 0.9312 | 3 | `cloudy->sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00403.jpg` |
| 0.9268 | 3 | `cloudy->sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00626.jpg` |
| 0.9215 | 3 | `sunny->cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00219.jpg` |
| 0.9199 | 3 | `cloudy->rainy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00764.jpg` |
| 0.9163 | 3 | `sunny->cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01467.jpg` |
| 0.9054 | 3 | `rainy->cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00143.jpg` |
| 0.9046 | 3 | `rainy->cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00220.jpg` |
| 0.9030 | 3 | `cloudy->sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00113.jpg` |

## Related Evidence

- Manual review snapshot: `docs/experiments/reviews/efficientnet-b0-seed42-bs32-w4-high-conf-errors.csv`
- The manual review snapshot is not used for automatic categories.
