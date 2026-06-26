# EfficientNet-B0 Logit Bias Diagnostic

Class-wise logit bias was selected on the same Internal Validation Split used for scoring each row. Treat this as an in-split diagnostic upper bound for decision calibration, not as a Submission Prediction Interface parameter.

- Summary CSV: `docs/diagnostics/efficientnet-b0-logit-bias-summary.csv`
- Changes CSV: `docs/diagnostics/efficientnet-b0-logit-bias-changes.csv`
- Fixed label: `cloudy`
- Bias range: `-1.0` to `1.0`
- Bias step: `0.05`

## Run Summary

| Run | Baseline Macro F1 | Biased Macro F1 | Delta | Bias | Changed | Corrected | Regressed | Wrong->Wrong |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `seed42` | 0.9392 | 0.9486 | 0.0094 | `cloudy:0.0000;rainy:0.1500;snowy:0.3500;sunny:-0.9000` | 19 | 12 | 7 | 0 |
| `train2025` | 0.9309 | 0.9414 | 0.0106 | `cloudy:0.0000;rainy:0.9000;snowy:-0.3500;sunny:0.0000` | 11 | 8 | 3 | 0 |
| `train3407` | 0.9278 | 0.9376 | 0.0097 | `cloudy:0.0000;rainy:-0.2500;snowy:-1.0000;sunny:-0.6500` | 9 | 7 | 2 | 0 |

## Per-Class F1

| Run | State | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | --- | ---: | ---: | ---: | ---: |
| `seed42` | baseline | 0.9483 | 0.9205 | 0.9308 | 0.9572 |
| `seed42` | biased | 0.9522 | 0.9379 | 0.9455 | 0.9590 |
| `train2025` | baseline | 0.9475 | 0.8966 | 0.9250 | 0.9544 |
| `train2025` | biased | 0.9497 | 0.9171 | 0.9419 | 0.9570 |
| `train3407` | baseline | 0.9436 | 0.9121 | 0.9000 | 0.9556 |
| `train3407` | biased | 0.9487 | 0.9231 | 0.9231 | 0.9554 |

## Confusion Matrices: `seed42`

Baseline:
| True \ Pred | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | ---: | ---: | ---: | ---: |
| `cloudy` | 413 | 4 | 2 | 18 |
| `rainy` | 6 | 81 | 1 | 1 |
| `snowy` | 3 | 2 | 74 | 2 |
| `sunny` | 12 | 0 | 1 | 380 |

Biased:
| True \ Pred | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | ---: | ---: | ---: | ---: |
| `cloudy` | 418 | 4 | 3 | 12 |
| `rainy` | 5 | 83 | 1 | 0 |
| `snowy` | 1 | 1 | 78 | 1 |
| `sunny` | 17 | 0 | 2 | 374 |

## Confusion Matrices: `train2025`

Baseline:
| True \ Pred | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | ---: | ---: | ---: | ---: |
| `cloudy` | 415 | 4 | 2 | 16 |
| `rainy` | 7 | 78 | 2 | 2 |
| `snowy` | 3 | 2 | 74 | 2 |
| `sunny` | 14 | 1 | 1 | 377 |

Biased:
| True \ Pred | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | ---: | ---: | ---: | ---: |
| `cloudy` | 415 | 6 | 0 | 16 |
| `rainy` | 4 | 83 | 1 | 1 |
| `snowy` | 4 | 2 | 73 | 2 |
| `sunny` | 14 | 1 | 0 | 378 |

## Confusion Matrices: `train3407`

Baseline:
| True \ Pred | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | ---: | ---: | ---: | ---: |
| `cloudy` | 410 | 7 | 6 | 14 |
| `rainy` | 4 | 83 | 1 | 1 |
| `snowy` | 3 | 2 | 72 | 4 |
| `sunny` | 15 | 1 | 0 | 377 |

Biased:
| True \ Pred | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | ---: | ---: | ---: | ---: |
| `cloudy` | 416 | 6 | 3 | 12 |
| `rainy` | 4 | 84 | 0 | 1 |
| `snowy` | 3 | 2 | 72 | 4 |
| `sunny` | 17 | 1 | 0 | 375 |

## Top Corrected Rows

| Run | True | Before | After | Confidence After | Path |
| --- | --- | --- | --- | ---: | --- |
| `seed42` | `cloudy` | `sunny` | `cloudy` | 0.5647 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01140.jpg` |
| `seed42` | `cloudy` | `sunny` | `cloudy` | 0.5461 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00401.jpg` |
| `seed42` | `snowy` | `cloudy` | `snowy` | 0.5327 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00314.jpg` |
| `seed42` | `snowy` | `cloudy` | `snowy` | 0.4969 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00108.jpg` |
| `seed42` | `snowy` | `rainy` | `snowy` | 0.4918 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00285.jpg` |
| `seed42` | `cloudy` | `sunny` | `cloudy` | 0.4825 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01523.jpg` |
| `seed42` | `rainy` | `sunny` | `rainy` | 0.4730 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00049.jpg` |
| `seed42` | `cloudy` | `sunny` | `cloudy` | 0.4728 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01896.jpg` |
| `seed42` | `rainy` | `cloudy` | `rainy` | 0.4705 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00026.jpg` |
| `seed42` | `cloudy` | `sunny` | `cloudy` | 0.4489 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00770.jpg` |

## Top Regressed Rows

| Run | True | Before | After | Confidence After | Path |
| --- | --- | --- | --- | ---: | --- |
| `seed42` | `sunny` | `sunny` | `snowy` | 0.7061 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00558.jpg` |
| `seed42` | `sunny` | `sunny` | `cloudy` | 0.6102 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00156.jpg` |
| `seed42` | `sunny` | `sunny` | `cloudy` | 0.5575 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00210.jpg` |
| `seed42` | `cloudy` | `cloudy` | `snowy` | 0.5346 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00302.jpg` |
| `seed42` | `sunny` | `sunny` | `cloudy` | 0.4867 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01142.jpg` |
| `seed42` | `sunny` | `sunny` | `cloudy` | 0.4481 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00174.jpg` |
| `seed42` | `sunny` | `sunny` | `cloudy` | 0.4348 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01108.jpg` |
| `train2025` | `cloudy` | `cloudy` | `rainy` | 0.5897 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01803.jpg` |
| `train2025` | `cloudy` | `cloudy` | `rainy` | 0.5778 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00442.jpg` |
| `train2025` | `snowy` | `snowy` | `cloudy` | 0.4856 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00029.jpg` |

## Top Wrong-to-Wrong Changes

| Run | True | Before | After | Confidence After | Path |
| --- | --- | --- | --- | ---: | --- |
|  |  |  |  |  |  |

## Interpretation Boundary

- Bias values were selected on the same Internal Validation Split.
- This report is a diagnostic upper bound for decision calibration.
- Do not use these bias values in the Submission Prediction Interface without cross-run or split validation.
