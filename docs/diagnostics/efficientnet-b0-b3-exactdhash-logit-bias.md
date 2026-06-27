# Class-Wise Logit Bias Diagnostic

Class-wise logit bias was selected on the same Internal Validation Split used for scoring each row. Treat this as an in-split diagnostic upper bound for decision calibration, not as a Submission Prediction Interface parameter.

- Summary CSV: `docs/diagnostics/efficientnet-b0-b3-exactdhash-logit-bias-summary.csv`
- Changes CSV: `docs/diagnostics/efficientnet-b0-b3-exactdhash-logit-bias-changes.csv`
- Fixed label: `cloudy`
- Bias range: `-1.0` to `1.0`
- Bias step: `0.05`

## Run Summary

| Run | Baseline Macro F1 | Biased Macro F1 | Delta | Bias | Changed | Corrected | Regressed | Wrong->Wrong |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `b0_exact` | 0.9361 | 0.9361 | 0.0000 | `cloudy:0.0000;rainy:0.3500;snowy:-0.1000;sunny:0.4500` | 4 | 2 | 2 | 0 |
| `b3_exact` | 0.9203 | 0.9286 | 0.0083 | `cloudy:0.0000;rainy:0.6500;snowy:-0.8500;sunny:0.1000` | 9 | 7 | 2 | 0 |

## Per-Class F1

| Run | State | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | --- | ---: | ---: | ---: | ---: |
| `b0_exact` | baseline | 0.9345 | 0.9017 | 0.9682 | 0.9399 |
| `b0_exact` | biased | 0.9342 | 0.9017 | 0.9682 | 0.9403 |
| `b3_exact` | baseline | 0.9252 | 0.8927 | 0.9250 | 0.9381 |
| `b3_exact` | biased | 0.9270 | 0.9071 | 0.9367 | 0.9434 |

## Confusion Matrices: `b0_exact`

Baseline:
| True \ Pred | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | ---: | ---: | ---: | ---: |
| `cloudy` | 414 | 3 | 0 | 14 |
| `rainy` | 10 | 78 | 1 | 2 |
| `snowy` | 2 | 1 | 76 | 1 |
| `sunny` | 29 | 0 | 0 | 360 |

Biased:
| True \ Pred | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | ---: | ---: | ---: | ---: |
| `cloudy` | 412 | 3 | 0 | 16 |
| `rainy` | 10 | 78 | 1 | 2 |
| `snowy` | 2 | 1 | 76 | 1 |
| `sunny` | 27 | 0 | 0 | 362 |

## Confusion Matrices: `b3_exact`

Baseline:
| True \ Pred | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | ---: | ---: | ---: | ---: |
| `cloudy` | 402 | 6 | 3 | 20 |
| `rainy` | 10 | 79 | 1 | 1 |
| `snowy` | 3 | 1 | 74 | 2 |
| `sunny` | 23 | 0 | 2 | 364 |

Biased:
| True \ Pred | `cloudy` | `rainy` | `snowy` | `sunny` |
| --- | ---: | ---: | ---: | ---: |
| `cloudy` | 400 | 8 | 3 | 20 |
| `rainy` | 8 | 83 | 0 | 0 |
| `snowy` | 3 | 1 | 74 | 2 |
| `sunny` | 21 | 0 | 1 | 367 |

## Top Corrected Rows

| Run | True | Before | After | Confidence After | Path |
| --- | --- | --- | --- | ---: | --- |
| `b0_exact` | `sunny` | `cloudy` | `sunny` | 0.4659 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00184.jpg` |
| `b0_exact` | `sunny` | `cloudy` | `sunny` | 0.4567 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00222.jpg` |
| `b3_exact` | `sunny` | `snowy` | `sunny` | 0.6376 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00895.jpg` |
| `b3_exact` | `rainy` | `snowy` | `rainy` | 0.6133 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00189.jpg` |
| `b3_exact` | `rainy` | `cloudy` | `rainy` | 0.5512 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00231.jpg` |
| `b3_exact` | `sunny` | `cloudy` | `sunny` | 0.4925 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00060.jpg` |
| `b3_exact` | `rainy` | `cloudy` | `rainy` | 0.4852 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00220.jpg` |
| `b3_exact` | `sunny` | `cloudy` | `sunny` | 0.4657 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01445.jpg` |

## Top Regressed Rows

| Run | True | Before | After | Confidence After | Path |
| --- | --- | --- | --- | ---: | --- |
| `b0_exact` | `cloudy` | `cloudy` | `sunny` | 0.5670 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01874.jpg` |
| `b0_exact` | `cloudy` | `cloudy` | `sunny` | 0.4280 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00332.jpg` |
| `b3_exact` | `cloudy` | `cloudy` | `rainy` | 0.6047 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01710.jpg` |
| `b3_exact` | `cloudy` | `cloudy` | `rainy` | 0.5417 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01194.jpg` |

## Top Wrong-to-Wrong Changes

| Run | True | Before | After | Confidence After | Path |
| --- | --- | --- | --- | ---: | --- |
|  |  |  |  |  |  |

## Interpretation Boundary

- Bias values were selected on the same Internal Validation Split.
- This report is a diagnostic upper bound for decision calibration.
- Do not use these bias values in the Submission Prediction Interface without cross-run or split validation.
