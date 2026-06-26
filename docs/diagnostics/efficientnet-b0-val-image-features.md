# EfficientNet-B0 Validation Image Features

Image feature diagnostics for the fixed split used by the seed42 EfficientNet-B0 Experiment Run.

- Predictions: `results/runs/efficientnet-b0-seed42-bs32-w4/val_predictions.csv`
- Stable error pool: `docs/diagnostics/efficientnet-b0-stable-error-pool.csv`
- CSV: `docs/diagnostics/efficientnet-b0-val-image-features.csv`
- Rows: `1000`
- Read failures: `0`
- Near-duplicate threshold: `5`

## Diagnostic Category Counts

| Category | Rows |
| --- | ---: |
| `not_in_stable_pool` | 913 |
| `random_sensitive` | 35 |
| `boundary_case` | 20 |
| `stable_high_conf_error` | 18 |
| `stable_model_error` | 9 |
| `low_priority` | 5 |

## Feature Means By True Label

| Group | Rows | Read OK | Brightness | Saturation | Contrast | Dark Ratio | Bright Ratio | Low Sat Ratio | Edge Density | Laplacian Var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cloudy` | 437 | 437 | 132.9646 | 53.8680 | 61.3555 | 0.1827 | 0.2316 | 0.4773 | 0.0923 | 967.4323 |
| `rainy` | 89 | 89 | 124.7477 | 46.3307 | 69.0280 | 0.2554 | 0.2551 | 0.5114 | 0.1120 | 1224.0863 |
| `snowy` | 81 | 81 | 146.2636 | 45.0024 | 60.0834 | 0.1444 | 0.3183 | 0.5291 | 0.1298 | 1548.2932 |
| `sunny` | 393 | 393 | 128.5024 | 93.7233 | 53.8586 | 0.1608 | 0.1519 | 0.1800 | 0.1138 | 1338.0428 |

## Feature Means By Seed42 Correctness

| Group | Rows | Read OK | Brightness | Saturation | Contrast | Dark Ratio | Bright Ratio | Low Sat Ratio | Edge Density | Laplacian Var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `correct` | 948 | 948 | 131.5541 | 68.1149 | 58.8630 | 0.1768 | 0.2086 | 0.3686 | 0.1054 | 1189.7303 |
| `error` | 52 | 52 | 131.6077 | 68.6393 | 61.2869 | 0.1893 | 0.2230 | 0.3516 | 0.1080 | 1059.8056 |

## Feature Means By Stable Pool Category

| Group | Rows | Read OK | Brightness | Saturation | Contrast | Dark Ratio | Bright Ratio | Low Sat Ratio | Edge Density | Laplacian Var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `boundary_case` | 20 | 20 | 133.2300 | 67.7203 | 61.1422 | 0.1818 | 0.2295 | 0.3973 | 0.0962 | 925.6224 |
| `low_priority` | 5 | 5 | 144.4017 | 46.9066 | 62.5871 | 0.1321 | 0.3593 | 0.5266 | 0.1059 | 936.9101 |
| `not_in_stable_pool` | 913 | 913 | 131.7194 | 68.2030 | 58.7758 | 0.1758 | 0.2089 | 0.3683 | 0.1057 | 1195.8646 |
| `random_sensitive` | 35 | 35 | 126.2161 | 71.2590 | 61.4593 | 0.2130 | 0.1977 | 0.3433 | 0.1131 | 1243.5419 |
| `stable_high_conf_error` | 18 | 18 | 123.9097 | 73.0366 | 65.0186 | 0.2347 | 0.2091 | 0.3130 | 0.1009 | 897.2976 |
| `stable_model_error` | 9 | 9 | 140.2759 | 52.7939 | 52.1749 | 0.1129 | 0.1748 | 0.3559 | 0.0945 | 919.7252 |

## Duplicate Hash Summary

- Exact dHash collision groups: `11`
- Samples in exact dHash collision groups: `22`
- Near-duplicate groups: `14`
- Samples in near-duplicate groups: `29`

## Top Exact dHash Groups

| Group | Size | Labels | Correct | Example Paths |
| --- | ---: | --- | ---: | --- |
| `01c1813a2ee50a72` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01694.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00456.jpg` |
| `1000008080800000` | 2 | `sunny:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01713.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00968.jpg` |
| `30303016acdcede1` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00551.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02041.jpg` |
| `5c5860c68478d0c6` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02039.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00682.jpg` |
| `61d828eacce9efaf` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02064.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01409.jpg` |
| `80c0c0c0c4c0e9c1` | 2 | `sunny:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00850.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01799.jpg` |
| `c48cce6369296b0f` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01029.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02112.jpg` |
| `d8f8fa796decac88` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01723.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02157.jpg` |
| `d9d959dbdececdd0` | 2 | `sunny:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00262.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01851.jpg` |
| `f27afe8c58f552d0` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02044.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02100.jpg` |

## Top Near-Duplicate Groups

| Group | Size | Labels | Correct | Example Paths |
| --- | ---: | --- | ---: | --- |
| `near_003` | 3 | `sunny:2; cloudy:1` | 3 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01882.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00570.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01801.jpg` |
| `near_001` | 2 | `sunny:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01807.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01804.jpg` |
| `near_002` | 2 | `sunny:1; rainy:1` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01927.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00108.jpg` |
| `near_004` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02044.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02100.jpg` |
| `near_005` | 2 | `sunny:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00850.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01799.jpg` |
| `near_006` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01029.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02112.jpg` |
| `near_007` | 2 | `sunny:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00262.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01851.jpg` |
| `near_008` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00551.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02041.jpg` |
| `near_009` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01694.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00456.jpg` |
| `near_010` | 2 | `cloudy:2` | 2 | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02125.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01292.jpg` |

## Darkest Rows

| Value | True | Pred | Correct | Diagnostic | Path |
| ---: | --- | --- | ---: | --- | --- |
| 48.0188 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00018.jpg` |
| 57.1164 | `rainy` | `rainy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00444.jpg` |
| 61.9409 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01301.jpg` |
| 62.7697 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02047.jpg` |
| 64.4468 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02089.jpg` |
| 67.3604 | `sunny` | `sunny` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01326.jpg` |
| 71.0562 | `rainy` | `rainy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00375.jpg` |
| 71.1021 | `sunny` | `sunny` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01099.jpg` |
| 73.2263 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02123.jpg` |
| 75.9580 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01990.jpg` |

## Brightest Rows

| Value | True | Pred | Correct | Diagnostic | Path |
| ---: | --- | --- | ---: | --- | --- |
| 201.6002 | `sunny` | `sunny` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00347.jpg` |
| 198.4686 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00805.jpg` |
| 195.7996 | `snowy` | `snowy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00224.jpg` |
| 195.6801 | `snowy` | `snowy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00104.jpg` |
| 192.2280 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01370.jpg` |
| 189.8516 | `snowy` | `snowy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00112.jpg` |
| 187.1310 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00871.jpg` |
| 185.3441 | `snowy` | `snowy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00278.jpg` |
| 184.5763 | `snowy` | `snowy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00293.jpg` |
| 182.3586 | `rainy` | `rainy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00019.jpg` |

## Lowest Saturation Rows

| Value | True | Pred | Correct | Diagnostic | Path |
| ---: | --- | --- | ---: | --- | --- |
| 0.0000 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02040.jpg` |
| 0.0000 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01644.jpg` |
| 0.0000 | `rainy` | `snowy` | 0 | `stable_model_error` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00175.jpg` |
| 0.0000 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01734.jpg` |
| 0.0000 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00752.jpg` |
| 0.0000 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02031.jpg` |
| 0.0000 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00551.jpg` |
| 0.0000 | `rainy` | `rainy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00108.jpg` |
| 0.0000 | `snowy` | `snowy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00132.jpg` |
| 0.0000 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02041.jpg` |

## Lowest Laplacian Variance Rows

| Value | True | Pred | Correct | Diagnostic | Path |
| ---: | --- | --- | ---: | --- | --- |
| 4.2540 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02048.jpg` |
| 4.2862 | `sunny` | `sunny` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01807.jpg` |
| 9.8246 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01982.jpg` |
| 9.9323 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01996.jpg` |
| 10.8539 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01976.jpg` |
| 15.4097 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02149.jpg` |
| 17.4346 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01980.jpg` |
| 18.3831 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02091.jpg` |
| 18.4367 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02057.jpg` |
| 19.2923 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02124.jpg` |

## Highest Edge Density Rows

| Value | True | Pred | Correct | Diagnostic | Path |
| ---: | --- | --- | ---: | --- | --- |
| 0.2620 | `rainy` | `cloudy` | 0 | `random_sensitive` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00026.jpg` |
| 0.2593 | `sunny` | `cloudy` | 0 | `stable_high_conf_error` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00420.jpg` |
| 0.2560 | `sunny` | `sunny` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01965.jpg` |
| 0.2407 | `sunny` | `sunny` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00494.jpg` |
| 0.2402 | `snowy` | `snowy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00129.jpg` |
| 0.2387 | `sunny` | `sunny` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00695.jpg` |
| 0.2359 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01697.jpg` |
| 0.2341 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01833.jpg` |
| 0.2294 | `snowy` | `snowy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00132.jpg` |
| 0.2268 | `cloudy` | `cloudy` | 1 | `` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01807.jpg` |
