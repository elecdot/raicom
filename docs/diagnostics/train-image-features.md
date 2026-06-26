# Training Set Image Features

Image feature diagnostics for the full official Training Set. This report summarizes observed data properties and does not propose automatic relabeling rules. dHash groups are coarse perceptual similarity candidates, not proof of duplicate source images or label errors.

- Training Image Root: `datasets/6a39ed934d7b489daf5f80a4-momodel/train`
- CSV: `docs/diagnostics/train-image-features.csv`
- Rows: `4999`
- Read failures: `0`
- Near-duplicate threshold: `5`

## Weather Category Counts

| Weather Category | Rows |
| --- | ---: |
| `cloudy` | 2184 |
| `rainy` | 446 |
| `snowy` | 403 |
| `sunny` | 1966 |

## Image Dimensions

| Width | Height | Rows |
| ---: | ---: | ---: |
| 224 | 224 | 4999 |

## Feature Means By True Label

| Group | Rows | Read OK | Brightness | Saturation | Contrast | Dark Ratio | Bright Ratio | Low Sat Ratio | Edge Density | Laplacian Var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cloudy` | 2184 | 2184 | 133.2597 | 53.2381 | 61.7070 | 0.1824 | 0.2394 | 0.4785 | 0.0927 | 949.7297 |
| `rainy` | 446 | 446 | 126.1771 | 43.7014 | 67.5851 | 0.2458 | 0.2506 | 0.5417 | 0.1102 | 1174.7733 |
| `snowy` | 403 | 403 | 150.7043 | 43.5380 | 56.5810 | 0.1213 | 0.3241 | 0.5522 | 0.1276 | 1466.1564 |
| `sunny` | 1966 | 1966 | 128.4627 | 94.7512 | 53.6185 | 0.1618 | 0.1505 | 0.1769 | 0.1113 | 1300.8234 |

## Feature Quantiles By True Label

| Label | Feature | P05 | P50 | P95 |
| --- | --- | ---: | ---: | ---: |
| `cloudy` | `brightness_mean` | 96.5865 | 134.3709 | 168.3475 |
| `cloudy` | `saturation_mean` | 18.3093 | 47.0792 | 110.5763 |
| `cloudy` | `contrast` | 37.8098 | 62.3175 | 83.7597 |
| `cloudy` | `edge_density` | 0.0179 | 0.0887 | 0.1786 |
| `cloudy` | `laplacian_variance` | 65.3729 | 784.0217 | 2465.8615 |
| `rainy` | `brightness_mean` | 87.4366 | 126.0932 | 164.8725 |
| `rainy` | `saturation_mean` | 14.2304 | 37.0943 | 100.2372 |
| `rainy` | `contrast` | 41.1357 | 69.1670 | 87.4724 |
| `rainy` | `edge_density` | 0.0458 | 0.1071 | 0.1896 |
| `rainy` | `laplacian_variance` | 378.8081 | 1050.2011 | 2384.0169 |
| `snowy` | `brightness_mean` | 109.1651 | 150.2411 | 193.8542 |
| `snowy` | `saturation_mean` | 6.9690 | 35.1809 | 110.2876 |
| `snowy` | `contrast` | 34.7499 | 55.5715 | 78.3357 |
| `snowy` | `edge_density` | 0.0605 | 0.1241 | 0.2070 |
| `snowy` | `laplacian_variance` | 440.1142 | 1304.7136 | 3047.3194 |
| `sunny` | `brightness_mean` | 95.6020 | 128.2373 | 163.3184 |
| `sunny` | `saturation_mean` | 48.1505 | 90.8612 | 151.4927 |
| `sunny` | `contrast` | 35.6508 | 52.6594 | 75.2155 |
| `sunny` | `edge_density` | 0.0386 | 0.1090 | 0.1916 |
| `sunny` | `laplacian_variance` | 189.6890 | 1131.8860 | 2980.5438 |

## Duplicate Hash Summary

- Exact dHash collision groups: `174`
- Samples in exact dHash collision groups: `373`
- Cross-label exact dHash groups: `2`
- Near-duplicate groups: `193`
- Samples in near-duplicate groups: `467`
- Cross-label near-duplicate groups: `6`

## Top Exact dHash Groups

| Group | Size | Labels | Example Paths |
| --- | ---: | --- | --- |
| `1000008080800000` | 6 | `sunny:6` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00968.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01713.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01805.jpg` |
| `0f17c3e3ec7c92dc` | 5 | `cloudy:5` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00581.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01084.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02059.jpg` |
| `c6c879eacc9d25e6` | 5 | `cloudy:5` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01225.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01294.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01784.jpg` |
| `00003038b8fcfc7c` | 4 | `rainy:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00375.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00421.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00430.jpg` |
| `1b3cca8e2d11c761` | 3 | `rainy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00300.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00429.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00442.jpg` |
| `23f25e9880f1a32f` | 3 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00993.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01349.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02183.jpg` |
| `32b2c29048a8e0a8` | 3 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01268.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01981.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02107.jpg` |
| `4828ccccc4e0a4cc` | 3 | `sunny:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01144.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01855.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01899.jpg` |
| `5c5860c68478d0c6` | 3 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00682.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01498.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02039.jpg` |
| `8584808208b0a387` | 3 | `sunny:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00571.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00986.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01942.jpg` |

## Top Cross-Label Exact dHash Groups

| Group | Size | Labels | Example Paths |
| --- | ---: | --- | --- |
| `e4b0e478704226a2` | 2 | `cloudy:1; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00393.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00226.jpg` |
| `ff7f3f2b29a59cc5` | 2 | `cloudy:1; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00994.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01585.jpg` |

## Top Near-Duplicate Groups

| Group | Size | Labels | Example Paths |
| --- | ---: | --- | --- |
| `near_085` | 25 | `sunny:12; cloudy:5; rainy:4; snowy:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00375.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00483.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01389.jpg` |
| `near_015` | 7 | `sunny:5; cloudy:2` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00570.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01267.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00765.jpg` |
| `near_158` | 6 | `sunny:6` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00968.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01713.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01805.jpg` |
| `near_019` | 5 | `cloudy:5` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00581.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01084.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02059.jpg` |
| `near_073` | 5 | `cloudy:5` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01225.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01294.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01784.jpg` |
| `near_128` | 5 | `sunny:5` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00254.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00619.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00669.jpg` |
| `near_154` | 5 | `sunny:5` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00906.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01839.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01844.jpg` |
| `near_119` | 4 | `rainy:3; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00193.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00264.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00428.jpg` |
| `near_122` | 4 | `rainy:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00375.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00421.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00430.jpg` |
| `near_125` | 4 | `sunny:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00116.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01366.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01804.jpg` |

## Top Cross-Label Near-Duplicate Groups

| Group | Size | Labels | Example Paths |
| --- | ---: | --- | --- |
| `near_085` | 25 | `sunny:12; cloudy:5; rainy:4; snowy:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00375.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00483.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01389.jpg` |
| `near_015` | 7 | `sunny:5; cloudy:2` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00570.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01267.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00765.jpg` |
| `near_119` | 4 | `rainy:3; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00193.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00264.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00428.jpg` |
| `near_002` | 2 | `cloudy:1; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00393.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00226.jpg` |
| `near_049` | 2 | `cloudy:1; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00994.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01585.jpg` |
| `near_087` | 2 | `cloudy:1; rainy:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01402.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00040.jpg` |

## Darkest Rows

| Value | True | Path |
| ---: | --- | --- |
| 30.7192 | `rainy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00439.jpg` |
| 43.7245 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02056.jpg` |
| 45.0478 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01972.jpg` |
| 47.2103 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00179.jpg` |
| 48.0188 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00018.jpg` |
| 51.5982 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02103.jpg` |
| 55.4241 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02088.jpg` |
| 56.0552 | `rainy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00001.jpg` |
| 57.1164 | `rainy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00444.jpg` |
| 57.6456 | `rainy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00056.jpg` |

## Brightest Rows

| Value | True | Path |
| ---: | --- | --- |
| 226.8205 | `snowy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00235.jpg` |
| 224.8809 | `snowy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00210.jpg` |
| 219.7012 | `snowy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00152.jpg` |
| 212.9789 | `snowy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00298.jpg` |
| 209.4544 | `snowy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00102.jpg` |
| 209.3642 | `snowy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00163.jpg` |
| 208.2476 | `snowy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00264.jpg` |
| 207.6570 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00842.jpg` |
| 207.4614 | `snowy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00266.jpg` |
| 206.8010 | `snowy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00237.jpg` |

## Lowest Saturation Rows

| Value | True | Path |
| ---: | --- | --- |
| 0.0000 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00235.jpg` |
| 0.0000 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00551.jpg` |
| 0.0000 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00752.jpg` |
| 0.0000 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00789.jpg` |
| 0.0000 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01016.jpg` |
| 0.0000 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01199.jpg` |
| 0.0000 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01377.jpg` |
| 0.0000 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01407.jpg` |
| 0.0000 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01517.jpg` |
| 0.0000 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01608.jpg` |

## Lowest Laplacian Variance Rows

| Value | True | Path |
| ---: | --- | --- |
| 1.4515 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00024.jpg` |
| 3.1424 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00208.jpg` |
| 3.2943 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01888.jpg` |
| 3.5998 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02001.jpg` |
| 3.8802 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01986.jpg` |
| 4.2540 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02048.jpg` |
| 4.2862 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01807.jpg` |
| 4.5307 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01916.jpg` |
| 6.5782 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00355.jpg` |
| 7.0337 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02164.jpg` |

## Highest Edge Density Rows

| Value | True | Path |
| ---: | --- | --- |
| 0.3560 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00524.jpg` |
| 0.3115 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01581.jpg` |
| 0.2909 | `snowy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/snowy/snowy_00159.jpg` |
| 0.2811 | `cloudy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01433.jpg` |
| 0.2767 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00118.jpg` |
| 0.2744 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00512.jpg` |
| 0.2642 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01020.jpg` |
| 0.2627 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01209.jpg` |
| 0.2620 | `rainy` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00026.jpg` |
| 0.2593 | `sunny` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00420.jpg` |
