# Split Duplicate Leakage Diagnostic

Duplicate-group diagnostics for an Internal Validation Split. dHash groups are coarse perceptual similarity candidates; a split leak is evidence of possible local validation optimism, not proof of duplicate source images or label errors.

- Features CSV: `docs/diagnostics/train-image-features.csv`
- Validation predictions: `results/runs/efficientnet-b0-seed42-bs32-w4/val_predictions.csv`
- CSV: `docs/diagnostics/split42-duplicate-leakage.csv`
- Training Set rows: `4999`
- Internal Validation Split rows: `1000`
- Inferred training rows: `3999`
- Validation error counts are model-specific values from the provided `val_predictions.csv` file.

## Duplicate Group Split Summary

| Group Type | Groups | Samples | Leaked Groups | Leaked Samples | Leaked Val Samples | Leaked Val Errors | Cross-Label Leaked Groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `exact_dhash` | 174 | 373 | 60 | 142 | 64 | 1 | 1 |
| `near_duplicate` | 193 | 467 | 70 | 202 | 80 | 1 | 3 |

## Leaked Exact dHash Groups

| Group Type | Group | Size | Train | Val | Val Errors | Cross Label | Labels | Example Val Paths | Example Train Paths |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `exact_dhash` | `e4b0e478704226a2` | 2 | 1 | 1 | 1 | 1 | `cloudy:1; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00393.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00226.jpg` |
| `exact_dhash` | `1000008080800000` | 6 | 4 | 2 | 0 | 0 | `sunny:6` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00968.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01713.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01805.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01872.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01902.jpg` |
| `exact_dhash` | `5c5860c68478d0c6` | 3 | 1 | 2 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00682.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02039.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01498.jpg` |
| `exact_dhash` | `f27afe8c58f552d0` | 3 | 1 | 2 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02044.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02100.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01272.jpg` |
| `exact_dhash` | `f8ccc0dabefbd8d8` | 3 | 1 | 2 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01292.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02125.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02111.jpg` |
| `exact_dhash` | `0f17c3e3ec7c92dc` | 5 | 4 | 1 | 0 | 0 | `cloudy:5` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02059.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00581.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01084.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02078.jpg` |
| `exact_dhash` | `c6c879eacc9d25e6` | 5 | 4 | 1 | 0 | 0 | `cloudy:5` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01225.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01294.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01784.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02082.jpg` |
| `exact_dhash` | `00003038b8fcfc7c` | 4 | 3 | 1 | 0 | 0 | `rainy:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00375.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00421.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00430.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00435.jpg` |
| `exact_dhash` | `23f25e9880f1a32f` | 3 | 2 | 1 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00993.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01349.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02183.jpg` |
| `exact_dhash` | `32b2c29048a8e0a8` | 3 | 2 | 1 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01268.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01981.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02107.jpg` |

## Leaked Near-Duplicate Groups

| Group Type | Group | Size | Train | Val | Val Errors | Cross Label | Labels | Example Val Paths | Example Train Paths |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `near_duplicate` | `near_002` | 2 | 1 | 1 | 1 | 1 | `cloudy:1; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00393.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00226.jpg` |
| `near_duplicate` | `near_085` | 25 | 21 | 4 | 0 | 1 | `sunny:12; cloudy:5; rainy:4; snowy:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01389.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00108.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00124.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00375.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00483.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01724.jpg` |
| `near_duplicate` | `near_015` | 7 | 4 | 3 | 0 | 1 | `sunny:5; cloudy:2` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00570.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01801.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01882.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01267.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00765.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01879.jpg` |
| `near_duplicate` | `near_158` | 6 | 4 | 2 | 0 | 0 | `sunny:6` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00968.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01713.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01805.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01872.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01902.jpg` |
| `near_duplicate` | `near_125` | 4 | 2 | 2 | 0 | 0 | `sunny:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01804.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01807.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00116.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01366.jpg` |
| `near_duplicate` | `near_031` | 3 | 1 | 2 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00682.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02039.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01498.jpg` |
| `near_duplicate` | `near_076` | 3 | 1 | 2 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02044.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02100.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01272.jpg` |
| `near_duplicate` | `near_077` | 3 | 1 | 2 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01292.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02125.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02111.jpg` |
| `near_duplicate` | `near_019` | 5 | 4 | 1 | 0 | 0 | `cloudy:5` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02059.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00581.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01084.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02078.jpg` |
| `near_duplicate` | `near_073` | 5 | 4 | 1 | 0 | 0 | `cloudy:5` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01225.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01294.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01784.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02082.jpg` |

## Leaked Cross-Label Duplicate Groups

| Group Type | Group | Size | Train | Val | Val Errors | Cross Label | Labels | Example Val Paths | Example Train Paths |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `exact_dhash` | `e4b0e478704226a2` | 2 | 1 | 1 | 1 | 1 | `cloudy:1; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00393.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00226.jpg` |
| `near_duplicate` | `near_002` | 2 | 1 | 1 | 1 | 1 | `cloudy:1; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00393.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00226.jpg` |
| `near_duplicate` | `near_085` | 25 | 21 | 4 | 0 | 1 | `sunny:12; cloudy:5; rainy:4; snowy:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01389.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00108.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00124.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00375.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00483.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01724.jpg` |
| `near_duplicate` | `near_015` | 7 | 4 | 3 | 0 | 1 | `sunny:5; cloudy:2` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00570.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01801.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01882.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01267.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00765.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01879.jpg` |
