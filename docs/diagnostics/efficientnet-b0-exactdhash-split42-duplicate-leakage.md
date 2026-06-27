# Split Duplicate Leakage Diagnostic

Duplicate-group diagnostics for an Internal Validation Split. dHash groups are coarse perceptual similarity candidates; a split leak is evidence of possible local validation optimism, not proof of duplicate source images or label errors.

- Features CSV: `docs/diagnostics/train-image-features.csv`
- Validation predictions: `results/runs/efficientnet-b0-exactdhash-split42-train42-bs32-w4/val_predictions.csv`
- CSV: `docs/diagnostics/efficientnet-b0-exactdhash-split42-duplicate-leakage.csv`
- Training Set rows: `4999`
- Internal Validation Split rows: `991`
- Inferred training rows: `4008`
- Validation error counts are model-specific values from the provided `val_predictions.csv` file.

## Duplicate Group Split Summary

| Group Type | Groups | Samples | Leaked Groups | Leaked Samples | Leaked Val Samples | Leaked Val Errors | Cross-Label Leaked Groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `exact_dhash` | 174 | 373 | 0 | 0 | 0 | 0 | 0 |
| `near_duplicate` | 193 | 467 | 14 | 65 | 19 | 0 | 3 |

## Leaked Exact dHash Groups

| Group Type | Group | Size | Train | Val | Val Errors | Cross Label | Labels | Example Val Paths | Example Train Paths |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
|  |  | 0 | 0 | 0 | 0 | 0 |  |  |  |

## Leaked Near-Duplicate Groups

| Group Type | Group | Size | Train | Val | Val Errors | Cross Label | Labels | Example Val Paths | Example Train Paths |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `near_duplicate` | `near_085` | 25 | 22 | 3 | 0 | 1 | `sunny:12; cloudy:5; rainy:4; snowy:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01877.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00439.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01938.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00375.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00483.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01389.jpg` |
| `near_duplicate` | `near_015` | 7 | 5 | 2 | 0 | 1 | `sunny:5; cloudy:2` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00765.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01801.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00570.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01267.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01879.jpg` |
| `near_duplicate` | `near_164` | 4 | 2 | 2 | 0 | 0 | `sunny:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01140.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01866.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01877.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01954.jpg` |
| `near_duplicate` | `near_058` | 3 | 1 | 2 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01087.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02128.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02141.jpg` |
| `near_duplicate` | `near_119` | 4 | 3 | 1 | 0 | 1 | `rainy:3; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01940.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00193.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00264.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00428.jpg` |
| `near_duplicate` | `near_033` | 3 | 2 | 1 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01984.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00768.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02073.jpg` |
| `near_duplicate` | `near_098` | 3 | 2 | 1 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02038.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01802.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02113.jpg` |
| `near_duplicate` | `near_111` | 3 | 2 | 1 | 0 | 0 | `cloudy:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02015.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02050.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02109.jpg` |
| `near_duplicate` | `near_186` | 3 | 2 | 1 | 0 | 0 | `sunny:3` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01928.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01784.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01926.jpg` |
| `near_duplicate` | `near_107` | 2 | 1 | 1 | 0 | 0 | `cloudy:2` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_02037.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01967.jpg` |

## Leaked Cross-Label Duplicate Groups

| Group Type | Group | Size | Train | Val | Val Errors | Cross Label | Labels | Example Val Paths | Example Train Paths |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `near_duplicate` | `near_085` | 25 | 22 | 3 | 0 | 1 | `sunny:12; cloudy:5; rainy:4; snowy:4` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01877.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00439.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01938.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00375.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00483.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01389.jpg` |
| `near_duplicate` | `near_015` | 7 | 5 | 2 | 0 | 1 | `sunny:5; cloudy:2` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_00765.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01801.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_00570.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/cloudy/cloudy_01267.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01879.jpg` |
| `near_duplicate` | `near_119` | 4 | 3 | 1 | 0 | 1 | `rainy:3; sunny:1` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/sunny/sunny_01940.jpg` | `datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00193.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00264.jpg`<br>`datasets/6a39ed934d7b489daf5f80a4-momodel/train/rainy/rainy_00428.jpg` |
