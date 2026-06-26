# Experiments

Manual review log for notable Experiment Runs. Keep this table short enough to review; use each run's `metadata.json` and `metrics.json` for full machine-readable details.

| Run ID | Model Candidate | Key Config | Internal Validation Macro F1 | Artifact | Notes | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `20260626-175735-efficientnet_b0-seed42` | `efficientnet_b0` | seed 42, bs 32, mild aug, inverse-sqrt class weights, label smoothing 0.05 | 0.9302 | `results/runs/20260626-175735-efficientnet_b0-seed42/model.pth` | Best epoch 24; strong first run. | Keep as first strong baseline evidence. |
| `efficientnet-b0-seed2025` | `efficientnet_b0` | seed 2025, bs 32, mild aug, inverse-sqrt class weights, label smoothing 0.05 | 0.9161 | `results/runs/efficientnet-b0-seed2025/model.pth` | Early stopped at epoch 10; best epoch 3. | Confirms lower-bound seed variance. |
| `efficientnet-b0-seed3407` | `efficientnet_b0` | seed 3407, bs 32, mild aug, inverse-sqrt class weights, label smoothing 0.05 | 0.9326 | `results/runs/efficientnet-b0-seed3407/model.pth` | Best epoch 23; strongest B0 run so far. | Use with seed 42/2025 to summarize B0 baseline. |
