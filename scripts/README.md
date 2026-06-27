# Scripts

Small helper scripts used by documented repository commands live here.

## Rules

- Scripts must be safe to run from the repository root.
- Scripts should be idempotent when possible.
- Keep scripts narrow and boring.
- Prefer scripts that support `justfile` recipes over one-off manual workflows.
- Avoid machine-specific absolute paths, local secrets, and hidden network dependencies.
- Agent-facing scripts must write caches and temporary files inside the workspace.
- Keep `agent-env.sh` lightweight and generic. Add tool-specific cache variables
only when the template or project has a concrete reason to support that tool.

## Current Scripts

- `agent-env.sh`: runs a command with workspace-local cache defaults such as
`.cache/` and `.cache/uv/`; `UV_CACHE_DIR` is included for possible Python
helper scripts.
- `check-data-layout.py`: validates that the local Training Image Root directly
contains the four weather category directories.
- `promote-submission.py`: copies a selected Model Artifact to the fixed
Submission Artifact path at `results/model_sample.pth`.
- `confirm-submission.py`: checks for submission-facing files and prints the
manual Submission Confirmation checklist.
- `smoke-predict.py`: validates `main.predict()` with an existing
`results/model_sample.pth` artifact in the active runtime.
- `audit-val-predictions.py`: creates a compact high-confidence validation error
audit pack from an Experiment Run's `val_predictions.csv`.
- `summarize-val-error-overlap.py`: creates a stable validation error pool
diagnostic report from comparable Experiment Runs.
- `audit-val-image-features.py`: creates image-statistic and dHash diagnostics
for an Experiment Run's validation images.
- `audit-train-image-features.py`: creates image-statistic and dHash
diagnostics for the full Training Set.
- `audit-split-duplicate-leakage.py`: audits exact and near-duplicate dHash
groups that cross a train/validation split boundary.
- `diagnose-logit-bias.py`: grid-searches class-wise logit bias on validation
prediction logits for diagnostic calibration analysis.
