# RAICOM Weather Classification

Workspace for a RAICOM weather image classification competition. The project targets macro F1 on the hosted momodel scoring environment while keeping local experiments reproducible and reviewable.

Update the following files when bootstrapping a new project:
```
README.md       # purpose / repo map /commands
CONTEXT.md      # project language and scope
AGENTS.md       # validation guidance / project-specific warnings
justfile        # actual commands
```

## Environment

Prerequisites:

- Git
- Bash
- `just`
- `uv` for local development helpers
- Python 3.9.5 for platform-compatibility checks

```bash
just setup
just check
```

The repository uses `uv` for local development tooling such as Jupytext.
`just setup` syncs only that local tool layer. Runtime dependencies remain
explicit pip-managed dependency sets:

- Platform CPU runtime: `python -m pip install -r requirements.txt`
- CUDA training runtime: `python -m pip install -r requirements-train-cu124.txt`

## Repo Map

| Path | Purpose |
| --- | --- |
| `CONTEXT.md` | Canonical language for this template's purpose and scope. |
| `AGENTS.md` | Working contract for coding agents and human-assisted automation. |
| `justfile` | Main command entry point. |
| `main.py` | Platform-facing prediction entry point. |
| `weather_model.py` | Shared baseline labels, image size, and model used by training. |
| `train.py` | Baseline training script. |
| `results/` | Canonical model artifact directory used by training and submission code. |
| `requirements.txt` | pip runtime dependency set for platform runs. |
| `requirements-train-cu124.txt` | pip dependency set for CUDA 12.4 training runs. |
| `scripts/` | Small helper scripts used by documented commands. |
| `docs/` | Durable project documentation and conventions. |
| `docs/adr/` | Optional architecture decision records. |
| `tmp/` | Local scratch space; only its README and ignore rules are tracked. |

## Commands

Prefer repository commands over ad hoc long commands:

```bash
just doctor       # show workspace status and recipes
just setup        # sync uv-managed local development tools
just check        # run lightweight non-training checks
just fmt          # format maintained Python sources
just lint         # lint maintained Python sources
just test         # run lightweight pytest tests
just train        # run baseline training in the active GPU runtime
just agent <cmd>  # run a command with workspace-local caches
```

## Current State

- The official tutorial is available as `tutorial.ipynb` and paired `tutorial.py`.
- `weather_model.py` is the training-side source of truth for baseline labels, image size, and model structure.
- `main.py` remains self-contained for platform submission and is checked against `weather_model.py`.
- Training data is expected under `datasets/6a39ed934d7b489daf5f80a4-momodel/train/`.
- Model artifacts are saved under `results/`; the submitted prediction code loads from this directory.
- `predict()` receives `cv2.imread` BGR input and converts it to RGB before model inference.
- The first runtime dependency set is pinned for Python 3.9 and CPU-capable platform runs.
- CUDA training has a separate Python 3.9.5 dependency file for CUDA 12.4 GPU environments.
- `just check` verifies the uv lock, lint, lightweight tests, Python syntax, local training data layout, and submission entrypoint drift without installing model runtime dependencies or training.
- `train.py` is a CLI baseline training driver; `just train` requires CUDA and writes `results/model_sample.pth` by default.

## Open Loops

- [ ] Add a smoke test for the `predict()` interface once a model artifact exists.
- [ ] Validate `requirements-train-cu124.txt` on an actual GPU training machine.
