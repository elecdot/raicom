set shell := ["bash", "-cu"]

# List recipes.
default:
    @just --list

# Prepare the workspace.
setup:
    @./scripts/agent-env.sh uv sync --dev
    @printf '\nRuntime dependency sets are intentionally not installed by `just setup`.\n'
    @printf 'Platform CPU runtime: python -m pip install -r requirements.txt\n'
    @printf 'CUDA training runtime: python -m pip install -r requirements-train-cu124.txt\n'
    @printf 'TODO: validate the CUDA training dependency set on an actual GPU machine.\n'

# Show workspace status.
doctor:
    @git status --short
    @printf '\n'
    @just --list

# Run lightweight checks that do not install runtime dependencies or train models.
check:
    @./scripts/agent-env.sh uv lock --check
    @just lint
    @./scripts/agent-env.sh uv run python -m py_compile main.py train.py models/*.py
    @./scripts/agent-env.sh uv run python scripts/check-data-layout.py
    @just test

# Run default Model Candidate training in the active Python runtime. Local GPU hint: --batch-size 32 --num-workers 4 --persistent-workers.
train *args:
    @python train.py --require-cuda {{args}}

# Create an audit pack from validation prediction errors.
audit-errors run *args:
    @python scripts/audit-val-predictions.py {{run}} {{args}}

# Smoke-test the platform predict interface in the active runtime.
smoke-predict *args:
    @python scripts/smoke-predict.py {{args}}

# Promote a Model Artifact to the fixed Submission Artifact path.
promote-submission artifact:
    @python scripts/promote-submission.py {{artifact}}

# Print the pre-submission confirmation checklist.
confirm-submission:
    @python scripts/confirm-submission.py

# Run with workspace-local caches.
agent *args:
    @./scripts/agent-env.sh {{args}}

# Format maintained Python sources.
fmt:
    @./scripts/agent-env.sh uv run ruff format main.py train.py models scripts tests

# Lint maintained Python sources.
lint:
    @./scripts/agent-env.sh uv run ruff check main.py train.py models scripts tests

# Run lightweight tests.
test:
    @./scripts/agent-env.sh uv run pytest -q tests
