set shell := ["bash", "-cu"]

# List recipes.
default:
    @just --list

# Prepare the workspace.
setup:
    @printf 'Run `uv sync --dev` for local development tools.\n'

# Show workspace status.
doctor:
    @git status --short
    @printf '\n'
    @just --list

# Run lightweight checks that do not install runtime dependencies or train models.
check:
    @./scripts/agent-env.sh uv lock --check
    @./scripts/agent-env.sh uv run python -m py_compile main.py train.py
    @./scripts/agent-env.sh uv run python scripts/check-data-layout.py

# Run with workspace-local caches.
agent *args:
    @./scripts/agent-env.sh {{args}}

# Configure a formatter.
fmt:
    @printf 'No formatter configured yet.\n' >&2
    @exit 1

# Configure a linter.
lint:
    @printf 'No linter configured yet.\n' >&2
    @exit 1

# No test suite is configured yet.
test:
    @printf 'No tests configured yet.\n' >&2
    @exit 1

# Configure a build.
build:
    @printf 'No build configured yet.\n' >&2
    @exit 1
