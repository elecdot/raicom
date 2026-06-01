set shell := ["bash", "-cu"]

# List recipes.
default:
    @just --list

# Prepare the base workspace.
setup:
    @printf 'No setup required for the base template.\n'

# Show workspace status.
doctor:
    @git status --short
    @printf '\n'
    @just --list

# Run the base local gate.
check: doctor

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

# Configure tests.
test:
    @printf 'No tests configured yet.\n' >&2
    @exit 1

# Configure a build.
build:
    @printf 'No build configured yet.\n' >&2
    @exit 1
