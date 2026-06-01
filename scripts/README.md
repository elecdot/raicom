# Scripts

Small helper scripts used by documented repository commands live here.

## Rules

- Scripts must be safe to run from the repository root.
- Scripts should be idempotent when possible.
- Keep scripts narrow and boring.
- Prefer scripts that support `justfile` recipes over one-off manual workflows.
- Avoid machine-specific absolute paths, local secrets, and hidden network dependencies.
- Agent-facing scripts must write caches and temporary files inside the workspace.

## Current Scripts

- `agent-env.sh`: runs a command with workspace-local cache defaults such as `.cache/` and `.cache/uv/`.
