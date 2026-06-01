# Minimal Workspace Template

A lightweight starting point for repositories that need clear commands, durable documentation, and agent-safe defaults.

Update the following files when bootstrapping a new project:
```
README.md       # purpose / repo map /commands
AGENTS.md       # validation matrix / domain-specific warnings
justfile        # actual commands
```

## Quick Start

Prerequisites:

- Git
- Bash
- `just`

```bash
just setup
just check
```

After copying this template, update this README with the new repository purpose and replace placeholder recipes in `justfile` as real tooling appears.

## Repo Map

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Working contract for coding agents and human-assisted automation. |
| `justfile` | Main command entry point. |
| `scripts/` | Small helper scripts used by documented commands. |
| `docs/` | Durable project documentation and conventions. |
| `docs/adr/` | Optional architecture decision records. |
| `tmp/` | Local scratch space; only its README and ignore rules are tracked. |

## Commands

Prefer repository commands over ad hoc long commands:

```bash
just doctor       # show workspace status and recipes
just check        # run the base local gate
just agent <cmd>  # run a command with workspace-local caches
```

The base template intentionally does not configure formatters, linters, tests, CI, docs sites, or language-specific dependencies. Add those only when the repository needs them, then promote the recurring command into `justfile`.

## Current State

- `<short status>`
- `<known limitation>`

## Open Loops

- [ ] `<next concrete task>`
