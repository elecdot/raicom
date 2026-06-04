# Minimal Workspace Template

A lightweight starting point for repositories that need clear commands, durable documentation, and agent-safe defaults.

Update the following files when bootstrapping a new project:
```
README.md       # purpose / repo map /commands
CONTEXT.md      # project language and scope
AGENTS.md       # validation guidance / project-specific warnings
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

## Bootstrap Checklist

- Replace this README's template purpose, current state, and open loops with the new project's concrete purpose and next task.
- Rewrite `CONTEXT.md` for the new project's language and scope once the project enters real development.
- Update `AGENTS.md` with project-specific validation commands, warnings, and working constraints.
- Replace failing placeholder `justfile` recipes with real commands as tooling appears, or leave them clearly marked as not configured.
- Run `just setup` and `just check` before starting regular work.

## Repo Map

| Path | Purpose |
| --- | --- |
| `CONTEXT.md` | Canonical language for this template's purpose and scope. |
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

Placeholder recipes such as `fmt`, `lint`, `test`, and `build` intentionally fail until configured so they cannot be mistaken for working quality gates.

## Current State

- Base workspace commands are available through `just`.
- No language stack, formatter, linter, tests, build, CI, or docs site is configured.

## Open Loops

- [ ] After copying this template, replace template language with the new project's purpose, commands, and first real task.
