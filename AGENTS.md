# AGENTS.md

This is the repository-level working contract for coding agents and 
human-assisted automation.

## Repository Orientation

- Read `README.md` first.
- Read relevant directory-level README files before editing inside a subtree.
- Prefer documented `just` recipes over ad hoc workflow shapes.
- Keep changes small, reviewable, and scoped to the request.
- Preserve user changes already present in the worktree.
- The root `README.md` serves as the primary entry point.
Additionally, each directory maintains its own `README.md` file to index its 
contents and define local conventions for that specific subtree.

## Commands And Tooling

- Treat `justfile` as the main command interface.
- Add recurring commands to `justfile` and document them briefly.
- Extract complex or multi-step workflow logic into dedicated helper scripts.
- Use `scripts/agent-env.sh` when a command may write cache or local tool 
state.
- Temporary tool caches may live under `.cache/`; do not commit them.

## Engineering Principles

- YAGNI: do not add abstractions, services, frameworks, CI, docs sites, or 
config until the current task needs them.
- DRY pragmatically: remove meaningful duplication, but do not introduce 
indirection that makes small code harder to understand.
- Prefer simple interfaces with deeper implementation over wide shallow 
modules.
- For bug fixes and behavior changes, prefer a narrow reproduce -> fix -> 
regression-test loop.
- For testable work, prefer one vertical slice at a time: one failing 
behavior test, minimal implementation, then refactor.

## Bulk Change Safety

- Before bulk renames, moves, metadata changes, formatting sweeps, or 
generated rewrites, produce a change plan with scope and examples.
- Do not perform destructive refactors unless explicitly requested.
- Prefer additive and reversible changes.

## Validation

- Run the narrowest useful check first, then broader checks when the change 
affects shared behavior.
- If a command is missing, blocked, or too expensive, state what was not run 
and why.

## Definition Of Done

A change is done when:

- The requested behavior is implemented end to end, not only sketched.
- Relevant docs, README entries, commands, or conventions are updated.
- Relevant validation was run, or the reason for skipping is stated.
- No unrelated rewrites, generated artifacts, local secrets, or temporary 
cache files are included.
- Remaining risks, unknowns, or follow-up work are explicit.