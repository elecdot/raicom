#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_root/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.cache/uv}"

if (($# == 0)); then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

exec "$@"
