#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/dist"

# `pixi run` exports MODULAR_HOME; a bare `bash build/build.sh` does not, and
# without it every import fails with "unable to locate module 'std'".
if [ -z "${MODULAR_HOME:-}" ] && command -v mojo >/dev/null 2>&1; then
    export MODULAR_HOME="$(dirname "$(dirname "$(command -v mojo)")")/share/max"
fi

mojo build --emit shared-lib -I "$root/src" "$root/src/capi.mojo" -o "$root/dist/libmojo-potpourri3d.so"
