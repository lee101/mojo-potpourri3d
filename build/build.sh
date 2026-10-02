#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# src/ported.mojo is the single entry point; -I src resolves the geometrycentral-mirroring
# module imports that hold the kernels. One compilation unit for the whole library.
mojo build --emit shared-lib -I "$repo_dir/src" \
    "$repo_dir/src/ported.mojo" \
    -o "$repo_dir/dist/libmojo-potpourri3d.so"
