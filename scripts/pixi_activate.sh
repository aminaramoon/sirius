#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  exit 0
fi

clang_cpp="$CONDA_PREFIX/bin/clang-cpp"
clang_pp="$CONDA_PREFIX/bin/clang++"

if [[ -x "$clang_cpp" && ! -e "$clang_pp" ]]; then
  ln -s "$clang_cpp" "$clang_pp"
fi
