#!/usr/bin/env bash
set -euo pipefail

# Run through pixi so the compiler and OpenSSL runtime match the Sirius build.
# The conda-forge OpenSSL package currently defines OPENSSL_NO_KTLS.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version=3.6.4
digest=9bffaa1ad1e07b354c21bd3324ec02fa15579f45a7d0494b3e74bc449b7333ef
prefix="$repo_dir/build/ktls-openssl"
mkdir -p "$repo_dir/build"
source_dir="$(mktemp -d "$repo_dir/build/ktls-source.XXXXXX")"
archive="$source_dir/openssl.tar.gz"
curl --fail --location --retry 3 \
  "https://github.com/openssl/openssl/releases/download/openssl-$version/openssl-$version.tar.gz" \
  --output "$archive"
printf '%s  %s\n' "$digest" "$archive" | sha256sum --check
tar -xzf "$archive" --directory "$source_dir" --strip-components=1
cd "$source_dir"
# Keep the environment's CA bundle; install libraries under a predictable lib/.
./Configure shared enable-ktls no-tests --prefix="$prefix" \
  --libdir=lib --openssldir="${CONDA_PREFIX:?Run via pixi run}/ssl"
make -j "${SIRIUS_BUILD_JOBS:-8}"
make install_sw
printf '\nkTLS OpenSSL installed at %s\nConfigure Sirius with -DSIRIUS_KTLS_OPENSSL_ROOT=%s\n' "$prefix" "$prefix"
