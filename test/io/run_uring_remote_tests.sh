#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
prefix="$repo_dir/build/ktls-openssl"
if [[ ! -f "$prefix/lib/libssl.so" ]]; then
  printf 'Run pixi run build-ktls-openssl and rebuild with SIRIUS_KTLS_OPENSSL_ROOT first.\n' >&2
  exit 1
fi
cert_dir="$(mktemp -d)"
trap 'rm -f -- "$cert_dir/cert.pem" "$cert_dir/key.pem"; rmdir -- "$cert_dir"' EXIT
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$prefix/bin/openssl" req -x509 -newkey rsa:2048 -noenc -days 1 \
  -subj /CN=localhost -addext subjectAltName=DNS:localhost \
  -keyout "$cert_dir/key.pem" -out "$cert_dir/cert.pem" 2>/dev/null
export SIRIUS_TEST_KTLS_CERT="$cert_dir/cert.pem"
export SIRIUS_TEST_KTLS_KEY="$cert_dir/key.pem"
"${SIRIUS_UNITTEST_BIN:-$repo_dir/build/release/extension/sirius/test/cpp/sirius_unittest}" \
  '[uring_remote]' "$@"
