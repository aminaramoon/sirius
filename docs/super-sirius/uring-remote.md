# S3 with io_uring and kernel TLS

`uring_remote_reactor` shares the `rest_reactor` read scheduler. Select it with
`sirius.executor.scan_manager.uring_remote.enabled: true`; the default remains
the curl REST transport. It uses `rest_n_reactors` worker threads and the existing
`rest` retry, footer, coalescing, and scan-budget settings. S3 LIST/globbing and
known-size opens continue through `rest_ioctx` via `uring_remote_ioctx`.

Both transports use the same adaptive 4–16 MiB physical GET planner, fragmented
cache fills, host reads, caller-host-to-device reads, staged device reads, and
CUDA event completion. Staging and caller buffers stay alive until all kernel
operations and device copies finish. HEAD, LIST, and the bind-time suffix footer
probe retain the existing blocking curl implementation; data GETs use io_uring.

## Build and test on EC2

Use a Linux GPU instance with io_uring enabled and kernel `CONFIG_TLS` support.
A current EC2 Linux kernel is recommended. If TLS is a loadable module, an
administrator can load it with `sudo modprobe tls`. Raise the process file
descriptor limit to cover all reactor connections, for example `ulimit -n 8192`.

The Pixi OpenSSL package currently disables kTLS. Build a private OpenSSL 3.6.4
with `enable-ktls`; the script verifies the pinned upstream SHA-256 and installs
under the repository's build directory, without changing the Pixi environment:

```bash
pixi install
pixi run build-ktls-openssl
# After the normal Sirius build configuration has been generated:
pixi run cmake -S duckdb -B build/release \
  -DSIRIUS_KTLS_OPENSSL_ROOT="$PWD/build/ktls-openssl"
pixi run cmake --build build/release \
  --target duckdb sirius_loadable_extension sirius_unittest s3_throughput_test -j 8
pixi run test-uring-remote
```

The test task creates a temporary localhost certificate, requires real kTLS
HTTPS reads with 80 concurrent connections, checks hostname rejection, and runs the HTTP, retry,
cancellation, concurrency, and host/device tests. It needs neither AWS credentials
nor Docker. GPU cases report a skip when no CUDA device is available.

For a fresh checkout, initialize submodules and run the repository's usual build
setup first (`git submodule update --init --recursive`, `pixi run make`). The
OpenSSL option applies to the launcher, loadable extension, and linked tests.
Rebuild the launcher too: an older executable can preload its non-kTLS OpenSSL
before loading the extension.
When launching a process that loads other OpenSSL users first, explicitly choose
the same private runtime:

```bash
export LD_LIBRARY_PATH="$PWD/build/ktls-openssl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SIRIUS_CONFIG_FILE=/path/to/private-sirius-uring.yaml
```

Copy `test/io/sirius_uring_remote_ec2.yaml` to a private file (`chmod 600`). Set
its endpoint and region for your bucket, then fill in `object_store.access_key`,
`secret_key`, and `session_token` (required for temporary credentials). Do not
commit real credentials. The existing signer does not read AWS environment
variables or fetch/refresh instance-role credentials itself.

```bash
pixi run build/release/extension/sirius/test/io/s3_throughput_test \
  --config "$SIRIUS_CONFIG_FILE" \
  --bucket s3://YOUR_BUCKET/YOUR_PREFIX \
  --n-reactors 1 --max-connection 256 --n-files 16 --per-file 1 --repeat 3
```

The benchmark now honors transport selection from `--config`. Its connection and
reactor command-line flags override the corresponding YAML settings. Compare
64/128/256/512 connections on the same files, instance, region, and reactor count;
select `enabled: false` for the curl baseline. This change does not assert a
throughput improvement without measuring the EC2 workload.

## Transport settings

| `scan_manager.uring_remote` key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Route S3 data reads through io_uring. |
| `max_connections` | `256` | Connections per worker; range 1–4096. |
| `queue_depth` | `1024` | SQ entries; at least twice the connection count, at most 32768. |
| `receive_buffer_bytes` | `256 KiB` | Maximum bytes per receive; range 16 KiB–4 MiB. |
| `max_header_bytes` | `64 KiB` | Total header/trailer budget per response; range 1 KiB–1 MiB. |
| `allow_plaintext` | `false` | Explicit HTTP test-endpoint opt-in. HTTPS always requires kTLS. |

Connection handshakes run concurrently on nonblocking sockets, with readiness
polled through io_uring. The reactor verifies certificate trust, hostname, and
kTLS **RX and TX** before sending a GET. It logs `verified kTLS RX=1 TX=1` once per
worker. `/proc/net/tls_stat` exposes kernel TLS counters for independent checking.
There is no HTTPS fallback to userspace decryption. Missing kernel/OpenSSL support
fails reads with a diagnostic.

The initial protocol is HTTP/1.1 over TLS 1.2 with ECDHE AES-GCM. Restricting TLS to
1.2 avoids TLS 1.3 post-handshake KeyUpdate/session-ticket handling when bypassing
`SSL_read`/`SSL_write`. TLS control records terminate the connection and enter the
existing retry policy. HTTP/2, TLS 1.3-only endpoints, proxies, and redirects are
not supported by this transport; configure the final regional S3 endpoint.

Normal Content-Length response bodies are received directly into the caller's
iovecs or pinned staging blocks using `IORING_OP_RECVMSG`. A bounded scratch
buffer, allocated lazily per used connection, handles headers, chunked transfer
framing, and close-delimited responses.
TLS receives set `IOSQE_ASYNC` so software decryption runs in kernel io_uring
workers instead of inline on the reactor's submission thread. The pool therefore
has one userspace reactor thread plus kernel I/O workers; kTLS does not imply NIC
hardware TLS offload on an EC2 instance.
Header parsing uses the pinned header-only
[cpp-httplib](https://github.com/yhirose/cpp-httplib/tree/v0.54.1) package.
The vcpkg overlay pins the same version without optional client dependencies.
The worker caches DNS addresses for 60 seconds and rotates them across new
connections, trying the remaining addresses after a connect failure. The first
lookup and cache refresh use the system resolver and can briefly block that
worker. Idle connections expire using `rest.conn_max_age_s`;
TCP keepalive covers HTTP/1.1, which has no HTTP/2 upkeep PING.

Kernel behavior and activation checks follow the upstream
[Linux kTLS documentation](https://docs.kernel.org/networking/tls.html) and
[OpenSSL BIO controls](https://docs.openssl.org/3.5/man3/BIO_ctrl/).
