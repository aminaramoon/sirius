# Committed spill metrics

Set `SIRIUS_EXP_SCAN_SPILL_METRICS=1` before starting Sirius and enable INFO logging.
The gate is cached on first use; all other values disable it.

Each successful GPU→HOST, GPU→DISK or HOST→DISK conversion emits one INFO message:

```text
[scan_sort_spill] event=conversion_committed batch_id=42 source_tier=GPU target_tier=HOST source_bytes=100000 gpu_uncompressed_bytes=100000 representation=compressed payload_bytes=20000 serialized_bytes=20512 compression_desired=1 compression_fallback=0
```

Fields are whitespace-separated `key=value` tokens. Byte counts are unsigned integers.

| Field | Meaning |
|---|---|
| `batch_id` | Stable cuCascade batch identifier; repeated spills can reuse it. |
| `source_tier`, `target_tier` | Actual conversion direction (`GPU`, `HOST`, `DISK`). |
| `source_bytes` | Source representation's `get_size_in_bytes()`, captured before conversion. |
| `gpu_uncompressed_bytes` | Source representation's decoded GPU footprint. |
| `representation` | Actual committed destination: `compressed` or `uncompressed`. |
| `payload_bytes` | Destination compressed payload excluding its header; otherwise native representation size. |
| `serialized_bytes` | Compressed HOST: header plus payload. Compressed DISK: `get_size_in_bytes()` (already includes header). Uncompressed: native representation size, not a file-stat measurement. |
| `compression_desired` | `1` when GPU spill compression policy offered this batch to the compressor, or an already-compressed HOST batch was written to DISK. |
| `compression_fallback` | `1` only when compression was desired but the committed destination is uncompressed. |

Compressed HOST `source_bytes` counts payload only; compressed DISK native size includes
the header. Header bytes are structural metadata, not host allocator block padding.

Messages are emitted after conversion commits, while the existing exclusive mutable
batch lock remains held. Failed reservations, failed/declined encodes, failed conversions,
upgrades and same-space no-ops do not themselves emit records. A declined encode followed
by a successful raw spill emits only the committed raw record, with fallback marked.
Instrumentation does not change freed-byte accounting, perform GPU work, or acquire
additional batch locks. Counts describe conversions, not unique batches or query totals.
