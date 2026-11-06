# Scan-batch sorting / compressed spill experiment

Run from the repository root, always via `pixi run`. Build the extension first.
The harness never builds or changes engine code. GPU workers are serialized, use
`CUDA_VISIBLE_DEVICES=0`, and each arm/query/repeat has a fresh process and isolated
spill directory. `--timeout` bounds **each** worker (default 300 seconds), including
CPU preparation/reference; a timeout kills its process group.

See [RESULTS.md](RESULTS.md) for the completed experiment and measured outcomes.

## Reproduce

```bash
# CPU-only preparation: 32M rows, 1M repeated shuffled keys, about 496 MiB parquet.
pixi run python bench/scan-sort-spill/experiment.py prepare \
  --rows 32000000 --output bench/scan-sort-spill/prepare-32m

# Run only after the baseline/sorted extension build is ready.
pixi run python bench/scan-sort-spill/experiment.py run \
  --config bench/scan-sort-spill/host-pressure.yaml \
  --output bench/scan-sort-spill/results-host --repeats 3 --timeout 300
pixi run python bench/scan-sort-spill/experiment.py run \
  --config bench/scan-sort-spill/disk-pressure.yaml \
  --output bench/scan-sort-spill/results-disk --repeats 3 --timeout 300

# Lightweight harness checks, no GPU or extension needed.
pixi run python -m unittest discover -s bench/scan-sort-spill -p 'test_*.py' -v
```

Default extension:
`sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension`.
Override with `--extension PATH`. Choose fresh output/data directories on reruns:
existing worker directories/data are intentionally not overwritten. GPU timing is
the explicit query call only; startup/model-planning overhead outside the call is
reported separately in wall time. There are no unreported GPU warmups. First-use
JIT/kernel/cache effects may therefore be visible; arm order alternates by repeat.

The CPU worker never loads Sirius. GPU workers set
`enable_duckdb_fallback=false` and explicitly execute `CALL gpu_execution(...)`.
The CALL by itself does **not** forbid CPU replay, which is why both are necessary.
Exact aggregate results must match CPU strings (integer arithmetic, no tolerance).
The sort query's DuckDB EXPLAIN must retain ORDER BY, and its GPU log must confirm
physical planning of ORDER_BY. Arm 1 must log actual successful sorting batches;
otherwise it is inconclusive, not an improvement. Arm 0 must sort zero batches.

## Inputs and memory

Synthetic fact rows have two BIGINT columns: deterministic `hash(i+1729) % 1000000`
keys in shuffled scan order and correlated `key*1000003 + hash(i+1729)%8` payloads.
The dimension contains each key exactly once. Hash behavior is DuckDB-versioned;
`data/manifest.json` records the generation SQL, version, row count, sizes and
streaming SHA256 digests. Parquet is uncompressed, row groups are 131072 rows.

Queries:

```sql
SELECT count(*),sum(f.payload) FROM fact f JOIN dim d ON f.k=d.k;
SELECT sum(payload),count(*) FROM
  (SELECT k,payload FROM fact ORDER BY k,payload);
```

All configurations cap GPU memory at 1 GiB and use 32 MiB hash batches,
64 MiB sort partitions; scan batches are 32 MiB in the regular configs and 8 MiB
in the pressure variants to leave headroom for sorting's conservative cold estimate.
They use compression minimum 1 MiB, maximum compressed fraction
0.99, encode reservation fraction 0.5. HOST uses 8 GiB **per NUMA node**, no disk.
DISK uses 128 MiB HOST per NUMA node plus a 16 GiB isolated disk tier. The latter
encourages but cannot guarantee disk writes. Disk configurations use one uring
reactor so pinned I/O staging fits the small HOST tier. The `*-pressure.yaml`
variants lower GPU monitor trigger/stop from 0.65/0.45 to 0.25/0.15: pristine
baseline smoke showed the streaming join did not spill at the higher thresholds.
The early-pressure variants exercised HOST and DISK for both queries. These are
pressure controls, not measurements of normal production performance. Dynamic filters, narrowed compressed
materialization, broadcast and resident-build join paths are disabled to avoid
hiding the scan-order effect. Edit a copied YAML to explore 2–4 GiB GPU caps,
smaller/larger scan batches, or encode reservation policies; retain effective YAMLs.
Larger GPU budgets may need larger facts to cause spills. A missing compressed
HOST/DISK commit is explicitly **inconclusive**, not zero-byte compression.

### Bounded real-data control

Prepare one explicit parquet file, limiting rows before copying. Example (confirm
that the file exists first; substitute the actual first parquet path if needed):

```bash
pixi run python bench/scan-sort-spill/experiment.py prepare \
  --source-parquet /localhome/local-aaramoon/tpch/sf1000/lineitem/part.0.parquet \
  --source-key l_suppkey --source-payload l_extendedprice --rows 32000000 \
  --data bench/scan-sort-spill/data-tpch \
  --output bench/scan-sort-spill/prepare-tpch
pixi run python bench/scan-sort-spill/experiment.py run \
  --data bench/scan-sort-spill/data-tpch --config bench/scan-sort-spill/host-pressure.yaml \
  --output bench/scan-sort-spill/results-tpch --repeats 3
```

Real-data fields are explicitly cast to BIGINT (including rounded decimal payloads),
so this is a bounded **projection control**, not an official TPC-H query/result.
Its dimension is derived by distinct key. SF1 files can be used the same way.

## Metrics and interpretation

`SIRIUS_EXP_SCAN_BATCH_SORT=0/1` selects the paired arms.
`SIRIUS_EXP_SCAN_SPILL_METRICS=1` enables committed-conversion diagnostics in both.
The harness forces a per-worker spdlog directory and DEBUG level, preserving raw
logs, stdout/stderr, plans, effective YAML, invocation/environment and result JSON.
`summary.json` holds full records; `summary.csv` gives per-direction/representation
rows with event counts, decoded-original/payload/header/committed bytes and
`committed_over_original` (lower is better). CSV scopes are `all_conversions` and
`scan_outputs`: the latter correlates `[scan_batch_output] batch_id` provenance
with committed spill events, excluding downstream partition/sort output batches.
JSON retains batch ids, row counts, byte counts and `sort_active` for audit.
Do not add the two overlapping scopes together. See [metrics.md](metrics.md).

Only **successful committed destinations** are counted; downgrade requested/freed
bytes and pre-conversion encode-attempt messages are never compression metrics.
Compressed HOST committed bytes include structural header + payload, not allocator
block padding; compressed DISK bytes are serialized header + payload. Raw values
are native representation accounting, not physical file-stat bytes. Directions
are separate: GPU→HOST→DISK writes two destinations, not two original input batches.
Repeated spills can repeat batch ids. Never sum directions as unique dataset size.

Report compression ratio **and original bytes selected for conversion** with event
counts. Asynchronous pressure can spill different batches/byte totals in each arm;
smaller committed totals alone do not establish better compression. Compare the
paired ratios and query times across repeats, separately for HOST and DISK, and
report fallbacks/uncompressed rows and inconclusive runs. No statistical-significance
claim is made by this harness. Exit status is nonzero for execution/correctness errors;
inspect statuses for inconclusive outcomes even when the command exits zero.
