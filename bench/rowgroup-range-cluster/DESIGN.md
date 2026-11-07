# Row-group range clustering + per-batch sort: experiment design

Design established before GPU execution. The final feature build/regressions subsequently passed, GPU execution was explicitly authorized, and the completed results are in [RESULTS.md](RESULTS.md). Future-tense matrix entries below document the original staging; observed counts and accounting take precedence in RESULTS.md.

## Question and estimand

Does changing split membership to combine similar Parquet key ranges **across files** improve committed spill compression beyond per-batch sorting alone? Do not assume latency improves. GPU execution must wait until the feature and its correctness checks are ready (satisfied before the recorded experiment).

Primary paired arms, identical physical file list and hashes, rows, projections, query, memory limits and batch caps:

- **A, batchsort-only:** clustering OFF, scan batch sorting ON.
- **B, cluster+batchsort:** clustering ON, scan batch sorting ON.
- Optional C, **collect+batchsort:** eager metadata collection and aggregate scheduling, original row-group order, sorting ON. This isolates metadata/scheduling changes from min/max reordering. The v1 implementation obtains admitted key metadata from scan batch sorting, so cluster-only / sorting-OFF is not a supported arm.

No arm-specific offline sorting, range partitioning, dataset copies, row filtering, or key-only permutation. The input layout may be deliberately constructed once, but both arms must read those exact files. Change scheduling/group membership only; retain complete `(k,payload)` rows including duplicates and nulls. Stable tie-breaking is `(min,max,file-id,row-group-id)`; missing/inexact metadata must conservatively use normal scheduling. Do not split with SQL predicates that duplicate/drop boundary values.

## Evidence from CPU metadata, not GPU measurements

`pixi run python bench/rowgroup-range-cluster/analyze_metadata.py` uses stock DuckDB 2.0.0 `parquet_metadata`; it never loads Sirius. Raw footers and computed quantiles are retained in `metadata-raw.json` and `metadata-summary.json`. Span is inclusive `max-min+1`; normalized span uses the full surveyed dataset domain, not a count-distinct estimate.

| Input | Files / RG | Rows | Median RG rows | Median key span / global span |
|---|---:|---:|---:|---:|
| SF1000 lineitem, supplier | 120 / 5,640 | 5,999,989,709 | 1,063,824 | 0.9999985 |
| SF1000 lineitem, orderkey | same | same | same | 0.0001773062 |
| SF1 lineitem, supplier | 1 / 6 | 6,001,215 | 1,000,061 | 1.0 |
| SF1 lineitem, orderkey | same | same | same | 0.1666667 |
| Existing 32M supplier projection | 1 / 244 | 32,000,000 | 131,072 | 0.9999872 |
| Existing 32M orderkey projection | 1 / 244 | 32,000,000 | 131,072 | 0.0040980 |
| Existing 32M hash synthetic | 1 / 245 | 32,000,000 | 131,072 | 0.9999880 |

All surveyed key RG statistics are nonmissing and exact, with zero nulls. Supplier ranges overlap essentially the whole domain, so grouping their tiny min/max variations is not a positive-mechanism test. SF1000 orderkey RGs are narrow, disjoint, and already increasing within each file (zero adjacent interval overlap in surveyed lexicographic file traversal). Numerically ordered files are the natural locality control; lexicographic filenames must not accidentally be described as globally ordered. SF1 has only one file and cannot prove cross-file grouping. Repeated orderkeys can cross boundaries in the projected file; its adjacent interval overlap fraction 0.7078 is mainly shared boundary keys, not broad ranges.

Original SF1000 RGs project to about 16.2MiB at 16 bytes/row, already larger than the 8MiB scan cap. Reordering whole original RGs may leave batches entirely within one RG, yielding no grouping mechanism. The feature must respect caps while handling oversized RGs; do not raise only treatment's cap. Use the new 8,192-row (~128KiB decoded) projected RGs so many fit in both arms' 8MiB batches. Real supplier and natural orderkey remain useful negative/locality controls, not evidence that extra improvement is expected.

## Bounded preparation, identical layout for every arm

`prepare_controls.py` creates two CPU-only datasets, each 33,554,432 rows (32Mi rows, deliberately a power of two rather than the old 32,000,000). Each has sixteen files, 256 RG/file, 8,192 rows/RG, two BIGINT columns, uncompressed Parquet (~512MiB/dataset). Logical RG index `g=256*file+rg`:

1. **Narrow shuffled ranges (positive mechanism):** `band=(17*g)%128`, `k=band*16384 + hash(i+1729)%16384`. Thirty-two RGs share each overlapping narrow range, spread across all sixteen files (two RG/bucket/file). Within RG, rows are hash-shuffled, never key-sorted. Sixty-four adjacent physical RGs cover widely separated bands; clustered 8MiB groups combine about two buckets, substantially increasing key repeats within a batch. This is stronger than merely concatenating disjoint monotone ranges, where local delta/bitpacking might already suffice. Payload is `k*1000003 + hash(i+1729)%8`, matching the earlier correlated-payload concept. Dataset has meaningful span diversity by location, not differing RG widths; the mechanism is similar min/max endpoints, not merely width ranking. Optional `narrow-independent` uses independent 50-bit hashed payloads with the same keys/layout to bound payload-correlation effects.
2. **Uniform-wide negative:** `k=hash(i+1729)%2097152`, same rows/file, decoded bytes and correlated payload expression. Every RG spans almost all keys. Metadata sorting should provide little range reduction; arbitrary tiny endpoint differences must not be sold as meaningful clustering.

Both layouts are generated once and shared unchanged by all arms. Preparation records SQL, DuckDB version, SHA256s, exact CPU aggregate references, and metadata-derived schedules. The same RG appears once in each schedule. Metadata simulation groups sixty-four RGs per 8MiB batch; actual engine split/batch logs must verify this correspondence because a schedule proxy is not an observed GPU batch. If engine requires a safety margin that admits fewer RGs, report actual membership and compare equal caps rather than silently changing the treatment.

The two real controls are now prepared by rechunking each existing 32M projection to requested 8,192-row RGs in one file, with no sorting. DuckDB 2.0.0 emits 3,903 RGs per file: median 8,192 rows, min 5,618, max 10,061. Source/output `EXCEPT ALL` both directions found zero differences; fingerprints and exact join references agree. Schedules greedily pack actual row counts under 8MiB, not fixed 64-RG groups. Supplier median simulated batch span is 9,999,969 baseline vs 9,999,176 clustered (still ~full domain); natural orderkey is 523,587 in both. These remain one-file controls, not cross-file proof. A future repacked real-range positive could distribute consecutive projected chunks round-robin over sixteen files, identically for both arms, but must be labeled offline repacking rather than a natural-order control. No full SF1000 data scan is needed.

## Actionable staged GPU matrix (not executed)

1. Feature correctness, tiny CPU-reference fixtures: duplicate keys, boundary-equal ranges, missing metadata, unsupported key, partial final RG, multiple files, overflow-safe signed min/max, nulls. Exact full-row multiset equivalence between arms; `(count,sum(payload))` alone is insufficient proof. Verify every `(file,RG)` read once and actual split bytes never exceed policy except explicitly supported oversized single RG handling.
2. Pilot: narrow and uniform-wide, A/B, HOST pressure, one join repeat = **4 GPU runs**. Use old `host-pressure.yaml` values copied into a new config, not edited in place. Require actual cross-file grouped splits, eligible/reordered=true on narrow, nontrivial span reduction on narrow, sorted scan outputs in both arms and direct scan-output compressed HOST commits. Report actual split rowgroup counts/decoded budgets rather than assuming the metadata proxy's 64-RG batches; coalescer preparation estimates include auxiliary decode memory. Otherwise stop, not proceed to a timing claim.
3. Primary: narrow-correlated, narrow-independent and uniform-wide, A/B, HOST and DISK, three alternating fresh-process repeats, join only = **36 GPU runs**. Fixed 1GiB GPU, 8MiB scan, 32MiB hash batch, pressure thresholds 0.25/0.15; HOST 8GiB per NUMA node or DISK config HOST 128MiB/node + isolated 16GiB disk, one reactor. Compression min 1MiB, max fraction .99, encode reserve .5; same disabled resident/broadcast/dynamic-filter paths as previous experiment.
4. Real controls: existing supplier and natural-order orderkey projections, A/B, two tiers, three repeats = **24 runs**, explicitly one-file controls, not cross-file proof. Add repacked real orderkey/supplier only if those datasets are prepared with checksums and a stated identical paired layout.
5. Optional collect+batchsort control C on narrow/uniform, two tiers, three repeats = **12 runs**; optional sort workload only after checking supported physical ORDER_BY. Prior orderkey sort plans rejected ordered UNION, and prior sort workloads spilled no identifiable direct scan outputs. Join is primary, not a discarded sort failure.

Start with 90s/query timeout and fresh spill directories; serialize GPU runs and alternate A/B order. No warmups hidden in final results. No GPU commands are run by either CPU script.

## Required telemetry and decision rules

Retain metadata extraction/planning time, metadata missing/exact counts, selected sort key, exact split membership `(file,RG)`, RG row counts, split decoded byte estimates, batch IDs, actual batch rows/bytes and `(min,max)` after materialization. Compare distribution of batch key spans and distinct source files/batch. A cross-file scheduler that still emits one independent batch per RG has not demonstrated this mechanism.

Report committed compressed header+payload/original decoded bytes, event count and original-byte denominator, **only direct scan-output batch IDs** for primary evidence; retain downstream totals separately. Separate GPU→HOST, GPU→DISK and HOST→DISK; do not double-count intermediate HOST plus DISK as one compression observation. Show fallbacks/raw commits, repeated batch IDs and incompletes. Different asynchronous spill selections preclude interpreting absolute committed totals alone. Query time includes metadata planning and scheduling overhead; report it independently from ratio and I/O request/read bytes.

A treatment may compress HOST batches enough that the 128MiB HOST tier never triggers DISK writes. Correct SQL with zero DISK traffic is a real traffic outcome, not an execution failure; its DISK compression ratio is undefined and the DISK ratio comparison is inconclusive, not a zero-byte ratio. Retain that pilot. If a separate forced-DISK ratio experiment is needed, copy a new config under this directory and lower HOST downgrade trigger/stop fractions symmetrically (for example .15/.08), retaining the same limits and settings for both arms. Record the pilot-based reason and effective configs; never tune only treatment. HOST compression remains primary.

Extra compression improvement = `1 - ratio(B)/ratio(A)` for paired destination/scope. Compare per-repeat ratios plus pooled byte-weighted ratios, not average of event ratios. Require CPU-reference equality, fallback disabled, both arms actually sorting, treatment actually clustering across files, and actual compressed direct scan-output commits. Otherwise inconclusive. Negative control may have no gain or regress; do not tune only treatment or select repeats. Three repeats are descriptive, not significance evidence.

For scheduler-independent codec corroboration, later encode every planned batch in each arm with the real codec, full-table coverage exactly once, fixed batch capacity and sort ON. Membership differs deliberately between arms, so compare full covered-table original/committed totals, not falsely matched per-batch rows. Existing old matched-row codec test only isolates sorting and cannot by itself test changed group membership.

## Existing harness boundaries

The old `experiment.py` supports only one `fact.parquet`, one sort-toggle pair and sort/committed-spill parsing; it remains untouched. New `bench/rowgroup-range-cluster/experiment.py` imports only its worker/metric helpers and adds explicit shared manifest file lists, immutable SHA256 checks before and after runs, independent cluster gate 0/1/collect with batchsort always ON, and raw split-membership evidence. `collect` is an optional eager-original-order control using the same aggregate scheduling as treatment, useful to separate clustering from metadata/scheduling overhead. The new harness defaults to CPU-only `cpu-check`; GPU execution requires explicit `run`. Prior join scan ratios improved sorting-only (synthetic .449→.225, supplier .322→.198), with no latency improvement; these are context, not baselines directly comparable to the new 32Mi-row multifile data.
