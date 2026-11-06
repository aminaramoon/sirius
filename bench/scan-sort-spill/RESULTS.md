# Results: scan-batch sorting and compressed spill

## Conclusion

**Per-scan-batch sorting improved committed compressed join spill ratios, but did
not improve join latency in these runs.** Synthetic correlated data needed about
50% fewer destination bytes per decoded byte; the real supplier-key projection
needed about 39% fewer. An already-ordered real orderkey control showed no gain.

No sort workload spilled a batch identifiable as a direct scan output. Their
compressed spills were downstream sort outputs, so they provide **no direct
evidence** of improved scan-output compression. Do not interpret their variable
spill totals or timings as proof of that mechanism.

## Setup and coverage

Worktree: `/localhome/local-aaramoon/workspace/sirius-scan-sort-spill`, branch
`experiment/scan-sort-spill`, based on freshly fetched upstream `main`
`834748f86d53e1fa145956169fbbb21fc8e5a7a1`. Submodules were initialized at that
commit's pinned revisions. The pristine full build and baseline spill tests passed
before tracked source edits; the original dirty worktree was left untouched.

- One visible A100, `CUDA_VISIBLE_DEVICES=0`; shared `host-pressure.yaml` or
  `disk-pressure.yaml` for both arms; 1 GiB GPU, 8 MiB scan batches, 32 MiB hash
  batches, GPU monitor trigger/stop 0.25/0.15. HOST-only has 8 GiB pinned capacity per NUMA node;
  DISK has 128 MiB HOST per NUMA node, one uring reactor and an isolated 16 GiB disk tier.
- Spill compression: enabled, minimum 1 MiB, maximum fraction 0.99, encode reserve
  0.5. Resident-build/broadcast joins, dynamic filters and compressed narrowed
  materialization disabled. See effective YAML in every run.
- Three fresh-process repeats per arm, alternating arm order, 90-second per-worker
  timeout. Timings include explicit query execution/planning, not extension startup.
  No hidden GPU warmups; pilot runs precede the final matrix.
- CPU references never load Sirius. GPU uses explicit `CALL gpu_execution` with
  `enable_duckdb_fallback=false`. All successful results exactly match CPU.
- Final coverage: **60 successful GPU runs, 6 unsupported GPU plans, no timeout or
  inconclusive runs**. The 48 main synthetic/supplier runs all passed. The 12
  orderkey join controls passed. Both arms of the orderkey HOST sort control
  rejected ordered UNION (`allow_out_of_order=false`), six attempts total. This is
  an existing unsupported plan shape, not a correctness mismatch. DISK orderkey
  sort was consequently not attempted. No CPU fallback was enabled.
- All sorted-arm successful queries logged real sorted batches/rows; baseline
  logged none. Supported sort queries retained ORDER BY in CPU and GPU plans.

### Inputs

Each fact contains 32,000,000 two-BIGINT rows in uncompressed parquet. Synthetic
keys are deterministic shuffled repeats over 1,000,000 values with strongly
key-correlated payload; dimension keys are unique. Real projections use the first
32M rows of `/localhome/local-aaramoon/tpch/sf1000/lineitem/part.0.parquet`, with
`l_suppkey` or `l_orderkey` as key and `l_extendedprice` **cast to BIGINT** as payload;
their dimensions are distinct projected keys. These are bounded projection
experiments, **not official TPC-H queries/results**, and the decimal cast changes
the original payload values. The orderkey control has zero decreasing key
transitions. Dataset manifests retain SQL, DuckDB 2.0.0 version, sizes and SHA256s.

## Primary evidence: committed direct scan-output bytes

Only successful destination conversions matching `[scan_batch_output] batch ids
are counted. Each row pools all three repeats: ratio = sum(committed header +
payload bytes) / sum(decoded original bytes). These are **not source-freed
summaries or encode attempts**. Async pressure can choose different original
bytes/events, shown explicitly below. Events may repeat a batch; they are not
unique dataset counts.

HOST rows below are HOST-only configurations. DISK rows combine GPU→DISK and
HOST→DISK **only at the DISK destination**, never adding the intermediate HOST
write. The aggregate JSON also retains the separate HOST stage of DISK runs and
all individual conversion directions.

| Input / destination | Arm | Events | Original bytes | Committed bytes | Committed/original |
|---|---|---:|---:|---:|---:|
| Synthetic / HOST | Baseline | 117 | 747,962,752 | 335,664,421 | 0.448772 |
| Synthetic / HOST | Sorted | 131 | 837,222,784 | 188,205,963 | 0.224798 |
| Synthetic / DISK | Baseline | 142 | 911,568,768 | 411,488,974 | 0.451407 |
| Synthetic / DISK | Sorted | 140 | 895,320,448 | 201,513,228 | 0.225074 |
| Supplier / HOST | Baseline | 105 | 667,020,480 | 214,768,491 | 0.321982 |
| Supplier / HOST | Sorted | 112 | 711,163,360 | 140,620,946 | 0.197734 |
| Supplier / DISK | Baseline | 103 | 652,564,720 | 210,114,173 | 0.321982 |
| Supplier / DISK | Sorted | 117 | 742,680,912 | 146,853,947 | 0.197735 |
| Ordered orderkey / HOST | Baseline | 113 | 713,546,736 | 116,603,465 | 0.163414 |
| Ordered orderkey / HOST | Sorted | 112 | 709,284,528 | 115,908,012 | 0.163415 |
| Ordered orderkey / DISK | Baseline | 107 | 675,849,664 | 110,445,173 | 0.163417 |
| Ordered orderkey / DISK | Sorted | 108 | 686,005,200 | 112,105,183 | 0.163417 |

All direct scan-output events above committed compressed representations, with
zero compression fallback events. Other operators also produced raw spills,
especially sub-1-MiB sort intermediates; those are retained separately in JSON/CSV,
not silently dropped from the records or mislabeled compressed. Compressed HOST
bytes count structural headers but not allocator block padding; raw DISK native
accounting is not a compressed-file measurement.

## Median end-to-end query-call time, seconds

| Input / tier config | Join baseline | Join sorted | Sort baseline | Sort sorted |
|---|---:|---:|---:|---:|
| Synthetic / HOST | 0.4321 | 0.4527 | 0.6301 | 0.6345 |
| Synthetic / DISK | 0.4715 | 0.4893 | 0.7987 | 0.7577 |
| Supplier / HOST | 0.5241 | 0.5676 | 0.6058 | 0.6414 |
| Supplier / DISK | 0.5457 | 0.5768 | 0.6902 | 0.6930 |
| Ordered orderkey / HOST | 0.5071 | 0.5210 | Unsupported | Unsupported |
| Ordered orderkey / DISK | 0.5426 | 0.5665 | Not attempted | Not attempted |

**No join speedup:** every sorted join median was slower. Sorting adds work and
opted-in scans disable converted-table steal for retry safety, so timing includes
that intentional overhead. Three repeats do not support a statistical-significance
claim. Sort timings are not direct scan-compression evidence; async pressure and
already-sorted downstream spill selection vary across arms.

## Matched-row codec control

A separate codec test encodes the **same 1,000,000-row, 16,000,000-byte batch** in
original and per-batch-sorted order, then commits to HOST and DISK. HOST and DISK
serialized bytes agree exactly; these eliminate async batch-selection differences.
Results come from `codec-results.json` / `codec-results.csv`.

| Matched batch | Original serialized bytes | Sorted serialized bytes | Original/raw | Sorted/raw |
|---|---:|---:|---:|---:|
| Shuffled, correlated payload | 4,775,919 | 157,615 | 0.298495 | 0.009851 |
| Shuffled, independent payload | 9,650,919 | 7,949,189 | 0.603182 | 0.496824 |
| Already sorted | 157,615 | 157,615 | 0.009851 | 0.009851 |

This codec fixture differs from the SQL dataset's key distribution and batch size;
its larger correlated gain is not the end-to-end SQL reduction. Encode-only times
in its raw records exclude scan sorting overhead and are not query speedups.

## Experimental implementation

`SIRIUS_EXP_SCAN_BATCH_SORT=1` opts in; default behavior is unchanged. The planner
pass in `src/planner/scan_sort_plan_pass.cpp` traces downstream join/order keys
through proven direct-reference projection/filter mappings to each scan. The GPU
scan in `src/op/scan/sirius_gpu_scan_operator.cpp` sorts and gathers all columns of
one completed batch together, never globally sorting or mutating cached input.
Existing global sorts remain in place. Unsupported key provenance/types, unsafe
order-sensitive plan shapes, and late-materialization scans are conservatively
skipped. Initial support covers fixed-width integer/decimal/date/time keys, not
floating-point, string, or nested keys.

`SIRIUS_EXP_SCAN_SPILL_METRICS=1` enables post-commit diagnostics in
`src/data/convertible_data_batch.hpp`; existing source-freed accounting is unchanged.
Cold sorting workspace estimates are conservative heuristics, not a proven cuDF
allocation bound; retry-safe source ownership and allocation-failure tests remain
important. These changes are experimental, not a recommendation to enable sorting
for all production scans.

## Artifacts, validation and reproduction

- `aggregate-results.json`: pooled direct scan metrics by destination and original
  direction; all GPU-origin encoding/conversion scope; actual HOST→DISK traffic;
  raw/failed/fallback counts; per-arm original bytes; medians and repeat times;
  matched codec records; all six explicit unsupported-plan errors.
- `results-final-{synthetic,tpch,tpch-orderkey}-{host,disk}/`: raw per-run logs,
  invocation/environment, extension SHA256, plans, effective YAMLs, result JSON,
  summary JSON/CSV. Generated run directories/data remain on disk and are ignored.
- `results-pilot-*` and `results-pristine-*` are **excluded** from final pooling.
- Focused C++ sorting: 290 assertions / 14 cases passed; matched codec: 214
  assertions passed; ordinary spill compression: 67 assertions passed. Harness
  CPU unit tests, Python compilation and YAML parsing passed.

Run matrix commands from [README.md](README.md), adding `--data data-tpch` or
`--data data-tpch-orderkey` with the full bench-relative path for real controls.
For the orderkey DISK control pass `--queries join`. Generate the aggregate with:

```bash
pixi run python bench/scan-sort-spill/analyze.py \
  --output bench/scan-sort-spill/aggregate-results.json
```

Scopes overlap: **never add direct scan scope to all-conversion scope, or HOST
stage bytes to DISK bytes as if both were unique original input**. The mechanism
is supported by committed direct scan join ratios plus matched-row controls, not
by a claim of faster queries or compression gains for arbitrary payloads.
