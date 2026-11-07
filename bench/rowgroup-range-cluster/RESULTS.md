# Results: row-group range clustering plus per-batch sorting

All arms read identical checksummed physical inputs and sort every admitted scan batch. Arm 0 is batchsort-only; arm 1 additionally groups similar row-group ranges across files. `collect` performs eager metadata/aggregate scheduling without range reordering. This is not a comparison with the older experiment’s different input layout.

**Finding:** narrow overlapping-range synthetic inputs compress substantially better with clustering, but treatment is much slower. Uniform-wide and natural-order controls do not support a universal compression improvement. Three repeats are descriptive, not statistical significance evidence.

See [README.md](README.md) for reproduction and [DESIGN.md](DESIGN.md) for the controls and staged design. Branch: `experiment/rowgroup-range-cluster`, based on prior feature commit [`f7c38a4c`](https://github.com/aminaramoon/sirius/commit/f7c38a4c) pushed to the personal fork. The binary dirty-source patch in every final output captures the subsequent uncommitted engine implementation.

Synthetic inputs each contain 33,554,432 two-BIGINT rows in 16 files / 4,096 rowgroups of 8,192 rows. Real supplier/orderkey controls each preserve the existing 32,000,000-row projection in one rechunked file / 3,903 rowgroups (median 8,192 rows). Payload is `l_extendedprice` rounded by DuckDB’s BIGINT cast: these are bounded projected controls, not full TPC-H tables or official TPC-H queries.

## Validation and setup

Prerequisite engine tests: new range clustering 599 assertions / 14 cases; existing scan sorting 290/14, spilling 67/5, and Parquet sizing 951/20, totaling 1,907 assertions / 53 cases. Six Python harness unit tests also pass.

77 successful GPU query processes: tiny exact-full-row validation (3), pressure pilots (8), final primary matrix (60), and HOST collect controls (6). No failed/inconclusive runs or timeouts. Tiny 32,768-row four-file join matches the exact CPU tuple multiset in 0/collect/1. Every final result matches its CPU reference with fallback disabled; row-group contract inventories match across arms/repeats, and input hashes remain unchanged.

Source/build frozen throughout. One A100; GPU cap 1GiB, scan cap 8MiB, hash cap 32MiB; shared original HOST/DISK pressure configs, thresholds .25/.15. HOST has 8GiB/node; DISK has 128MiB HOST/node and isolated 16GiB disk. No pressure tuning was necessary: all DISK runs committed compressed direct scan outputs. Fresh subprocess per arm/repeat, alternating main A/B order, no hidden warmups. Collect controls were measured afterward, not interleaved with A/B.

The experimental engine implementation is in `src/planner/sirius_physical_plan_generator.cpp`, `src/op/scan/parquet_scan_core.{cpp,hpp}`, and associated scan descriptor/provider plumbing. `SIRIUS_EXP_ROWGROUP_RANGE_CLUSTER=1` enables range reordering; `collect` collects metadata with original order; 0 retains baseline scheduling. All arms set `SIRIUS_EXP_SCAN_BATCH_SORT=1`, `SIRIUS_EXP_ROWGROUP_RANGE_AUDIT=1` (observer-only in baseline), and `SIRIUS_EXP_SCAN_SPILL_METRICS=1`. Eligibility is conservative: plain Parquet and first admitted signed INT32/INT64 key. Missing/unsupported key statistics cause whole-aggregate original-order fallback, never dropping rowgroups.

Provenance records extension and installed dynamic library SHA256 plus binary dirty-working-tree patch (HEAD alone is the previous feature), hashes of harnesses, effective configs, commands, and environment. Installed library SHA256: `5a4f9f3eb4b68a63378aeb32e202d97801412b21f6ceb2160eba11aaa8ec8e43`. Tiny/pilot library+patch provenance was captured immediately afterward and marked post-run; final provenance was captured before execution.

## Primary: all direct scan outputs, pooled across three repeats

Ratios below include **raw and compressed** successful destination commits. Show compressed-only ratios separately; raw tails are not dropped. Repeated spills count repeated traffic, not unique dataset size. Original-byte denominators vary because asynchronous pressure selects different batches. Never sum HOST and DISK directions as unique input.

| Input | Destination | Arm | Events | Original bytes | Stored bytes | All stored/original | Compressed-only | Raw events | Fallbacks | Median query s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| narrow-correlated | DISK | 0 | 160 | 1,338,882,048 | 298,711,328 | 0.223105 | 0.223105 | 0 | 0 | 0.5883 |
| narrow-correlated | DISK | 1 | 108 | 905,748,480 | 41,321,132 | 0.045621 | 0.045621 | 0 | 0 | 2.8844 |
| narrow-correlated | HOST | 0 | 126 | 1,047,937,024 | 233,475,141 | 0.222795 | 0.222696 | 1 | 0 | 0.4596 |
| narrow-correlated | HOST | 1 | 160 | 1,329,854,464 | 60,512,191 | 0.045503 | 0.045407 | 1 | 0 | 3.0586 |
| narrow-correlated | HOST | collect | 123 | 1,022,003,200 | 227,880,970 | 0.222975 | 0.222874 | 1 | 0 | 0.4439 |
| narrow-independent | DISK | 0 | 180 | 1,503,903,744 | 634,427,020 | 0.421853 | 0.421853 | 0 | 0 | 0.6697 |
| narrow-independent | DISK | 1 | 115 | 964,454,400 | 381,008,790 | 0.395051 | 0.395051 | 0 | 0 | 3.7709 |
| narrow-independent | HOST | 0 | 129 | 1,046,657,024 | 441,122,835 | 0.421459 | 0.421165 | 4 | 0 | 0.4861 |
| narrow-independent | HOST | 1 | 155 | 1,279,797,248 | 507,143,298 | 0.396268 | 0.396143 | 2 | 0 | 3.8082 |
| real-orderkey | DISK | 0 | 115 | 943,827,536 | 154,207,415 | 0.163385 | 0.163385 | 0 | 0 | 1.5631 |
| real-orderkey | DISK | 1 | 114 | 935,609,904 | 152,864,586 | 0.163385 | 0.163385 | 0 | 0 | 1.5917 |
| real-orderkey | HOST | 0 | 114 | 935,609,904 | 152,864,286 | 0.163385 | 0.163385 | 0 | 0 | 1.5118 |
| real-orderkey | HOST | 1 | 118 | 968,285,216 | 158,202,499 | 0.163384 | 0.163384 | 0 | 0 | 1.5139 |
| real-supplier | DISK | 0 | 112 | 918,962,256 | 179,861,332 | 0.195722 | 0.195722 | 0 | 0 | 1.6225 |
| real-supplier | DISK | 1 | 115 | 945,054,672 | 184,902,901 | 0.195653 | 0.195653 | 0 | 0 | 1.6728 |
| real-supplier | HOST | 0 | 112 | 919,093,328 | 179,885,796 | 0.195721 | 0.195721 | 0 | 0 | 1.5263 |
| real-supplier | HOST | 1 | 108 | 887,362,384 | 173,612,844 | 0.195650 | 0.195650 | 0 | 0 | 1.5453 |
| uniform-wide | DISK | 0 | 163 | 1,362,880,512 | 325,268,443 | 0.238662 | 0.238662 | 0 | 0 | 0.6002 |
| uniform-wide | DISK | 1 | 109 | 914,135,040 | 217,512,117 | 0.237943 | 0.237943 | 0 | 0 | 2.9959 |
| uniform-wide | HOST | 0 | 127 | 1,038,782,464 | 247,833,308 | 0.238581 | 0.238289 | 3 | 0 | 0.4537 |
| uniform-wide | HOST | 1 | 168 | 1,397,204,992 | 333,402,111 | 0.238621 | 0.238548 | 1 | 0 | 2.8690 |
| uniform-wide | HOST | collect | 131 | 1,080,068,096 | 257,664,617 | 0.238563 | 0.238376 | 2 | 0 | 0.4689 |

## Paired extra compression and latency

| Input | Tier | Extra reduction: 1 - ratio(1)/ratio(0) | Runtime 1 / 0 |
|---|---|---:|---:|
| narrow-correlated | HOST | 79.576% | 6.65x |
| narrow-correlated | DISK | 79.552% | 4.90x |
| narrow-independent | HOST | 5.977% | 7.83x |
| narrow-independent | DISK | 6.353% | 5.63x |
| real-orderkey | HOST | 0.000% | 1.00x |
| real-orderkey | DISK | 0.000% | 1.02x |
| real-supplier | HOST | 0.036% | 1.01x |
| real-supplier | DISK | 0.035% | 1.03x |
| uniform-wide | HOST | -0.017% | 6.32x |
| uniform-wide | DISK | 0.301% | 4.99x |

Collect HOST controls retain baseline-like compression and subsecond medians, unlike reordered treatment. This points to range reordering/interleaved reads rather than eager metadata collection alone as the dominant difference, without claiming isolated causal profiling. Correlated narrow data shows ~80% extra storage reduction; independent payload shows only ~6%, emphasizing payload-correlation dependence.

## Fact-only scan outputs

Dimensions also scan/spill; the following isolates fact scan batches using operator provenance. The aggregate JSON additionally retains dimension-only and all-operator metrics, every conversion direction, per-repeat ratios and event denominators.

| Input | Tier | Arm | Fact events | Fact original | Fact stored | Fact all ratio |
|---|---|---|---:|---:|---:|---:|
| narrow-correlated | DISK | 0 | 160 | 1,338,882,048 | 298,711,328 | 0.223105 |
| narrow-correlated | DISK | 1 | 108 | 905,748,480 | 41,321,132 | 0.045621 |
| narrow-correlated | HOST | 0 | 126 | 1,047,937,024 | 233,475,141 | 0.222795 |
| narrow-correlated | HOST | 1 | 160 | 1,329,854,464 | 60,512,191 | 0.045503 |
| narrow-correlated | HOST | collect | 123 | 1,022,003,200 | 227,880,970 | 0.222975 |
| narrow-independent | DISK | 0 | 180 | 1,503,903,744 | 634,427,020 | 0.421853 |
| narrow-independent | DISK | 1 | 115 | 964,454,400 | 381,008,790 | 0.395051 |
| narrow-independent | HOST | 0 | 129 | 1,046,657,024 | 441,122,835 | 0.421459 |
| narrow-independent | HOST | 1 | 155 | 1,279,797,248 | 507,143,298 | 0.396268 |
| real-orderkey | DISK | 0 | 115 | 943,827,536 | 154,207,415 | 0.163385 |
| real-orderkey | DISK | 1 | 114 | 935,609,904 | 152,864,586 | 0.163385 |
| real-orderkey | HOST | 0 | 114 | 935,609,904 | 152,864,286 | 0.163385 |
| real-orderkey | HOST | 1 | 118 | 968,285,216 | 158,202,499 | 0.163384 |
| real-supplier | DISK | 0 | 112 | 918,962,256 | 179,861,332 | 0.195722 |
| real-supplier | DISK | 1 | 115 | 945,054,672 | 184,902,901 | 0.195653 |
| real-supplier | HOST | 0 | 112 | 919,093,328 | 179,885,796 | 0.195721 |
| real-supplier | HOST | 1 | 108 | 887,362,384 | 173,612,844 | 0.195650 |
| uniform-wide | DISK | 0 | 163 | 1,362,880,512 | 325,268,443 | 0.238662 |
| uniform-wide | DISK | 1 | 109 | 914,135,040 | 217,512,117 | 0.237943 |
| uniform-wide | HOST | 0 | 127 | 1,038,782,464 | 247,833,308 | 0.238581 |
| uniform-wide | HOST | 1 | 168 | 1,397,204,992 | 333,402,111 | 0.238621 |
| uniform-wide | HOST | collect | 131 | 1,080,068,096 | 257,664,617 | 0.238563 |

## Actual split geometry (fact and dimension separate)

Representative first repeat per case; complete repeat telemetry is in JSON. These are actual packed split logs, not the earlier 64-RG/8MiB metadata proxy. Auxiliary preparation accounting reduces synthetic fact splits to 63 RGs. Baseline already coalesces a few adjacent files; treatment groups all 16.

| Input / tier | Arm | Fact RG/split: count | Fact files/split: count | Median fact key span | Dimension RG/split: count |
|---|---|---|---|---:|---|
| narrow-correlated / DISK | 0 | {63: 65, 1: 1} | {1: 51, 2: 15} | 2,031,615 | {126: 2, 4: 1} |
| narrow-correlated / DISK | 1 | {63: 65, 1: 1} | {16: 65, 1: 1} | 49,150 | {126: 2, 4: 1} |
| narrow-correlated / HOST | 0 | {63: 65, 1: 1} | {1: 51, 2: 15} | 2,031,614 | {126: 2, 4: 1} |
| narrow-correlated / HOST | 1 | {63: 65, 1: 1} | {16: 65, 1: 1} | 49,150 | {126: 2, 4: 1} |
| narrow-correlated / HOST | collect | {63: 65, 1: 1} | {1: 51, 2: 15} | 2,031,615 | {126: 2, 4: 1} |
| narrow-independent / DISK | 0 | {63: 65, 1: 1} | {1: 51, 2: 15} | 2,031,615 | {126: 2, 4: 1} |
| narrow-independent / DISK | 1 | {63: 65, 1: 1} | {16: 65, 1: 1} | 49,150 | {126: 2, 4: 1} |
| narrow-independent / HOST | 0 | {63: 65, 1: 1} | {1: 51, 2: 15} | 2,031,615 | {126: 2, 4: 1} |
| narrow-independent / HOST | 1 | {63: 65, 1: 1} | {16: 65, 1: 1} | 49,150 | {126: 2, 4: 1} |
| real-orderkey / DISK | 0 | {63: 35, 62: 27, 24: 1} | {1: 63} | 515,069 | {124: 3, 125: 4, 97: 1} |
| real-orderkey / DISK | 1 | {63: 35, 62: 27, 24: 1} | {1: 63} | 515,069 | {125: 3, 124: 4, 98: 1} |
| real-orderkey / HOST | 0 | {63: 35, 62: 27, 24: 1} | {1: 63} | 515,069 | {124: 3, 125: 4, 97: 1} |
| real-orderkey / HOST | 1 | {63: 35, 62: 27, 24: 1} | {1: 63} | 515,069 | {125: 3, 124: 4, 98: 1} |
| real-supplier / DISK | 0 | {63: 35, 62: 27, 24: 1} | {1: 63} | 9,999,970 | {125: 1, 124: 8, 20: 1} |
| real-supplier / DISK | 1 | {63: 40, 62: 22, 19: 1} | {1: 63} | 9,999,164 | {124: 5, 125: 4, 17: 1} |
| real-supplier / HOST | 0 | {63: 35, 62: 27, 24: 1} | {1: 63} | 9,999,970 | {125: 1, 124: 8, 20: 1} |
| real-supplier / HOST | 1 | {63: 40, 62: 22, 19: 1} | {1: 63} | 9,999,164 | {124: 5, 125: 4, 17: 1} |
| uniform-wide / DISK | 0 | {63: 65, 1: 1} | {1: 51, 2: 15} | 2,097,146 | {126: 2, 4: 1} |
| uniform-wide / DISK | 1 | {63: 65, 1: 1} | {16: 47, 15: 17, 14: 1, 1: 1} | 2,096,972 | {126: 2, 4: 1} |
| uniform-wide / HOST | 0 | {63: 65, 1: 1} | {1: 51, 2: 15} | 2,097,146 | {126: 2, 4: 1} |
| uniform-wide / HOST | 1 | {63: 65, 1: 1} | {16: 47, 15: 17, 14: 1, 1: 1} | 2,096,972 | {126: 2, 4: 1} |
| uniform-wide / HOST | collect | {63: 65, 1: 1} | {1: 51, 2: 15} | 2,097,146 | {126: 2, 4: 1} |

## All-operator destination traffic (including raw intermediates)

These are actual committed destination accounting bytes across all operators, not just scan outputs. DISK is actual destination-write traffic; repeated writes remain repeated. Raw physical file sizes can differ from native accounting.

| Input | Tier | Arm | Original bytes | Stored bytes | Raw events | Raw stored bytes | Fallbacks |
|---|---|---|---:|---:|---:|---:|---:|
| narrow-correlated | DISK | 0 | 1,389,213,696 | 298,922,225 | 0 | 0 | 0 |
| narrow-correlated | DISK | 1 | 956,080,128 | 41,532,029 | 0 | 0 | 0 |
| narrow-correlated | HOST | 0 | 1,148,207,104 | 241,259,514 | 1 | 133,120 | 0 |
| narrow-correlated | HOST | 1 | 1,380,186,112 | 60,723,088 | 1 | 133,120 | 0 |
| narrow-correlated | HOST | collect | 1,072,334,848 | 228,091,867 | 1 | 133,120 | 0 |
| narrow-independent | DISK | 0 | 1,587,265,704 | 648,764,012 | 7 | 28,784 | 0 |
| narrow-independent | DISK | 1 | 1,014,786,048 | 381,219,687 | 0 | 0 | 0 |
| narrow-independent | HOST | 0 | 1,096,988,672 | 441,333,732 | 4 | 530,432 | 0 |
| narrow-independent | HOST | 1 | 1,330,128,896 | 507,354,195 | 2 | 266,240 | 0 |
| real-orderkey | DISK | 0 | 1,365,749,520 | 292,425,806 | 0 | 0 | 0 |
| real-orderkey | DISK | 1 | 1,471,285,496 | 323,619,361 | 3 | 12,336 | 0 |
| real-orderkey | HOST | 0 | 1,414,898,464 | 313,902,885 | 4 | 96 | 0 |
| real-orderkey | HOST | 1 | 1,382,763,072 | 295,150,019 | 8 | 192 | 0 |
| real-supplier | DISK | 0 | 1,441,943,520 | 365,312,058 | 7 | 3,129,296 | 0 |
| real-supplier | DISK | 1 | 1,424,747,248 | 353,618,522 | 7 | 857,240 | 0 |
| real-supplier | HOST | 0 | 1,351,080,504 | 336,280,674 | 8 | 5,183,912 | 0 |
| real-supplier | HOST | 1 | 1,393,801,112 | 360,102,983 | 21 | 6,640,968 | 0 |
| uniform-wide | DISK | 0 | 1,512,302,592 | 349,414,423 | 0 | 0 | 0 |
| uniform-wide | DISK | 1 | 964,466,688 | 217,723,014 | 0 | 0 | 0 |
| uniform-wide | HOST | 0 | 1,089,114,232 | 248,044,325 | 8 | 397,432 | 0 |
| uniform-wide | HOST | 1 | 1,447,536,640 | 333,613,008 | 1 | 133,120 | 0 |
| uniform-wide | HOST | collect | 1,130,399,840 | 257,875,610 | 6 | 266,336 | 0 |

## Accounting and limitations

Direct scan raw HOST events were all unrequested compression (`compression_desired=0`, fallback=0), with small tails below the 1MiB threshold; they are included above. All final direct-scan DISK commits were compressed. All-operator raw intermediates are retained separately, including events too small to encode.

Compressed stored bytes include committed payload plus serialized/structural header, not allocator padding. Raw accounting follows the native representation and is not necessarily file-stat physical bytes. DISK rows measure actual successful DISK destination commits (GPU→DISK plus HOST→DISK), never adding intermediate HOST storage. `all_conversions` in JSON/CSV retains downstream traffic and raw/threshold cases separately. Compression-desired and fallback counts are preserved: raw events with no compression request must not be mislabeled failed codec attempts.

Scheduler/metadata overhead is material: treatment takes roughly seconds instead of subsecond baseline queries. Cross-file reordered input creates many interleaved slices/read requests; this is a plausible cost, not an isolated causal profiling result. Collect controls isolate eager collection/aggregate scheduling from reordering while retaining batchsort; their measurements were subsequent rather than randomized/interleaved. No engine optimization was made mid-matrix.

Synthetic positive inputs intentionally repeat overlapping narrow buckets across files. This establishes a mechanism under favorable layout, not a production recommendation. Supplier metadata is nearly full-domain; natural orderkey is already local. Independent payload bounds correlation-dependent gains. Prior scan-sort experiment ratios such as .449→.225 are not paired baselines for these new multifile inputs.

Reproduce CPU analysis: `pixi run python bench/rowgroup-range-cluster/analyze.py`. Raw run logs, membership inventories and full provenance remain in ignored `results-*` directories; compact aggregate JSON/CSV and this report are retained.
