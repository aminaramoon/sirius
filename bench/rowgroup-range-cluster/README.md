# Row-group range clustering experiment

**Completed: 77 successful GPU runs, including 66 final measurements.** See [RESULTS.md](RESULTS.md) for outcomes and [DESIGN.md](DESIGN.md) for the planned controls. Narrow correlated inputs achieved ~80% extra compression reduction but substantially slower queries; negative/locality controls showed little or no gain. Old `bench/scan-sort-spill/` is untouched.

## Prepared inputs

All generated data live in ignored `data-*` directories (~2.83GB total including dimensions). Every paired arm reads the same exact files, verified against SHA256 manifests.

| Dataset | Rows | Files / RG | Median simulated 8MiB batch key span: physical → clustered |
|---|---:|---:|---:|
| narrow-correlated | 33,554,432 | 16 / 4096 | 2,097,142 → 32,768 |
| narrow-independent | 33,554,432 | 16 / 4096 | 2,097,142 → 32,768 |
| uniform-wide | 33,554,432 | 16 / 4096 | 2,097,145 → 2,096,976 |
| real-supplier | 32,000,000 | 1 / 3903 | 9,999,969 → 9,999,176 |
| real-orderkey | 32,000,000 | 1 / 3903 | 523,587 → 523,587 |

These are metadata scheduling proxies, **not measured GPU batching or compression**. Synthetic narrow clustered simulated batches contain RGs from all 16 files; their span shrinks from ~100% to 1.5625% of domain. Each of 128 overlapping narrow buckets occurs in 32 RGs. Independent-payload variant changes only payload entropy. Uniform-wide is a negative control; supplier is similarly near full-domain. Natural orderkey already has locality. Real rechunking preserves complete source projected rows (`EXCEPT ALL` both directions = zero).

CPU references and all hashes passed in five fresh CPU subprocesses, no extension loaded. Six harness unit tests passed. Full original SF1000 footer survey (120 files, 5640 RG) is summarized in `metadata-summary.json`; small per-dataset summary JSONs retain generation SQL, provenance and scheduling statistics. Raw footers and full membership schedules are generated artifacts.

## CPU-only reproduction

Use fresh data directories / remove only your own generated outputs before recreating inputs. Preparation deliberately refuses overwrite.

```bash
pixi run python bench/rowgroup-range-cluster/analyze_metadata.py
pixi run python bench/rowgroup-range-cluster/prepare_controls.py \
  --datasets narrow-correlated uniform-wide narrow-independent real-supplier real-orderkey
pixi run python -m unittest discover -s bench/rowgroup-range-cluster -p 'test_*.py' -v
pixi run python bench/rowgroup-range-cluster/experiment.py cpu-check \
  --data bench/rowgroup-range-cluster/data-narrow-correlated \
  --output bench/rowgroup-range-cluster/results-new-cpu-check
```

`prepare_controls.py` confines DuckDB temporary storage to its own ignored output directory. `cpu-check` is the harness default action and never loads Sirius.

## Tiny full-row correctness fixture

`prepare_tiny.py` creates a separate ignored four-file fixture: 32,768 rows, sixteen 2,048-row RGs, repeated keys with independently varying associated payloads. CPU full-row fetching and aggregate/hash checks passed. The final build and regression tests passed before GPU execution. All three tiny GPU arms passed exact tuple-multiset checks before pressure pilots. Reproduction commands (use fresh output paths; fixture preparation refuses overwrite):

```bash
pixi run python bench/rowgroup-range-cluster/prepare_tiny.py
pixi run python bench/rowgroup-range-cluster/experiment.py correctness \
  --data bench/rowgroup-range-cluster/data-tiny \
  --output bench/rowgroup-range-cluster/results-tiny-gpu-correctness \
  --arms 0 collect 1 --repeats 1
```

The correctness action fetches every `(f.k,f.payload)` from the actual GPU join and compares the exact sorted tuple multiset, including duplicates, against the CPU result. It disables fallback, checks all input SHA256s before/after and audits rowgroup membership. It does not require spilling or interpret tiny-query compression. Full-row mode refuses inputs above 100,000 rows. A CPU-only rehearsal is `experiment.py cpu-check --full-rows --data .../data-tiny --output FRESH_PATH`.

## GPU reproduction — only after feature build/validation

Example pilot command (the recorded pilot used this configuration; choose a fresh output directory):

```bash
pixi run python bench/rowgroup-range-cluster/experiment.py run \
  --data bench/rowgroup-range-cluster/data-narrow-correlated \
  --config bench/scan-sort-spill/host-pressure.yaml \
  --output bench/rowgroup-range-cluster/results-pilot-narrow-host \
  --repeats 1 --arms 0 1 --timeout 90
```

Arms `0` and `1` set `SIRIUS_EXP_ROWGROUP_RANGE_CLUSTER=0/1`; both always set `SIRIUS_EXP_SCAN_BATCH_SORT=1` and metrics ON. Optional `--arms 0 collect 1` adds eager original-order scheduling with no range reordering. Same config is used in all arms; no GPU allocations occur during preparation. DISK uses the existing `disk-pressure.yaml`, replacing only its spill-directory placeholder in each worker's new effective config. GPU execution is serialized, `CUDA_VISIBLE_DEVICES=0`, each query/arm/repeat a fresh process; fallback disabled, exact CPU result checks, query-call time and subprocess wall time retained.

`summary.json` records raw cluster plans/split membership, actual sort batches and committed spill metrics, separated into `scan_output_metrics` and `all_conversion_metrics`. No direct scan compressed commits, absent actual sorting, no eligible/reordered cross-file narrow treatment, or missing per-contract membership evidence produces an explicit inconclusive status. Duplicate/missing rowgroups within a logged contract fail the run. Inconclusive statuses do not claim improvement and, like the old harness, may exit zero; inspect every status. All arms set `SIRIUS_EXP_ROWGROUP_RANGE_AUDIT=1`. Baseline gate0 preserves streaming scheduling and emits observer-only split membership, without a cluster inventory plan. The harness infers baseline contract geometry from those splits and compares complete per-run contract inventories across 0/collect/1 without assuming contract IDs are stable across runs. The new engine markers include contract IDs. CPU checks alone do not validate GPU feature semantics. The completed experiment additionally passed exact full-row tiny GPU checks, full row-group inventory comparisons, CPU aggregate references, and prerequisite engine regressions. See RESULTS.md for scope and limitations.
