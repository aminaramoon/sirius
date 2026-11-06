# Scan-batch sorting and spill compression experiment

Base: upstream/main `834748f86d53e1fa145956169fbbb21fc8e5a7a1`.

## Question

Does sorting each individual scan output batch by downstream join or ordering keys reduce the bytes stored by the existing Simpatico spill codec? Sorting is local to a batch, not global. Measure compression and runtime separately; neither an end-to-end speedup nor a universal compression gain is assumed.

## Implementation boundaries

- `SIRIUS_EXP_SCAN_BATCH_SORT=1` enables immutable planning-time key metadata; default off.
- Analyze the physical plan before GPU pipeline expansion. Trace direct consumer key references back through only proven projection/filter mappings to a TABLE_SCAN, then copy metadata into its GPU_SCAN replacement. Unsupported expressions or semantic boundaries are skipped.
- Gather every output column using one `cudf::sorted_order` permutation at the common GPU_SCAN execution tail, after all materialization/carrier normalization paths. Use the supplied task stream and memory resource, preserve asynchronous owner lifetimes and existing read-retirement handling, never mutate cached input. Conservatively skip deferred-key/late-materialization and order-sensitive unsupported shapes.
- Keep downstream joins, partitions and global sorts intact. No new global order guarantee.
- Account for gathered output, permutation and scratch in initial memory estimation; retain existing measured-history and OOM retry machinery.

## Measurement

- `SIRIUS_EXP_SCAN_SPILL_METRICS=1` enables successful committed conversion diagnostics under the existing batch lock.
- Existing downgrade byte counts represent source bytes freed, NOT compressed destination size. Precommit codec debug logs can count unsuccessful target attempts. Neither is used as committed compressed bytes.
- Report actual compressed payload plus header consistently: HOST representation size is payload only; DISK representation size is the full file. Record GPU→HOST, GPU→DISK and HOST→DISK separately; the last is additional movement, not newly encoded input.
- Report raw fallback batches, original byte denominators, repeated-spill traffic and actual spill destinations. Zero spills means no pressure experiment measurement.
- Matched-batch hidden C++ experiment uses the real HOST/DISK converters and decode roundtrips on identical deterministic unsorted/sorted row sets. Include correlated-payload, independent-payload and already-ordered controls. This isolates codec effects from scheduler-dependent spill selection.
- End-to-end subprocess runs differ only by sort toggle, validate exact integer results against DuckDB CPU, reject CPU fallback, and capture key admission, sorting, successful spills and elapsed time. Force pressure using bounded GPU/HOST budgets and an explicit disk directory.

## Validation

Baseline upstream build and existing spill tests precede tracked-source edits. Feature validation includes key mapping/unsupported-expression skips, row association, default-off behavior, memory estimates, codec roundtrips and pressure runs. Use `pixi run` and isolate an A100 with `CUDA_VISIBLE_DEVICES=0`.

A multi-model counsel invocation was attempted but could not start because its configured model alias `sol-6.1` is unavailable. A separate read-only safety reviewer is reviewing the conservative implementation plan instead.
