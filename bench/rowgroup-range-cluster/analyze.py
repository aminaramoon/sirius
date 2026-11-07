"""Reproduce final CPU analysis from preserved run JSON and raw logs; no GPU use."""

import collections
import csv
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
COUNTERS = (
    "events",
    "original_bytes",
    "committed_bytes",
    "payload_bytes",
    "header_bytes",
    "compression_desired_events",
    "compression_fallback_events",
)


def add(destination, source):
    for k in COUNTERS:
        destination[k] = destination.get(k, 0) + source.get(k, 0)


def ratios(values):
    values = dict(values)
    values["stored_over_original"] = (
        values.get("committed_bytes", 0) / values["original_bytes"]
        if values.get("original_bytes")
        else None
    )
    return values


def parse(directory, ids):
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in (".txt", ".log"):
            continue
        for line in path.read_text(errors="replace").splitlines():
            if "[scan_sort_spill]" not in line:
                continue
            f = dict(
                re.findall(r"(\w+)=([\w.-]+)", line.split("[scan_sort_spill]", 1)[1])
            )
            if f.get("event") != "conversion_committed" or f["batch_id"] not in ids:
                continue
            direction = (
                f["source_tier"].upper()
                + "->"
                + f["target_tier"].upper()
                + "/"
                + f["representation"]
            )
            v = {
                "events": 1,
                "original_bytes": int(f["gpu_uncompressed_bytes"]),
                "committed_bytes": int(f["serialized_bytes"]),
                "payload_bytes": int(f["payload_bytes"]),
                "header_bytes": int(f["serialized_bytes"]) - int(f["payload_bytes"]),
                "compression_desired_events": int(f["compression_desired"]),
                "compression_fallback_events": int(f["compression_fallback"]),
            }
            add(result.setdefault(direction, {}), v)
    return result


def destination(metrics, tier):
    total, compressed, raw = {}, {}, {}
    for direction, v in metrics.items():
        if direction.split("/")[0].split("->")[1] != tier:
            continue
        add(total, v)
        add(compressed if direction.endswith("/compressed") else raw, v)
    return {
        "total": ratios(total),
        "compressed": ratios(compressed),
        "raw": ratios(raw),
    }


def telemetry(splits):
    spans = [
        int(s["span_max"]) - int(s["span_min"]) + 1
        for s in splits
        if s.get("span_min") != "unknown" and s.get("span_max") != "unknown"
    ]
    return {
        "splits": len(splits),
        "rowgroups_per_split": dict(
            collections.Counter(int(s["rowgroups"]) for s in splits)
        ),
        "files_per_split": dict(collections.Counter(int(s["files"]) for s in splits)),
        "span_min": min(spans) if spans else None,
        "span_median": statistics.median(spans) if spans else None,
        "span_max": max(spans) if spans else None,
        "rows": sum(int(s["rows"]) for s in splits),
        "bytes_sum": sum(int(s["bytes"]) for s in splits),
    }


def main():
    runs = []
    groups = {}
    inventories = {}
    provenances = []
    for directory in sorted(HERE.glob("results-final-*")):
        manifest = json.loads((directory / "data-manifest.json").read_text())
        provenances.append(json.loads((directory / "provenance.json").read_text()))
        dataset = manifest["dataset"]
        tier = "DISK" if directory.name.endswith("-disk") else "HOST"
        fact_rg_count = len(
            json.loads((HERE / ("data-" + dataset) / "schedules.json").read_text())[
                "physical"
            ]
        )
        for record in json.loads((directory / "summary.json").read_text()):
            assert all(a["status"] == "ok" for a in record["membership_audit"])
            inventory = sorted(
                (a["signature"][0], tuple(a["inventory"]))
                for a in record["membership_audit"]
            )
            identity = (dataset, tier)
            if identity in inventories:
                assert (
                    inventory == inventories[identity]
                ), "Cross-directory rowgroup inventory mismatch"
            else:
                inventories[identity] = inventory
            by_operator = collections.defaultdict(list)
            for batch_id, output in record["scan_outputs"].items():
                by_operator[output["operator_id"]].append((batch_id, output))
            fact_operators = [
                op
                for op, entries in by_operator.items()
                if sum(int(o["rows"]) for _, o in entries) == manifest["rows"]
            ]
            if len(fact_operators) != 1:
                raise RuntimeError("Ambiguous fact operator: " + str(directory))
            fact_ids = {b for b, _ in by_operator[fact_operators[0]]}
            all_ids = set(record["scan_outputs"])
            scopes = {
                "all_scan": record["scan_output_metrics"],
                "fact_scan": parse(Path(record["directory"]), fact_ids),
                "dimension_scan": parse(Path(record["directory"]), all_ids - fact_ids),
                "all_conversions": record["all_conversion_metrics"],
            }
            fact_contracts = {
                a["contract_id"]
                for a in record["membership_audit"]
                if int(a["signature"][2]) == fact_rg_count
                and int(a["signature"][1]) == len(manifest["fact_files"])
            }
            assert len(fact_contracts) == 1
            fact_splits = [
                s
                for s in record["cluster_metrics"]["splits"]
                if s["contract_id"] in fact_contracts
            ]
            dim_splits = [
                s
                for s in record["cluster_metrics"]["splits"]
                if s["contract_id"] not in fact_contracts
            ]
            assert sum(int(s["rows"]) for s in fact_splits) == manifest["rows"]
            item = {
                "dataset": dataset,
                "tier": tier,
                "arm": record["arm"],
                "repeat": record["repeat"],
                "status": record["status"],
                "query_seconds": record["result"]["query_seconds"],
                "directory": record["directory"],
                "fact_operator_id": fact_operators[0],
                "fact_split_telemetry": telemetry(fact_splits),
                "dimension_split_telemetry": telemetry(dim_splits),
                "directions_by_scope": scopes,
                "destination_by_scope": {
                    k: destination(v, tier) for k, v in scopes.items()
                },
            }
            runs.append(item)
            key = (dataset, tier, record["arm"])
            group = groups.setdefault(
                key,
                {
                    "dataset": dataset,
                    "tier": tier,
                    "arm": record["arm"],
                    "runs": [],
                    "pooled": {},
                },
            )
            group["runs"].append(item)
            for scope, values in scopes.items():
                dest = destination(values, tier)
                for representation, v in dest.items():
                    add(
                        group["pooled"]
                        .setdefault(scope, {})
                        .setdefault(representation, {}),
                        v,
                    )
    aggregate = []
    for group in groups.values():
        group["median_query_seconds"] = statistics.median(
            r["query_seconds"] for r in group["runs"]
        )
        group["per_repeat_scan_ratios"] = [
            r["destination_by_scope"]["all_scan"]["total"]["stored_over_original"]
            for r in group["runs"]
        ]
        group["pooled"] = {
            scope: {rep: ratios(v) for rep, v in reps.items()}
            for scope, reps in group["pooled"].items()
        }
        aggregate.append(group)
    provenance_sets = {
        field: sorted({p[field] for p in provenances})
        for field in (
            "installed_library_sha256",
            "extension_sha256",
            "working_tree_patch_sha256",
        )
    }
    assert all(
        len(values) == 1 for values in provenance_sets.values()
    ), "Engine changed between final measurements"
    output = {
        "provenance_sha256": provenance_sets,
        "cross_directory_inventories_identical": True,
        "final_gpu_runs": len(runs),
        "statuses": dict(collections.Counter(r["status"] for r in runs)),
        "notes": [
            "Committed conversions only; traffic may repeat batches.",
            "DISK counts destination writes, not intermediate HOST+DISK.",
            "Ratios are byte-weighted pooled ratios.",
            "Fact operator identified by total scan rows; fact contract by prepared footer inventory.",
        ],
        "groups": aggregate,
    }
    (HERE / "aggregate-results.json").write_text(json.dumps(output, indent=2) + "\n")
    with (HERE / "aggregate-results.csv").open("w") as stream:
        columns = [
            "dataset",
            "tier",
            "arm",
            "scope",
            "representation",
            "events",
            "original_bytes",
            "committed_bytes",
            "stored_over_original",
            "compression_desired_events",
            "compression_fallback_events",
            "median_query_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for g in aggregate:
            for scope, reps in g["pooled"].items():
                for representation, values in reps.items():
                    writer.writerow(
                        {
                            **{
                                k: g[k]
                                for k in (
                                    "dataset",
                                    "tier",
                                    "arm",
                                    "median_query_seconds",
                                )
                            },
                            "scope": scope,
                            "representation": representation,
                            **{k: values.get(k) for k in columns if k in values},
                        }
                    )
    report = [
        "# Results: row-group range clustering plus per-batch sorting",
        "",
        "All arms read identical checksummed physical inputs and sort every admitted scan batch. Arm 0 is batchsort-only; arm 1 additionally groups similar row-group ranges across files. `collect` performs eager metadata/aggregate scheduling without range reordering. This is not a comparison with the older experiment’s different input layout.",
        "",
        "**Finding:** narrow overlapping-range synthetic inputs compress substantially better with clustering, but treatment is much slower. Uniform-wide and natural-order controls do not support a universal compression improvement. Three repeats are descriptive, not statistical significance evidence.",
        "",
        "See [README.md](README.md) for reproduction and [DESIGN.md](DESIGN.md) for the controls and staged design. Branch: `experiment/rowgroup-range-cluster`, based on prior feature commit [`f7c38a4c`](https://github.com/aminaramoon/sirius/commit/f7c38a4c) pushed to the personal fork. The binary dirty-source patch in every final output captures the subsequent uncommitted engine implementation.",
        "",
        "Synthetic inputs each contain 33,554,432 two-BIGINT rows in 16 files / 4,096 rowgroups of 8,192 rows. Real supplier/orderkey controls each preserve the existing 32,000,000-row projection in one rechunked file / 3,903 rowgroups (median 8,192 rows). Payload is `l_extendedprice` rounded by DuckDB’s BIGINT cast: these are bounded projected controls, not full TPC-H tables or official TPC-H queries.",
        "",
        "## Validation and setup",
        "",
        "Prerequisite engine tests: new range clustering 599 assertions / 14 cases; existing scan sorting 290/14, spilling 67/5, and Parquet sizing 951/20, totaling 1,907 assertions / 53 cases. Six Python harness unit tests also pass.",
        "",
        "77 successful GPU query processes: tiny exact-full-row validation (3), pressure pilots (8), final primary matrix (60), and HOST collect controls (6). No failed/inconclusive runs or timeouts. Tiny 32,768-row four-file join matches the exact CPU tuple multiset in 0/collect/1. Every final result matches its CPU reference with fallback disabled; row-group contract inventories match across arms/repeats, and input hashes remain unchanged.",
        "",
        "Source/build frozen throughout. One A100; GPU cap 1GiB, scan cap 8MiB, hash cap 32MiB; shared original HOST/DISK pressure configs, thresholds .25/.15. HOST has 8GiB/node; DISK has 128MiB HOST/node and isolated 16GiB disk. No pressure tuning was necessary: all DISK runs committed compressed direct scan outputs. Fresh subprocess per arm/repeat, alternating main A/B order, no hidden warmups. Collect controls were measured afterward, not interleaved with A/B.",
        "",
        "The experimental engine implementation is in `src/planner/sirius_physical_plan_generator.cpp`, `src/op/scan/parquet_scan_core.{cpp,hpp}`, and associated scan descriptor/provider plumbing. `SIRIUS_EXP_ROWGROUP_RANGE_CLUSTER=1` enables range reordering; `collect` collects metadata with original order; 0 retains baseline scheduling. All arms set `SIRIUS_EXP_SCAN_BATCH_SORT=1`, `SIRIUS_EXP_ROWGROUP_RANGE_AUDIT=1` (observer-only in baseline), and `SIRIUS_EXP_SCAN_SPILL_METRICS=1`. Eligibility is conservative: plain Parquet and first admitted signed INT32/INT64 key. Missing/unsupported key statistics cause whole-aggregate original-order fallback, never dropping rowgroups.",
        "",
        "Provenance records extension and installed dynamic library SHA256 plus binary dirty-working-tree patch (HEAD alone is the previous feature), hashes of harnesses, effective configs, commands, and environment. Installed library SHA256: `5a4f9f3eb4b68a63378aeb32e202d97801412b21f6ceb2160eba11aaa8ec8e43`. Tiny/pilot library+patch provenance was captured immediately afterward and marked post-run; final provenance was captured before execution.",
        "",
        "## Primary: all direct scan outputs, pooled across three repeats",
        "",
        "Ratios below include **raw and compressed** successful destination commits. Show compressed-only ratios separately; raw tails are not dropped. Repeated spills count repeated traffic, not unique dataset size. Original-byte denominators vary because asynchronous pressure selects different batches. Never sum HOST and DISK directions as unique input.",
        "",
        "| Input | Destination | Arm | Events | Original bytes | Stored bytes | All stored/original | Compressed-only | Raw events | Fallbacks | Median query s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in aggregate:
        a = g["pooled"]["all_scan"]
        t = a["total"]
        c = a["compressed"]
        r = a["raw"]
        report.append(
            f"| {g['dataset']} | {g['tier']} | {g['arm']} | {t.get('events',0)} | {t.get('original_bytes',0):,} | {t.get('committed_bytes',0):,} | {t['stored_over_original']:.6f} | {c['stored_over_original']:.6f} | {r.get('events',0)} | {t.get('compression_fallback_events',0)} | {g['median_query_seconds']:.4f} |"
        )
    report += [
        "",
        "## Paired extra compression and latency",
        "",
        "| Input | Tier | Extra reduction: 1 - ratio(1)/ratio(0) | Runtime 1 / 0 |",
        "|---|---|---:|---:|",
    ]
    for dataset in sorted({g["dataset"] for g in aggregate}):
        for tier in ("HOST", "DISK"):
            pair = {
                g["arm"]: g
                for g in aggregate
                if g["dataset"] == dataset
                and g["tier"] == tier
                and g["arm"] in ("0", "1")
            }
            a, b = pair["0"], pair["1"]
            reduction = (
                1
                - b["pooled"]["all_scan"]["total"]["stored_over_original"]
                / a["pooled"]["all_scan"]["total"]["stored_over_original"]
            )
            report.append(
                f"| {dataset} | {tier} | {100*reduction:.3f}% | {b['median_query_seconds']/a['median_query_seconds']:.2f}x |"
            )
    report += [
        "",
        "Collect HOST controls retain baseline-like compression and subsecond medians, unlike reordered treatment. This points to range reordering/interleaved reads rather than eager metadata collection alone as the dominant difference, without claiming isolated causal profiling. Correlated narrow data shows ~80% extra storage reduction; independent payload shows only ~6%, emphasizing payload-correlation dependence.",
        "",
        "## Fact-only scan outputs",
        "",
        "Dimensions also scan/spill; the following isolates fact scan batches using operator provenance. The aggregate JSON additionally retains dimension-only and all-operator metrics, every conversion direction, per-repeat ratios and event denominators.",
        "",
        "| Input | Tier | Arm | Fact events | Fact original | Fact stored | Fact all ratio |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for g in aggregate:
        t = g["pooled"]["fact_scan"]["total"]
        report.append(
            f"| {g['dataset']} | {g['tier']} | {g['arm']} | {t.get('events',0)} | {t.get('original_bytes',0):,} | {t.get('committed_bytes',0):,} | {t['stored_over_original']:.6f} |"
        )
    report += [
        "",
        "## Actual split geometry (fact and dimension separate)",
        "",
        "Representative first repeat per case; complete repeat telemetry is in JSON. These are actual packed split logs, not the earlier 64-RG/8MiB metadata proxy. Auxiliary preparation accounting reduces synthetic fact splits to 63 RGs. Baseline already coalesces a few adjacent files; treatment groups all 16.",
        "",
        "| Input / tier | Arm | Fact RG/split: count | Fact files/split: count | Median fact key span | Dimension RG/split: count |",
        "|---|---|---|---|---:|---|",
    ]
    for g in aggregate:
        r = g["runs"][0]
        f = r["fact_split_telemetry"]
        d = r["dimension_split_telemetry"]
        report.append(
            f"| {g['dataset']} / {g['tier']} | {g['arm']} | {f['rowgroups_per_split']} | {f['files_per_split']} | {f['span_median']:,.0f} | {d['rowgroups_per_split']} |"
        )
    report += [
        "",
        "## All-operator destination traffic (including raw intermediates)",
        "",
        "These are actual committed destination accounting bytes across all operators, not just scan outputs. DISK is actual destination-write traffic; repeated writes remain repeated. Raw physical file sizes can differ from native accounting.",
        "",
        "| Input | Tier | Arm | Original bytes | Stored bytes | Raw events | Raw stored bytes | Fallbacks |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for g in aggregate:
        a = g["pooled"]["all_conversions"]
        t = a["total"]
        r = a["raw"]
        report.append(
            f"| {g['dataset']} | {g['tier']} | {g['arm']} | {t.get('original_bytes',0):,} | {t.get('committed_bytes',0):,} | {r.get('events',0)} | {r.get('committed_bytes',0):,} | {t.get('compression_fallback_events',0)} |"
        )
    report += [
        "",
        "## Accounting and limitations",
        "",
        "Direct scan raw HOST events were all unrequested compression (`compression_desired=0`, fallback=0), with small tails below the 1MiB threshold; they are included above. All final direct-scan DISK commits were compressed. All-operator raw intermediates are retained separately, including events too small to encode.",
        "",
        "Compressed stored bytes include committed payload plus serialized/structural header, not allocator padding. Raw accounting follows the native representation and is not necessarily file-stat physical bytes. DISK rows measure actual successful DISK destination commits (GPU→DISK plus HOST→DISK), never adding intermediate HOST storage. `all_conversions` in JSON/CSV retains downstream traffic and raw/threshold cases separately. Compression-desired and fallback counts are preserved: raw events with no compression request must not be mislabeled failed codec attempts.",
        "",
        "Scheduler/metadata overhead is material: treatment takes roughly seconds instead of subsecond baseline queries. Cross-file reordered input creates many interleaved slices/read requests; this is a plausible cost, not an isolated causal profiling result. Collect controls isolate eager collection/aggregate scheduling from reordering while retaining batchsort; their measurements were subsequent rather than randomized/interleaved. No engine optimization was made mid-matrix.",
        "",
        "Synthetic positive inputs intentionally repeat overlapping narrow buckets across files. This establishes a mechanism under favorable layout, not a production recommendation. Supplier metadata is nearly full-domain; natural orderkey is already local. Independent payload bounds correlation-dependent gains. Prior scan-sort experiment ratios such as .449→.225 are not paired baselines for these new multifile inputs.",
        "",
        "Reproduce CPU analysis: `pixi run python bench/rowgroup-range-cluster/analyze.py`. Raw run logs, membership inventories and full provenance remain in ignored `results-*` directories; compact aggregate JSON/CSV and this report are retained.",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(report) + "\n")
    print("Analyzed", len(runs), "final runs", output["statuses"])
    for g in aggregate:
        print(
            g["dataset"],
            g["tier"],
            g["arm"],
            round(g["pooled"]["all_scan"]["total"]["stored_over_original"], 6),
            round(g["median_query_seconds"], 4),
        )


if __name__ == "__main__":
    main()
