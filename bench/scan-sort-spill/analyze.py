#!/usr/bin/env python3
"""Pool committed byte counters (never average batch ratios) across final repeats."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
COUNTERS = (
    "events",
    "original_bytes",
    "payload_bytes",
    "header_bytes",
    "committed_bytes",
    "native_source_bytes",
    "compression_desired_events",
    "compression_fallback_events",
)


def pool(items, metric_key, predicate=lambda key: True):
    buckets = {}
    for item in items:
        for key, values in item[metric_key].items():
            if not predicate(key):
                continue
            bucket = buckets.setdefault(key, {name: 0 for name in COUNTERS})
            for name in COUNTERS:
                bucket[name] += values[name]
    for bucket in buckets.values():
        bucket["committed_over_original"] = (
            bucket["committed_bytes"] / bucket["original_bytes"]
            if bucket["original_bytes"]
            else None
        )
    return buckets


def by_destination(metrics):
    result = {}
    for direction, values in metrics.items():
        if not direction.endswith("/compressed"):
            continue
        destination = direction.split("->", 1)[1].split("/", 1)[0]
        bucket = result.setdefault(destination, {name: 0 for name in COUNTERS})
        for name in COUNTERS:
            bucket[name] += values[name]
    for bucket in result.values():
        bucket["committed_over_original"] = (
            bucket["committed_bytes"] / bucket["original_bytes"]
            if bucket["original_bytes"]
            else None
        )
    return result


def aggregate(root):
    groups = defaultdict(list)
    failures = []
    all_items = []
    for path in sorted(root.glob("results-final-*/summary.json")):
        label = path.parent.name.removeprefix("results-final-")
        for item in json.loads(path.read_text()):
            all_items.append(item)
            groups[(label, item["query"], item["sorted"])].append(item)
            if item["status"] != "ok":
                log = Path(item["directory"]) / "process.log"
                failures.append(
                    {
                        "matrix": label,
                        "query": item["query"],
                        "repeat": item["repeat"],
                        "sorted": item["sorted"],
                        "status": item["status"],
                        "directory": item["directory"],
                        "error_tail": log.read_text()[-2000:] if log.exists() else None,
                    }
                )
    summaries = []
    for (label, query, arm), items in sorted(groups.items()):
        successful = [item for item in items if item["status"] == "ok"]
        times = [item["result"]["query_seconds"] for item in successful]
        summaries.append(
            {
                "matrix": label,
                "query": query,
                "sorted": arm,
                "attempted_repeats": len(items),
                "successful_repeats": len(successful),
                "statuses": dict(Counter(item["status"] for item in items)),
                "median_query_seconds": statistics.median(times) if times else None,
                "query_seconds": times,
                "sorted_batches": sum(
                    item["sort_metrics"]["sorted_batches"] for item in successful
                ),
                "sorted_rows": sum(
                    item["sort_metrics"]["sorted_rows"] for item in successful
                ),
                "scan_output_metrics": pool(successful, "scan_output_metrics"),
                "scan_output_compressed_destinations": by_destination(
                    pool(successful, "scan_output_metrics")
                ),
                "gpu_encoding_metrics": pool(
                    successful, "metrics", lambda key: key.startswith("GPU->")
                ),
                "host_to_disk_traffic_metrics": pool(
                    successful, "metrics", lambda key: key.startswith("HOST->DISK/")
                ),
                "all_conversion_metrics": pool(successful, "metrics"),
                "runs": [item["directory"] for item in items],
            }
        )
    return {
        "run_counts": dict(Counter(item["status"] for item in all_items)),
        "total_runs": len(all_items),
        "groups": summaries,
        "failures": failures,
        "matched_codec_records": json.loads((root / "codec-results.json").read_text())[
            "records"
        ],
        "caveats": [
            "Only status=ok runs contribute bytes/times; failed/inconclusive attempts remain listed.",
            "Ratios are sum(committed destination bytes)/sum(decoded original bytes), not mean ratios.",
            "scan_output_metrics is direct scan batch provenance; empty means no direct scan spill evidence.",
            "gpu_encoding_metrics includes all GPU-origin outputs, including downstream already-sorted batches.",
            "HOST->DISK is another destination write, not another newly encoded input; scopes overlap, never sum scopes.",
            "Compression comparisons use /compressed keys; /uncompressed keys separately expose fallback or size-gate traffic.",
            "Async pressure selects different original byte totals; compare original bytes/events alongside ratios.",
            "All sort runs lack direct scan-output spills; do not infer scan compression benefit from downstream sort spill ratios.",
            "Orderkey control sort rejected both arms (ordered UNION unsupported); DISK orderkey intentionally join-only.",
            "Sorting disables converted-table steal on opted-in scans for retry safety; timing includes this overhead.",
            "Pristine smoke and pilot directories are excluded from this final aggregate.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE / "aggregate.json")
    args = parser.parse_args()
    report = aggregate(args.root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["run_counts"], indent=2))
    for item in report["groups"]:
        compressed = {
            key: round(value["committed_over_original"], 6)
            for key, value in item["scan_output_metrics"].items()
            if key.endswith("/compressed")
        }
        print(
            item["matrix"],
            item["query"],
            item["sorted"],
            item["successful_repeats"],
            item["median_query_seconds"],
            compressed,
        )


if __name__ == "__main__":
    main()
