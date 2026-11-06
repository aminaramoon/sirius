#!/usr/bin/env python3
"""Bounded paired compressed-spill experiment. Run via pixi run python."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
QUERIES = {
    "join": "SELECT count(*), sum(f.payload) FROM fact f JOIN dim d ON f.k=d.k",
    "sort": "SELECT sum(payload), count(*) FROM (SELECT k,payload FROM fact ORDER BY k,payload)",
}
MARKER = "[scan_sort_spill]"


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier(value):
    return '"' + value.replace('"', '""') + '"'


def connect(data):
    import duckdb

    con = duckdb.connect(config={"allow_unsigned_extensions": "true", "threads": "4"})
    con.execute("SET memory_limit='4GiB'")
    con.execute("SET preserve_insertion_order=true")
    for table in ("fact", "dim"):
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet({literal(data / (table + '.parquet'))})"
        )
    return con


def worker(args):
    if args.worker == "prepare":
        import duckdb

        args.data.mkdir(parents=True, exist_ok=True)
        if any(
            (args.data / name).exists()
            for name in ("fact.parquet", "dim.parquet", "manifest.json")
        ):
            raise RuntimeError(
                "Refusing to overwrite existing data; choose a fresh --data directory"
            )
        con = duckdb.connect(config={"threads": "4"})
        con.execute("SET memory_limit='4GiB'")
        con.execute("SET preserve_insertion_order=true")
        # No global shuffle workspace: a versioned deterministic hash yields repeated,
        # shuffled keys in scan order, while payload remains strongly key-correlated.
        fact_sql = f"""SELECT k::BIGINT AS k,
            (k * 1000003 + (hash(i + {args.seed}) % 8)::BIGINT)::BIGINT AS payload
            FROM (SELECT i, (hash(i + {args.seed}) % {args.keys})::BIGINT AS k
                  FROM range({args.rows}) t(i))"""
        if args.source_parquet:
            fact_sql = (
                f"SELECT {identifier(args.source_key)}::BIGINT AS k, "
                f"{identifier(args.source_payload)}::BIGINT AS payload "
                f"FROM read_parquet({literal(args.source_parquet)}) LIMIT {args.rows}"
            )
        dim_sql = (
            f"SELECT DISTINCT k FROM read_parquet({literal(args.data / 'fact.parquet')})"
            if args.source_parquet
            else f"SELECT i::BIGINT AS k FROM range({args.keys}) t(i)"
        )
        for table, sql in (("fact", fact_sql), ("dim", dim_sql)):
            con.execute(
                f"COPY ({sql}) TO {literal(args.data / (table + '.parquet'))} (FORMAT PARQUET, COMPRESSION UNCOMPRESSED, ROW_GROUP_SIZE 131072)"
            )
        save(
            args.data / "manifest.json",
            {
                "rows": args.rows,
                "keys": args.keys,
                "seed": args.seed,
                "duckdb_version": duckdb.__version__,
                "fact_sql": fact_sql,
                "source_parquet": (
                    str(args.source_parquet) if args.source_parquet else None
                ),
                "actual_rows": con.execute(
                    f"SELECT count(*) FROM read_parquet({literal(args.data / 'fact.parquet')})"
                ).fetchone()[0],
                "files": {
                    t: {
                        "bytes": (args.data / (t + ".parquet")).stat().st_size,
                        "sha256": sha256_file(args.data / (t + ".parquet")),
                    }
                    for t in ("fact", "dim")
                },
            },
        )
        return
    con = connect(args.data)
    query = QUERIES[args.query]
    # Save the optimized DuckDB plan and require the physical ORDER_BY to survive.
    plan = "\n".join(str(row[-1]) for row in con.execute("EXPLAIN " + query).fetchall())
    Path("duckdb-plan.txt").write_text(plan + "\n")
    if args.query == "sort" and "ORDERBY" not in re.sub(r"[^A-Z]", "", plan.upper()):
        raise RuntimeError("ORDER_BY optimized away: invalid sort experiment")
    if args.worker == "gpu":
        con.execute(f"LOAD {literal(args.extension)}")
        con.execute("SET gpu_execution=false")
        con.execute("SET enable_duckdb_fallback=false")
        con.execute("SET sirius_log_level='debug'")
        # Explicit CALL avoids transparent-interception uncertainty; fallback is
        # independently disabled since CALL alone can otherwise replay on CPU.
        statement = "CALL gpu_execution(" + literal(query) + ")"
    else:
        statement = query  # CPU worker never loads the extension.
    start = time.perf_counter()
    rows = con.execute(statement).fetchall()
    elapsed = time.perf_counter() - start
    save(
        args.result,
        {
            "rows": [[str(v) for v in row] for row in rows],
            "query_seconds": elapsed,
            "query": query,
            "execution": args.worker,
            "fallback_disabled": args.worker == "gpu",
        },
    )
    con.close()


def launch(args, directory, kind, query=None, sorted_arm=0):
    directory.mkdir(parents=True, exist_ok=False)
    result = directory / "result.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        kind,
        "--data",
        str(args.data),
        "--extension",
        str(args.extension),
        "--result",
        str(result),
        "--rows",
        str(args.rows),
        "--keys",
        str(args.keys),
        "--seed",
        str(args.seed),
    ]
    if args.source_parquet:
        command += [
            "--source-parquet",
            str(args.source_parquet),
            "--source-key",
            args.source_key,
            "--source-payload",
            args.source_payload,
        ]
    if query:
        command += ["--query", query]
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        SIRIUS_EXP_SCAN_BATCH_SORT=str(sorted_arm),
        SIRIUS_EXP_SCAN_SPILL_METRICS="1",
        SIRIUS_CONFIG_FILE=str(args.config),
        SIRIUS_LOG_BACKEND="spdlog",
        SIRIUS_LOG_LEVEL="debug",
        SIRIUS_LOG_DIR=str(directory / "log"),
    )
    env.pop("SIRIUS_DISABLE", None)
    if kind != "gpu":
        env["SIRIUS_DISABLE"] = "1"
    # Confine spill files to this run, never a shared global directory.
    config = args.config.read_text().replace("@SPILL_DIR@", str(directory / "spill"))
    (directory / "spill").mkdir()
    (directory / "effective.yaml").write_text(config)
    env["SIRIUS_CONFIG_FILE"] = str(directory / "effective.yaml")
    save(
        directory / "invocation.json",
        {
            "command": command,
            "environment": {
                k: v for k, v in env.items() if k.startswith(("SIRIUS_", "CUDA_"))
            },
            "timeout": args.timeout,
        },
    )
    started = time.perf_counter()
    timed_out = False
    with (directory / "process.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    return {
        "returncode": code,
        "timeout": timed_out,
        "wall_seconds": time.perf_counter() - started,
        "result": json.loads(result.read_text()) if result.exists() else None,
    }


def parse_sort_metrics(text):
    plans = []
    batches = []
    for line in text.splitlines():
        if "[scan_batch_sort_plan]" in line:
            plans.append(dict(re.findall(r"(\w+)=([\w.-]+)", line)))
        elif "[scan_batch_sort]" in line:
            batches.append(dict(re.findall(r"(\w+)=([\w.-]+)", line)))
    return {
        "plans": plans,
        "sorted_batches": len(batches),
        "sorted_rows": sum(int(item["rows"]) for item in batches),
        "batches": batches,
    }


def parse_scan_outputs(text):
    outputs = {}
    for line in text.splitlines():
        if "[scan_batch_output]" in line:
            fields = dict(re.findall(r"(\w+)=([\w.-]+)", line))
            outputs[fields["batch_id"]] = fields
    return outputs


def parse_metrics(directory, batch_ids=None):
    totals = {}
    # Native log sinks may write files rather than stdout. Each line is read
    # from its actual file exactly once; do not concatenate copied log bundles.
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in (".log", ".txt"):
            continue
        with path.open(errors="replace") as stream:
            for line in stream:
                if MARKER not in line:
                    continue
                fields = dict(re.findall(r"(\w+)=([\w.-]+)", line.split(MARKER, 1)[1]))
                if fields.get("event") != "conversion_committed":
                    continue
                if batch_ids is not None and fields["batch_id"] not in batch_ids:
                    continue
                destination = fields["target_tier"].upper()
                representation = fields["representation"]
                # Keep source as a separate axis: HOST->DISK is a second write,
                # not another newly encoded GPU spill.
                key = f"{fields['source_tier'].upper()}->{destination}/{representation}"
                bucket = totals.setdefault(
                    key,
                    {
                        "events": 0,
                        "original_bytes": 0,
                        "payload_bytes": 0,
                        "header_bytes": 0,
                        "committed_bytes": 0,
                        "compression_desired_events": 0,
                        "compression_fallback_events": 0,
                        "native_source_bytes": 0,
                    },
                )
                bucket["events"] += 1
                bucket["native_source_bytes"] += int(fields["source_bytes"])
                bucket["compression_desired_events"] += int(
                    fields["compression_desired"]
                )
                bucket["compression_fallback_events"] += int(
                    fields["compression_fallback"]
                )
                bucket["original_bytes"] += int(fields["gpu_uncompressed_bytes"])
                bucket["payload_bytes"] += int(fields["payload_bytes"])
                bucket["header_bytes"] += int(fields["serialized_bytes"]) - int(
                    fields["payload_bytes"]
                )
                bucket["committed_bytes"] += int(fields["serialized_bytes"])
    for bucket in totals.values():
        bucket["committed_over_original"] = (
            bucket["committed_bytes"] / bucket["original_bytes"]
            if bucket["original_bytes"]
            else None
        )
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "run", "summarize"), nargs="?", default="run"
    )
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--config", type=Path, default=HERE / "host.yaml")
    parser.add_argument(
        "--extension",
        type=Path,
        default=REPO
        / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension",
    )
    parser.add_argument(
        "--source-parquet",
        type=Path,
        help="optional one-file bounded real-data projection",
    )
    parser.add_argument("--source-key", default="l_suppkey")
    parser.add_argument(
        "--source-payload",
        default="l_extendedprice",
        help="cast to BIGINT when preparing real data",
    )
    parser.add_argument(
        "--require-tier",
        choices=("HOST", "DISK"),
        help="default: DISK if config enables disk, otherwise HOST",
    )
    parser.add_argument("--rows", type=int, default=32_000_000)
    parser.add_argument("--keys", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--queries", nargs="+", choices=tuple(QUERIES), default=list(QUERIES)
    )
    parser.add_argument(
        "--worker", choices=("prepare", "cpu", "gpu"), help=argparse.SUPPRESS
    )
    parser.add_argument("--query", choices=tuple(QUERIES), help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for attr in ("data", "output", "config", "extension"):
        setattr(args, attr, getattr(args, attr).resolve())
    if args.source_parquet:
        args.source_parquet = args.source_parquet.resolve()
    args.require_tier = args.require_tier or (
        "DISK" if "downgrade_root_dirs:" in args.config.read_text() else "HOST"
    )
    if min(args.rows, args.keys, args.repeats, args.timeout) <= 0:
        parser.error("rows, keys, repeats and timeout must be positive")
    if args.worker:
        worker(args)
        return
    if args.action == "summarize":
        print(
            json.dumps(json.loads((args.output / "summary.json").read_text()), indent=2)
        )
        return
    if args.action == "prepare":
        info = launch(args, args.output / "prepare", "prepare")
        save(args.output / "prepare-status.json", info)
        if info["returncode"]:
            raise SystemExit("Preparation failed; see prepare/process.log")
        return
    if not args.extension.is_file() or not (args.data / "manifest.json").is_file():
        parser.error("built --extension and prepared --data/manifest.json required")
    args.output.mkdir(parents=True, exist_ok=True)
    save(
        args.output / "data-manifest.json",
        json.loads((args.data / "manifest.json").read_text()),
    )
    save(
        args.output / "provenance.json",
        {
            "extension": str(args.extension),
            "extension_mtime_ns": args.extension.stat().st_mtime_ns,
            "extension_sha256": sha256_file(args.extension),
            "config": str(args.config),
            "python": sys.version,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
        },
    )
    summaries = []
    failed = False
    for query in args.queries:
        cpu = launch(args, args.output / f"cpu-{query}", "cpu", query)
        save(args.output / f"cpu-{query}" / "status.json", cpu)
        if cpu["returncode"] or not cpu["result"]:
            raise SystemExit(f"CPU reference failed: cpu-{query}/process.log")
        for repeat in range(args.repeats):
            # Alternate arm order to reduce cache/thermal bias across pairs.
            for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                directory = args.output / f"{query}-r{repeat + 1}-sort{arm}"
                info = launch(args, directory, "gpu", query, arm)
                metrics = parse_metrics(directory)
                native_logs = "\n".join(
                    p.read_text(errors="replace")
                    for p in (directory / "log").glob("*.log")
                )
                sort_metrics = parse_sort_metrics(native_logs)
                scan_outputs = parse_scan_outputs(native_logs)
                scan_output_metrics = parse_metrics(
                    directory, batch_ids=set(scan_outputs)
                )
                gpu_sort_present = bool(
                    re.search(
                        r"Creating sirius physical plan for logical operator type: (?:ORDER_BY|Order By)",
                        native_logs,
                    )
                )
                valid = bool(
                    info["returncode"] == 0
                    and info["result"]
                    and info["result"]["rows"] == cpu["result"]["rows"]
                    and (query != "sort" or gpu_sort_present)
                )
                compressed = any(
                    k.endswith(f"->{args.require_tier}/compressed") and v["events"]
                    for k, v in metrics.items()
                )
                status = (
                    "ok"
                    if valid and compressed
                    else (
                        f"inconclusive_no_compressed_{args.require_tier.lower()}_spill"
                        if valid
                        else "failed"
                    )
                )
                if valid and arm and not sort_metrics["sorted_batches"]:
                    status = "inconclusive_sort_arm_did_not_sort"
                if valid and not arm and sort_metrics["sorted_batches"]:
                    status = "failed_baseline_unexpectedly_sorted"
                    valid = False
                record = {
                    "query": query,
                    "repeat": repeat + 1,
                    "sorted": arm,
                    "directory": str(directory),
                    "correct": valid,
                    "gpu_sort_present": gpu_sort_present if query == "sort" else None,
                    "status": status,
                    "sort_metrics": sort_metrics,
                    "scan_outputs": scan_outputs,
                    "scan_output_metrics": scan_output_metrics,
                    "required_compressed_tier": args.require_tier,
                    "metrics": metrics,
                    **info,
                }
                save(directory / "status.json", record)
                summaries.append(record)
                save(args.output / "summary.json", summaries)
                failed |= not valid
    with (args.output / "summary.csv").open("w", newline="") as stream:
        columns = [
            "query",
            "repeat",
            "sorted",
            "status",
            "scope",
            "query_seconds",
            "source_destination_representation",
            "events",
            "original_bytes",
            "payload_bytes",
            "header_bytes",
            "committed_bytes",
            "committed_over_original",
            "native_source_bytes",
            "compression_desired_events",
            "compression_fallback_events",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in summaries:
            for scope, metrics_key in (
                ("all_conversions", "metrics"),
                ("scan_outputs", "scan_output_metrics"),
            ):
                for tier, values in (item[metrics_key] or {"none": {}}).items():
                    writer.writerow(
                        {
                            **{
                                k: item[k]
                                for k in ("query", "repeat", "sorted", "status")
                            },
                            "scope": scope,
                            "query_seconds": (item["result"] or {}).get(
                                "query_seconds"
                            ),
                            "source_destination_representation": tier,
                            **values,
                        }
                    )
    print(f"Wrote {args.output / 'summary.json'} and summary.csv")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
