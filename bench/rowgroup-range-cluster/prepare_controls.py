"""Bounded CPU-only shared-layout controls; never loads a DuckDB extension.

Run with pixi run python bench/rowgroup-range-cluster/prepare_controls.py.
Existing output directories are never overwritten. Defaults: two ~512MiB facts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import duckdb

HERE = Path(__file__).resolve().parent
RG_ROWS = 8192
FILES = 16
GROUPS_PER_FILE = 256
BUCKETS = 128
KEYS_PER_BUCKET = 16384
KEYS = BUCKETS * KEYS_PER_BUCKET


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def schedule_stats(groups):
    # Greedy whole-RG packing at equal 8MiB decoded cap; real RG sizes vary.
    batches = []
    current, rows = [], 0
    for group in groups:
        if current and rows + group[2] > 524288:
            batches.append(current)
            current, rows = [], 0
        current.append(group)
        rows += group[2]
    if current:
        batches.append(current)
    spans = sorted(max(g[4] for g in b) - min(g[3] for g in b) + 1 for b in batches)
    return {
        "batches": len(batches),
        "span_min": min(spans),
        "span_median": spans[len(spans) // 2],
        "span_max": max(spans),
        "span_over_domain_median": spans[len(spans) // 2]
        / (max(g[4] for g in groups) - min(g[3] for g in groups) + 1),
        "source_files_per_batch_min": min(len({g[0] for g in b}) for b in batches),
        "source_files_per_batch_max": max(len({g[0] for g in b}) for b in batches),
        "batch_rows": sorted({sum(g[2] for g in b) for b in batches}),
    }


def prepare(name):
    out = HERE / ("data-" + name)
    out.mkdir(exist_ok=False)
    con = duckdb.connect(
        config={
            "threads": "1",
            "memory_limit": "1GiB",
            "temp_directory": str(out / "tmp"),
        }
    )
    con.execute("SET preserve_insertion_order=true")
    statements = []
    paths = []
    for file_id in range(FILES):
        start = file_id * GROUPS_PER_FILE * RG_ROWS
        end = start + GROUPS_PER_FILE * RG_ROWS
        key = (
            (
                f"((17 * (i // {RG_ROWS})) % {BUCKETS}) * {KEYS_PER_BUCKET} + "
                f"hash(i+1729) % {KEYS_PER_BUCKET}"
            )
            if name != "uniform-wide"
            else f"hash(i+1729) % {KEYS}"
        )
        payload = (
            "(hash(i+900001) % 1125899906842624)::BIGINT"
            if name == "narrow-independent"
            else "(k*1000003 + (hash(i+1729)%8)::BIGINT)::BIGINT"
        )
        sql = f"SELECT k, {payload} AS payload FROM (SELECT i, ({key})::BIGINT AS k FROM range({start},{end}) t(i))"
        path = out / f"fact-{file_id:02d}.parquet"
        con.execute(
            f"COPY ({sql}) TO {literal(path)} (FORMAT PARQUET, COMPRESSION UNCOMPRESSED, ROW_GROUP_SIZE {RG_ROWS})"
        )
        paths.append(path)
        statements.append(sql)
    dim = out / "dim.parquet"
    con.execute(
        f"COPY (SELECT i::BIGINT AS k FROM range({KEYS}) t(i)) TO {literal(dim)} (FORMAT PARQUET, COMPRESSION UNCOMPRESSED, ROW_GROUP_SIZE {RG_ROWS})"
    )
    file_list = "[" + ",".join(literal(p) for p in paths) + "]"
    meta = con.execute(
        f"SELECT file_name,row_group_id,row_group_num_rows,stats_min_value::BIGINT,stats_max_value::BIGINT FROM parquet_metadata({file_list}) WHERE path_in_schema='k' ORDER BY file_name,row_group_id"
    ).fetchall()
    assert len(meta) == FILES * GROUPS_PER_FILE
    assert all(g[2] == RG_ROWS for g in meta)
    clustered = sorted(meta, key=lambda g: (g[3], g[4], g[0], g[1]))
    assert sorted((g[0], g[1]) for g in clustered) == sorted((g[0], g[1]) for g in meta)
    fingerprint_sql = f"SELECT count(*),sum(k::HUGEINT),sum(payload::HUGEINT),sum(hash(k,payload)::HUGEINT),min(k),max(k) FROM read_parquet({file_list})"
    fingerprint = [str(x) for x in con.execute(fingerprint_sql).fetchone()]
    query = f"SELECT count(*),sum(f.payload) FROM read_parquet({file_list}) f JOIN read_parquet({literal(dim)}) d ON f.k=d.k"
    reference = [str(x) for x in con.execute(query).fetchone()]
    assert reference[0] == str(FILES * GROUPS_PER_FILE * RG_ROWS)
    assert reference[1] == fingerprint[2]
    manifest = {
        "duckdb_version": duckdb.__version__,
        "cpu_only": True,
        "dataset": name,
        "rows": FILES * GROUPS_PER_FILE * RG_ROWS,
        "row_group_rows": RG_ROWS,
        "keys": KEYS,
        "buckets": BUCKETS if name != "uniform-wide" else None,
        "sql_by_file": statements,
        "fact_files": [str(p.relative_to(HERE)) for p in paths],
        "dim_file": str(dim.relative_to(HERE)),
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in paths + [dim]
        },
        "fingerprint_sql": fingerprint_sql,
        "fingerprint": fingerprint,
        "cpu_join_sql": query,
        "cpu_join_reference": reference,
        "physical_schedule": schedule_stats(meta),
        "clustered_schedule": schedule_stats(clustered),
        "schedule_note": "Metadata-only equal 8MiB cap simulation, not measured engine batching or compression.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "schedules.json").write_text(
        json.dumps({"physical": meta, "clustered": clustered}, indent=2) + "\n"
    )
    (HERE / (name + "-summary.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("dataset", "rows", "physical_schedule", "clustered_schedule")
            },
            indent=2,
        ),
        flush=True,
    )
    con.close()


def prepare_real(name):
    source_name = "data-tpch" if name == "real-supplier" else "data-tpch-orderkey"
    source = HERE.parent / "scan-sort-spill" / source_name
    out = HERE / ("data-" + name)
    out.mkdir(exist_ok=False)
    con = duckdb.connect(
        config={
            "threads": "1",
            "memory_limit": "1GiB",
            "temp_directory": str(out / "tmp"),
        }
    )
    con.execute("SET preserve_insertion_order=true")
    statements = []
    for table in ("fact", "dim"):
        sql = f"SELECT * FROM read_parquet({literal(source / (table+'.parquet'))})"
        statements.append(sql)
        con.execute(
            f"COPY ({sql}) TO {literal(out / (table+'.parquet'))} (FORMAT PARQUET, COMPRESSION UNCOMPRESSED, ROW_GROUP_SIZE {RG_ROWS})"
        )

    def fingerprint(path):
        return [
            str(x)
            for x in con.execute(
                f"SELECT count(*),sum(k::HUGEINT),sum(payload::HUGEINT),sum(hash(k,payload)::HUGEINT),min(k),max(k) FROM read_parquet({literal(path)})"
            ).fetchone()
        ]

    fp = fingerprint(out / "fact.parquet")
    assert fp == fingerprint(source / "fact.parquet")
    # Full multiset check, not only aggregate fingerprints.
    mismatch = con.execute(
        f"SELECT count(*) FROM ((SELECT * FROM read_parquet({literal(source/'fact.parquet')}) EXCEPT ALL SELECT * FROM read_parquet({literal(out/'fact.parquet')})) UNION ALL (SELECT * FROM read_parquet({literal(out/'fact.parquet')}) EXCEPT ALL SELECT * FROM read_parquet({literal(source/'fact.parquet')})))"
    ).fetchone()[0]
    assert mismatch == 0
    meta = con.execute(
        "SELECT file_name,row_group_id,row_group_num_rows,stats_min_value::BIGINT,stats_max_value::BIGINT FROM parquet_metadata(?) WHERE path_in_schema='k' ORDER BY file_name,row_group_id",
        [str(out / "fact.parquet")],
    ).fetchall()
    clustered = sorted(meta, key=lambda g: (g[3], g[4], g[0], g[1]))
    query = f"SELECT count(*),sum(f.payload) FROM read_parquet({literal(out/'fact.parquet')}) f JOIN read_parquet({literal(out/'dim.parquet')}) d ON f.k=d.k"
    ref = [str(x) for x in con.execute(query).fetchone()]
    assert ref == [fp[0], fp[2]]
    manifest = {
        "duckdb_version": duckdb.__version__,
        "cpu_only": True,
        "dataset": name,
        "rows": int(fp[0]),
        "row_group_rows": RG_ROWS,
        "source_manifest": json.loads((source / "manifest.json").read_text()),
        "source_fact_sha256": digest(source / "fact.parquet"),
        "sql_by_table": statements,
        "fact_files": [str((out / "fact.parquet").relative_to(HERE))],
        "dim_file": str((out / "dim.parquet").relative_to(HERE)),
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in (out / "fact.parquet", out / "dim.parquet")
        },
        "fingerprint": fp,
        "source_multiset_difference_rows": mismatch,
        "cpu_join_sql": query,
        "cpu_join_reference": ref,
        "physical_schedule": schedule_stats(meta),
        "clustered_schedule": schedule_stats(clustered),
        "schedule_note": "Metadata-only equal8MiB cap simulation, not observed engine batches or codec compression.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "schedules.json").write_text(
        json.dumps({"physical": meta, "clustered": clustered}, indent=2) + "\n"
    )
    (HERE / (name + "-summary.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("dataset", "rows", "physical_schedule", "clustered_schedule")
            },
            indent=2,
        ),
        flush=True,
    )
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=(
            "narrow-correlated",
            "narrow-independent",
            "uniform-wide",
            "real-supplier",
            "real-orderkey",
        ),
        default=["narrow-correlated", "uniform-wide"],
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        (prepare_real if dataset.startswith("real-") else prepare)(dataset)
