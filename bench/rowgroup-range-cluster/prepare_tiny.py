"""CPU-only tiny multifile row-association fixture; refuses overwrites."""

import json
from pathlib import Path
import duckdb
from prepare_controls import digest, literal, schedule_stats

HERE = Path(__file__).resolve().parent
out = HERE / "data-tiny"
out.mkdir(exist_ok=False)
con = duckdb.connect(
    config={
        "threads": "1",
        "memory_limit": "256MiB",
        "temp_directory": str(out / "tmp"),
    }
)
con.execute("SET preserve_insertion_order=true")
paths = []
statements = []
for f in range(4):
    path = out / f"fact-{f:02d}.parquet"
    sql = f"""SELECT k, (k*1000003+i%997)::BIGINT AS payload FROM
        (SELECT i, (((i//2048)*3)%8*128+hash(i+1729)%128)::BIGINT AS k
         FROM range({f*8192},{(f+1)*8192}) t(i))"""
    con.execute(
        f"COPY ({sql}) TO {literal(path)} (FORMAT PARQUET,COMPRESSION UNCOMPRESSED,ROW_GROUP_SIZE 2048)"
    )
    paths.append(path)
    statements.append(sql)
dim = out / "dim.parquet"
con.execute(
    f"COPY (SELECT i::BIGINT AS k FROM range(1024) t(i)) TO {literal(dim)} (FORMAT PARQUET,COMPRESSION UNCOMPRESSED,ROW_GROUP_SIZE 2048)"
)
file_list = "[" + ",".join(literal(p) for p in paths) + "]"
meta = con.execute(
    f"SELECT file_name,row_group_id,row_group_num_rows,stats_min_value::BIGINT,stats_max_value::BIGINT FROM parquet_metadata({file_list}) WHERE path_in_schema='k' ORDER BY file_name,row_group_id"
).fetchall()
assert len(meta) == 16 and all(r[2] == 2048 for r in meta)
query = f"SELECT count(*),sum(f.payload) FROM read_parquet({file_list}) f JOIN read_parquet({literal(dim)}) d ON f.k=d.k"
ref = [str(x) for x in con.execute(query).fetchone()]
assert ref[0] == "32768"
manifest = {
    "dataset": "tiny",
    "duckdb_version": duckdb.__version__,
    "cpu_only": True,
    "rows": 32768,
    "row_group_rows": 2048,
    "keys": 1024,
    "sql_by_file": statements,
    "fact_files": [str(p.relative_to(HERE)) for p in paths],
    "dim_file": str(dim.relative_to(HERE)),
    "files": {
        p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in paths + [dim]
    },
    "cpu_join_sql": query,
    "cpu_join_reference": ref,
    "physical_schedule": schedule_stats(meta),
    "clustered_schedule": schedule_stats(
        sorted(meta, key=lambda r: (r[3], r[4], r[0], r[1]))
    ),
    "purpose": "Full-row multiset GPU correctness fixture, not spill-pressure measurement.",
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(out / "schedules.json").write_text(
    json.dumps(
        {
            "physical": meta,
            "clustered": sorted(meta, key=lambda r: (r[3], r[4], r[0], r[1])),
        },
        indent=2,
    )
    + "\n"
)
(HERE / "tiny-summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
con.close()
print(
    "Prepared CPU-only tiny fixture: 32768 rows,4 files,16 rowgroups; no extension loaded."
)
