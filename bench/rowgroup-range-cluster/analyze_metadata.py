"""CPU-only footer survey. Never loads Sirius or changes existing inputs."""

import json
from pathlib import Path
import duckdb

HERE = Path(__file__).resolve().parent
con = duckdb.connect(config={"threads": "4", "memory_limit": "1GiB"})


def quantiles(values):
    v = sorted(values)
    return {str(p): v[round((len(v) - 1) * p)] for p in (0, 0.1, 0.5, 0.9, 1)}


results = {}
raw = {}
paths = {
    "sf1000_lineitem": "/localhome/local-aaramoon/tpch/sf1000/lineitem/*.parquet",
    "sf1_lineitem": "/localhome/local-aaramoon/tpch/sf1/lineitem/*.parquet",
    **{
        name: str(HERE.parent / "scan-sort-spill" / name / "fact.parquet")
        for name in ("data", "data-tpch", "data-tpch-orderkey")
    },
}
for directory in sorted(HERE.glob("data-*")):
    if (directory / "manifest.json").exists():
        paths[directory.name] = str(directory / "fact*.parquet")
for name, path in paths.items():
    rows = con.execute(
        """SELECT file_name,row_group_id,row_group_num_rows,path_in_schema,
        try_cast(stats_min_value AS BIGINT),try_cast(stats_max_value AS BIGINT),
        stats_null_count,min_is_exact,max_is_exact
        FROM parquet_metadata(?) WHERE path_in_schema IN ('l_suppkey','l_orderkey','k')
        ORDER BY file_name,row_group_id,path_in_schema""",
        [path],
    ).fetchall()
    raw[name] = rows
    result = {}
    for key in sorted({r[3] for r in rows}):
        rr = [r for r in rows if r[3] == key]
        valid = [r for r in rr if r[4] is not None and r[5] is not None]
        lo, hi = min(r[4] for r in valid), max(r[5] for r in valid)
        span = hi - lo + 1
        widths = [r[5] - r[4] + 1 for r in valid]
        result[key] = {
            "files": len({r[0] for r in rr}),
            "row_groups": len(rr),
            "rows": sum(r[2] for r in rr),
            "rows_per_rg_quantiles": quantiles([r[2] for r in rr]),
            "global_min": lo,
            "global_max": hi,
            "width_quantiles": quantiles(widths),
            "width_over_global_span_quantiles": quantiles([w / span for w in widths]),
            "missing_minmax": len(rr) - len(valid),
            "exact_minmax": sum(r[7] is True and r[8] is True for r in rr),
            "null_rows": sum(r[6] or 0 for r in rr),
            "adjacent_overlap_fraction": sum(
                a[4] <= b[5] and b[4] <= a[5] for a, b in zip(valid, valid[1:])
            )
            / max(1, len(valid) - 1),
            "first_ranges": [
                {"file": Path(r[0]).name, "rg": r[1], "min": r[4], "max": r[5]}
                for r in valid[:4]
            ],
        }
    results[name] = result
(HERE / "metadata-raw.json").write_text(json.dumps(raw, indent=2) + "\n")
(HERE / "metadata-summary.json").write_text(
    json.dumps({"duckdb_version": duckdb.__version__, "datasets": results}, indent=2)
    + "\n"
)
print(json.dumps(results, indent=2))
