#!/usr/bin/env python3
"""New multifile paired harness. CPU-check is safe before feature builds; run is GPU.

Reuses only old harness query worker/metric parsing helpers; never edits old files.
"""

import argparse
import importlib.util
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
spec = importlib.util.spec_from_file_location(
    "old_scan_experiment", HERE.parent / "scan-sort-spill" / "experiment.py"
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def save(path, value):
    old.save(path, value)


def load_manifest(data):
    manifest = json.loads((data / "manifest.json").read_text())
    facts = [(HERE / p).resolve() for p in manifest["fact_files"]]
    dim = (HERE / manifest["dim_file"]).resolve()
    if not all(p.parent == data.resolve() for p in facts + [dim]):
        raise ValueError(
            "Manifest files must all belong to the selected data directory"
        )
    return manifest, facts, dim


def verify_files(data):
    manifest, facts, dim = load_manifest(data)
    for path in facts + [dim]:
        if old.sha256_file(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError("Input hash mismatch: " + str(path))
    return manifest


def connect(data):
    import duckdb

    _, facts, dim = load_manifest(data)
    con = duckdb.connect(
        config={
            "allow_unsigned_extensions": "true",
            "threads": "4",
            "temp_directory": str(Path.cwd() / "cpu-tmp"),
        }
    )
    con.execute("SET memory_limit='4GiB'")
    con.execute("SET preserve_insertion_order=true")
    file_list = "[" + ",".join(old.literal(p) for p in facts) + "]"
    con.execute(f"CREATE VIEW fact AS SELECT * FROM read_parquet({file_list})")
    con.execute(f"CREATE VIEW dim AS SELECT * FROM read_parquet({old.literal(dim)})")
    return con


def cluster_metrics(text):
    plans, splits = [], []
    for line in text.splitlines():
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if "[rowgroup_range_cluster_split]" in line:
            splits.append(fields)
        elif "[rowgroup_range_cluster]" in line:
            plans.append(fields)
    return {
        "plans": plans,
        "splits": splits,
        "crossfile_splits": sum(int(x.get("files", "0")) > 1 for x in splits),
        "multi_rg_splits": sum(int(x.get("rowgroups", "0")) > 1 for x in splits),
    }


def launch(args, directory, worker, arm="0"):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "spill").mkdir()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker,
        "--data",
        str(args.data),
        "--extension",
        str(args.extension),
        "--result",
        str(directory / "result.json"),
    ]
    if args.full_rows:
        command.append("--full-rows")
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        SIRIUS_EXP_SCAN_BATCH_SORT="1",
        SIRIUS_EXP_ROWGROUP_RANGE_CLUSTER=arm,
        SIRIUS_EXP_ROWGROUP_RANGE_AUDIT="1",
        SIRIUS_EXP_SCAN_SPILL_METRICS="1",
        SIRIUS_LOG_BACKEND="spdlog",
        SIRIUS_LOG_LEVEL="debug",
        SIRIUS_LOG_DIR=str(directory / "log"),
    )
    if worker == "cpu":
        env["SIRIUS_DISABLE"] = "1"
    else:
        env.pop("SIRIUS_DISABLE", None)
    config = args.config.read_text().replace("@SPILL_DIR@", str(directory / "spill"))
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
    start = time.perf_counter()
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
    result = (
        json.loads((directory / "result.json").read_text())
        if (directory / "result.json").exists()
        else None
    )
    if args.full_rows and result:
        result["rows"] = sorted(result["rows"])
    return {
        "returncode": code,
        "timeout": timed_out,
        "wall_seconds": time.perf_counter() - start,
        "result": result,
    }


def audit_membership(cluster):
    """Per-contract inventory, without conflating fact and dimension file indices."""
    results = []
    plans = list(cluster["plans"])
    if not plans:
        # Baseline observer emits split records only; infer contract geometry.
        for contract in sorted(
            {
                s.get("contract_id")
                for s in cluster["splits"]
                if s.get("contract_id") is not None
            }
        ):
            splits = [s for s in cluster["splits"] if s.get("contract_id") == contract]
            members = [
                x for s in splits for x in s.get("membership", "").split(",") if x
            ]
            plans.append(
                {
                    "contract_id": contract,
                    "key": splits[0].get("key"),
                    "files": str(len({x.split(":")[0] for x in members})),
                    "rowgroups": str(len(members)),
                    "inferred_baseline": "true",
                }
            )
    for plan in plans:
        contract = plan.get("contract_id")
        if contract is None:
            results.append({"status": "unverified_missing_contract_id", "plan": plan})
            continue
        splits = [s for s in cluster["splits"] if s.get("contract_id") == contract]
        members = [x for s in splits for x in s.get("membership", "").split(",") if x]
        expected = int(plan.get("rowgroups", "0"))
        status = (
            "ok"
            if len(members) == len(set(members)) == expected
            else "failed_duplicate_or_missing_membership"
        )
        if any(
            len(s.get("membership", "").split(",")) != int(s.get("rowgroups", "0"))
            for s in splits
        ):
            status = "failed_split_membership_count"
        results.append(
            {
                "contract_id": contract,
                "status": status,
                "expected_rowgroups": expected,
                "observed_rowgroups": len(members),
                "unique_rowgroups": len(set(members)),
                "inventory": sorted(members),
                "signature": [plan.get(k) for k in ("key", "files", "rowgroups")],
            }
        )
    return results


def assess(info, reference, arm, dataset, sort, cluster, scan_metrics, tier):
    if info["returncode"] or not info["result"] or info["result"]["rows"] != reference:
        return "failed_execution_or_reference"
    if not sort["sorted_batches"]:
        return "inconclusive_no_actual_scan_sort"
    if arm == "0" and cluster["plans"]:
        return "failed_baseline_cluster_gate"
    if arm != "0" and not cluster["multi_rg_splits"]:
        return "inconclusive_no_grouped_split"
    if arm == "1" and dataset.startswith("narrow-") and not cluster["crossfile_splits"]:
        return "inconclusive_no_crossfile_split"
    if (
        arm == "1"
        and dataset.startswith("narrow-")
        and not any(
            p.get("eligible") in ("true", "1")
            and p.get("reordered") in ("true", "1")
            and int(p.get("files", "0")) > 1
            for p in cluster["plans"]
        )
    ):
        return "inconclusive_positive_not_eligible_and_reordered"
    if not any(
        key.endswith("->" + tier + "/compressed") and value["events"]
        for key, value in scan_metrics.items()
    ):
        return "inconclusive_no_direct_scan_compressed_" + tier.lower() + "_spill"
    return "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("cpu-check", "run", "correctness"),
        default="cpu-check",
    )
    parser.add_argument("--data", type=Path, default=HERE / "data-narrow-correlated")
    parser.add_argument("--output", type=Path, default=HERE / "results-cpu-check")
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE.parent / "scan-sort-spill" / "host-pressure.yaml",
    )
    parser.add_argument(
        "--extension",
        type=Path,
        default=REPO
        / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension",
    )
    parser.add_argument(
        "--arms", nargs="+", choices=("0", "1", "collect"), default=["0", "1"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--require-tier", choices=("HOST", "DISK"))
    parser.add_argument(
        "--full-rows",
        action="store_true",
        help="Exact full-row multiset mode, limited to <=100000 input rows",
    )
    parser.add_argument("--worker", choices=("cpu", "gpu"), help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for attr in ("data", "output", "config", "extension"):
        setattr(args, attr, getattr(args, attr).resolve())
    if min(args.repeats, args.timeout) <= 0 or len(set(args.arms)) != len(args.arms):
        parser.error("Positive repeats/timeout and distinct arms required")
    args.full_rows |= args.action == "correctness"
    if args.full_rows and load_manifest(args.data)[0]["rows"] > 100000:
        parser.error("Full-row validation is restricted to tiny <=100000-row fixtures")
    if args.worker:
        # Old worker retains ORDER_BY checks/fallback policy/exact string results.
        if args.full_rows:
            old.QUERIES["join"] = (
                "SELECT f.k,f.payload FROM fact f JOIN dim d ON f.k=d.k"
            )
        old.connect = connect
        args.query = "join"
        old.worker(args)
        return
    if args.action in ("run", "correctness") and not args.extension.is_file():
        parser.error("Built extension required for GPU run")
    manifest = verify_files(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "data-manifest.json", manifest)
    patch = subprocess.check_output(["git", "diff", "--binary"], cwd=REPO)
    (args.output / "engine-working-tree.patch").write_bytes(patch)
    library = REPO / "build/release/install/lib/libsirius.so.0.0.0"
    save(
        args.output / "provenance.json",
        {
            "installed_library": str(library),
            "installed_library_sha256": (
                old.sha256_file(library) if library.exists() else None
            ),
            "working_tree_patch_sha256": old.hashlib.sha256(patch).hexdigest(),
            "harness_sha256": old.sha256_file(Path(__file__)),
            "base_harness_sha256": old.sha256_file(
                HERE.parent / "scan-sort-spill" / "experiment.py"
            ),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "extension": str(args.extension),
            "extension_sha256": (
                old.sha256_file(args.extension) if args.action == "run" else None
            ),
            "python": sys.version,
            "action": args.action,
            "scan_sort_always_enabled": True,
            "arm_definitions": {
                "0": "batchsort-only",
                "1": "cluster+batchsort",
                "collect": "eager-original-order+batchsort",
            },
        },
    )
    cpu = launch(args, args.output / "cpu", "cpu")
    save(args.output / "cpu" / "status.json", cpu)
    reference = [manifest["cpu_join_reference"]]
    if args.full_rows and cpu["result"]:
        reference = cpu["result"]["rows"]
        full_aggregate = [
            str(len(reference)),
            str(sum(int(row[1]) for row in reference)),
        ]
        if full_aggregate != manifest["cpu_join_reference"]:
            raise SystemExit("CPU full-row aggregate differs from prepared reference")
    if cpu["returncode"] or not cpu["result"] or cpu["result"]["rows"] != reference:
        raise SystemExit("CPU reference mismatch/failure; see cpu/status.json")
    if args.action == "cpu-check":
        verify_files(args.data)
        print("CPU-only reference and all input hashes verified; no extension loaded.")
        return
    tier = args.require_tier or (
        "DISK" if "downgrade_root_dirs:" in args.config.read_text() else "HOST"
    )
    records = []
    for repeat in range(args.repeats):
        arms = args.arms if repeat % 2 == 0 else list(reversed(args.arms))
        for arm in arms:
            directory = args.output / f"join-r{repeat+1}-cluster{arm}"
            info = launch(args, directory, "gpu", arm)
            native = "\n".join(
                p.read_text(errors="replace") for p in (directory / "log").glob("*.log")
            )
            sort = old.parse_sort_metrics(native)
            outputs = old.parse_scan_outputs(native)
            scan = old.parse_metrics(directory, batch_ids=set(outputs))
            cluster = cluster_metrics(native)
            if args.full_rows:
                status = (
                    "ok_full_row_multiset"
                    if not info["returncode"]
                    and info["result"]
                    and info["result"]["rows"] == reference
                    else "failed_full_row_multiset"
                )
            else:
                status = assess(
                    info, reference, arm, manifest["dataset"], sort, cluster, scan, tier
                )
            audit = audit_membership(cluster)
            if any(a["status"].startswith("failed") for a in audit):
                status = "failed_membership_audit"
            elif status in ("ok", "ok_full_row_multiset") and (
                not audit or any(a["status"] != "ok" for a in audit)
            ):
                status = "inconclusive_membership_audit"
            record = {
                "repeat": repeat + 1,
                "arm": arm,
                "status": status,
                "directory": str(directory),
                "sort_metrics": sort,
                "cluster_metrics": cluster,
                "membership_audit": audit,
                "scan_outputs": outputs,
                "scan_output_metrics": scan,
                "all_conversion_metrics": old.parse_metrics(directory),
                **info,
            }
            save(directory / "status.json", record)
            records.append(record)
            save(args.output / "summary.json", records)
    verify_files(args.data)
    comparisons = {}
    for record in records:
        for audit in record["membership_audit"]:
            if audit["status"] != "ok":
                continue
            signature = "|".join(str(x) for x in audit["signature"])
            entry = comparisons.setdefault(
                signature,
                {
                    "identical_inventories": True,
                    "arms": [],
                    "inventory": audit["inventory"],
                },
            )
            entry["identical_inventories"] &= entry["inventory"] == audit["inventory"]
            entry["arms"].append({"arm": record["arm"], "repeat": record["repeat"]})
    run_inventories = [
        {
            "arm": r["arm"],
            "repeat": r["repeat"],
            "contracts": sorted(
                (str(a["signature"][0]), a["inventory"])
                for a in r["membership_audit"]
                if a["status"] == "ok"
            ),
        }
        for r in records
    ]
    complete_run_match = bool(
        run_inventories and run_inventories[0]["contracts"]
    ) and all(
        r["contracts"] == run_inventories[0]["contracts"] for r in run_inventories
    )
    save(
        args.output / "inventory-comparison.json",
        {
            "all_run_contract_inventories_identical": complete_run_match,
            "run_inventories": run_inventories,
            "by_key_filecount_rowgroupcount": comparisons,
            "baseline_note": "Gate0 observer emits split markers without inventory plan; baseline contract geometry is inferred from split records and checked against other arms.",
            "contract_note": "Different contract IDs matched by key/files/rowgroups; inspect raw contracts for any ambiguous signatures.",
        },
    )
    if not complete_run_match or any(
        not c["identical_inventories"] for c in comparisons.values()
    ):
        raise SystemExit(
            "Different rowgroup inventories across arms/repeats; inspect inventory-comparison.json"
        )
    print(
        json.dumps(
            [
                {"repeat": r["repeat"], "arm": r["arm"], "status": r["status"]}
                for r in records
            ],
            indent=2,
        )
    )
    if any(r["status"].startswith("failed") for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
