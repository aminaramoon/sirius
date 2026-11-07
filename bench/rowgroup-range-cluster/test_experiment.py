import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "rowgroup_experiment", HERE / "experiment.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class HarnessTests(unittest.TestCase):
    def test_cluster_parser_keeps_membership(self):
        value = m.cluster_metrics(
            "[rowgroup_range_cluster] mode=cluster eligible=128\n[rowgroup_range_cluster_split] files=2 rowgroups=4 rows=32768 bytes=524288 span_min=-5 span_max=9 membership=0:0,1:0,0:1,1:1"
        )
        self.assertEqual(value["crossfile_splits"], 1)
        self.assertEqual(value["splits"][0]["membership"], "0:0,1:0,0:1,1:1")
        self.assertEqual(value["splits"][0]["span_min"], "-5")

    def test_assessment_requires_scan_not_downstream_spill(self):
        info = {"returncode": 0, "result": {"rows": [["1", "2"]]}}
        sort = {"sorted_batches": 1}
        cluster = {
            "plans": [{"eligible": "true", "reordered": "true", "files": "16"}],
            "multi_rg_splits": 1,
            "crossfile_splits": 1,
        }
        self.assertIn(
            "inconclusive_no_direct_scan",
            m.assess(
                info, [["1", "2"]], "1", "narrow-correlated", sort, cluster, {}, "HOST"
            ),
        )
        scan = {"GPU->HOST/compressed": {"events": 1}}
        self.assertEqual(
            m.assess(
                info,
                [["1", "2"]],
                "1",
                "narrow-correlated",
                sort,
                cluster,
                scan,
                "HOST",
            ),
            "ok",
        )
        self.assertEqual(
            m.assess(
                info,
                [["1", "2"]],
                "0",
                "narrow-correlated",
                sort,
                cluster,
                scan,
                "HOST",
            ),
            "failed_baseline_cluster_gate",
        )

    def test_no_crossfile_positive_is_inconclusive(self):
        self.assertEqual(
            m.assess(
                {"returncode": 0, "result": {"rows": []}},
                [],
                "1",
                "narrow-independent",
                {"sorted_batches": 1},
                {"plans": [{}], "multi_rg_splits": 1, "crossfile_splits": 0},
                {"GPU->HOST/compressed": {"events": 1}},
                "HOST",
            ),
            "inconclusive_no_crossfile_split",
        )

    def test_membership_is_scoped_per_contract(self):
        c = {
            "plans": [
                {"contract_id": "a", "rowgroups": "2"},
                {"contract_id": "b", "rowgroups": "1"},
            ],
            "splits": [
                {"contract_id": "a", "rowgroups": "2", "membership": "0:0,0:1"},
                {"contract_id": "b", "rowgroups": "1", "membership": "0:0"},
            ],
        }
        self.assertEqual([a["status"] for a in m.audit_membership(c)], ["ok", "ok"])
        c["splits"][0]["membership"] = "0:0,0:0"
        self.assertTrue(m.audit_membership(c)[0]["status"].startswith("failed"))

    def test_baseline_observer_without_plan(self):
        c = {
            "plans": [],
            "splits": [
                {
                    "contract_id": "a",
                    "key": "k",
                    "rowgroups": "2",
                    "membership": "0:0,1:0",
                },
                {
                    "contract_id": "a",
                    "key": "k",
                    "rowgroups": "2",
                    "membership": "0:1,1:1",
                },
            ],
        }
        audit = m.audit_membership(c)
        self.assertEqual(audit[0]["status"], "ok")
        self.assertEqual(audit[0]["signature"], ["k", "2", "4"])
        self.assertEqual(audit[0]["unique_rowgroups"], 4)

    def test_real_schedule_honors_cap(self):
        spec = importlib.util.spec_from_file_location(
            "rowgroup_prep", HERE / "prepare_controls.py"
        )
        prep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prep)
        stats = prep.schedule_stats(
            [("f", i, 8193, i * 10, i * 10 + 9) for i in range(130)]
        )
        self.assertLessEqual(max(stats["batch_rows"]), 524288)
        self.assertEqual(stats["batches"], 3)


if __name__ == "__main__":
    unittest.main()
