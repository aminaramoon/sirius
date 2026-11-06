"""CPU-only harness tests; no extension or GPU required."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment", HERE / "experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
analysis_spec = importlib.util.spec_from_file_location("analyze", HERE / "analyze.py")
analyze = importlib.util.module_from_spec(analysis_spec)
analysis_spec.loader.exec_module(analyze)


class MetricsTest(unittest.TestCase):
    def test_only_committed_bytes_and_separate_directions(self):
        with tempfile.TemporaryDirectory(dir=HERE) as name:
            path = Path(name)
            (path / "native.log").write_text(
                "downgrade to_host bytes=999999999\n"
                "[scan_sort_spill] event=attempt source_bytes=999\n"
                "[scan_sort_spill] event=conversion_committed batch_id=4 source_tier=GPU target_tier=HOST source_bytes=1000 gpu_uncompressed_bytes=1000 representation=compressed payload_bytes=90 serialized_bytes=100 compression_desired=1 compression_fallback=0\n"
                "[scan_sort_spill] event=conversion_committed batch_id=4 source_tier=HOST target_tier=DISK source_bytes=90 gpu_uncompressed_bytes=1000 representation=compressed payload_bytes=90 serialized_bytes=100 compression_desired=1 compression_fallback=0\n"
                "[scan_sort_spill] event=conversion_committed batch_id=5 source_tier=GPU target_tier=HOST source_bytes=2000 gpu_uncompressed_bytes=2000 representation=uncompressed payload_bytes=2000 serialized_bytes=2000 compression_desired=1 compression_fallback=1\n"
            )
            values = experiment.parse_metrics(path)
            self.assertEqual(len(values), 3)
            host = values["GPU->HOST/compressed"]
            self.assertEqual(host["committed_bytes"], 100)
            self.assertEqual(host["header_bytes"], 10)
            self.assertEqual(host["committed_over_original"], 0.1)
            self.assertEqual(values["HOST->DISK/compressed"]["original_bytes"], 1000)
            scan_values = experiment.parse_metrics(path, batch_ids={"4"})
            self.assertEqual(len(scan_values), 2)
            self.assertNotIn("GPU->HOST/uncompressed", scan_values)

    def test_successful_sort_rows(self):
        result = experiment.parse_sort_metrics(
            "[scan_batch_sort_plan] keys=1 scan_node_id=7 admitted=1\n"
            "[scan_batch_sort] rows=100 keys=1 operator_id=9\n"
            "[scan_batch_sort] rows=200 keys=1 operator_id=9\n"
        )
        self.assertEqual(result["sorted_batches"], 2)
        self.assertEqual(result["sorted_rows"], 300)
        self.assertEqual(result["plans"][0]["admitted"], "1")

    def test_scan_output_provenance(self):
        values = experiment.parse_scan_outputs(
            "[scan_batch_output] batch_id=42 operator_id=2 rows=100 bytes=1600 sort_active=true\n"
        )
        self.assertEqual(values["42"]["sort_active"], "true")
        self.assertEqual(values["42"]["rows"], "100")

    def test_destination_pooling_does_not_double_count_host_stage(self):
        def metric(original, committed):
            return {
                **{name: 0 for name in analyze.COUNTERS},
                "original_bytes": original,
                "committed_bytes": committed,
                "events": 1,
            }

        pooled = analyze.by_destination(
            {
                "GPU->HOST/compressed": metric(1000, 400),
                "HOST->DISK/compressed": metric(1000, 400),
                "GPU->DISK/compressed": metric(500, 100),
                "GPU->DISK/uncompressed": metric(99, 99),
            }
        )
        self.assertEqual(pooled["HOST"]["committed_bytes"], 400)
        self.assertEqual(pooled["DISK"]["committed_bytes"], 500)
        self.assertEqual(pooled["DISK"]["original_bytes"], 1500)
        self.assertEqual(pooled["DISK"]["events"], 2)
        self.assertAlmostEqual(pooled["DISK"]["committed_over_original"], 1 / 3)

    def test_sql_quoting(self):
        self.assertEqual(experiment.literal("can't"), "'can''t'")
        self.assertEqual(experiment.identifier('a"b'), '"a""b"')


if __name__ == "__main__":
    unittest.main()
