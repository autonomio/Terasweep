import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts/model_run.py"
BINARY = os.environ.get("TERASWEEP_TEST_BINARY", str(ROOT / "target/terasweep-funnel"))


def invoke(*args, input=None):
    return subprocess.run(
        [sys.executable, "-B", TOOL, "--binary", BINARY, *args],
        input=input,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )


class ModelRunTest(unittest.TestCase):
    def setUp(self):
        fixture = TemporaryDirectory()
        self.addCleanup(fixture.cleanup)
        root = Path(fixture.name)
        train, test = root / "train.bin", root / "test.bin"
        train.write_bytes(struct.pack("<8sQQ6d", b"RIDGE001", 3, 1, 1, 2, 3, 2, 4, 6))
        test.write_bytes(struct.pack("<8sQQ4d", b"RIDGE001", 2, 1, 4, 5, -999, 999))
        self.packet = {
            "version": 1,
            "model": {
                "kind": "ridge",
                "alpha": 1.0,
                "fit_intercept": True,
                "solver_eps": 1e-12,
            },
            "train": str(train),
            "test": str(test),
        }

    def test_ridge_cli_and_mcp_share_same_predictions(self):
        cli = invoke("fit", "-", input=json.dumps(self.packet))
        self.assertEqual(cli.returncode, 0, cli.stderr)
        expected = json.loads(cli.stdout)
        self.assertAlmostEqual(expected["predictions"][0], 20 / 3)
        self.assertAlmostEqual(expected["predictions"][1], 8)
        self.assertEqual(expected["train"]["rows"], 3)
        self.assertGreater(expected["fit_predict_seconds"], 0)
        requests = [
            {
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-11-25"},
            },
            {"method": "notifications/initialized"},
            {"id": 2, "method": "tools/list"},
            {
                "id": 3,
                "method": "tools/call",
                "params": {"name": "terasweep_fit", "arguments": self.packet},
            },
        ]
        protocol = "".join(
            json.dumps(dict(request, jsonrpc="2.0")) + "\n" for request in requests
        )
        mcp = invoke("mcp", input=protocol)
        self.assertEqual(mcp.returncode, 0, mcp.stderr)
        init, listing, run = map(json.loads, mcp.stdout.splitlines())
        self.assertEqual(init["result"]["protocolVersion"], "2025-11-25")
        schemas = json.loads(invoke("schema").stdout)
        self.assertEqual(
            listing["result"]["tools"][0]["inputSchema"], schemas["request"]
        )
        actual = run["result"]["structuredContent"]
        for receipt in (actual, expected):
            receipt.pop("fit_predict_seconds")
        self.assertEqual(actual, expected)

    def test_dlinear_single_window(self):
        # JSON Schema treats numeric values with zero fractional part as integers.
        self.packet["version"] = 1.0
        self.packet["model"] = {
            "kind": "dlinear",
            "alpha": 1.0,
            "lookback": 1.0,
            "kernel": 1.0,
            "solver_eps": 1e-12,
        }
        result = invoke("fit", "-", input=json.dumps(self.packet))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertAlmostEqual(json.loads(result.stdout)["predictions"][0], 20 / 3)
        self.packet["model"]["kernel"] = 2
        invalid = invoke("fit", "-", input=json.dumps(self.packet))
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("dlinear.kernel must be odd", invalid.stderr)

    def test_extra_keys_and_unsupported_versions_fail(self):
        for packet in (dict(self.packet, runs=100), dict(self.packet, version=2)):
            result = invoke("fit", "-", input=json.dumps(packet))
            self.assertEqual(result.returncode, 1)
            self.assertFalse(result.stdout)
            self.assertIsInstance(json.loads(result.stderr)["error"], str)

    def test_seeded_ridge_sweep_matches_all_scalar_levels(self):
        root = Path(self.packet["train"]).parent
        config = root / "sweep.json"
        config.write_text(
            json.dumps(
                {
                    "seed": 20260514,
                    "alphas": {"log_min": 0, "log_max": 1, "count": 2},
                    "fit_intercept": [True, False],
                    "solver_eps": [1e-12],
                    "signal_rule": ["gt", "gte", "lt", "lte", "abs_gt", "abs_gte"],
                    "threshold_signed": {"min": 5, "max": 9, "step": 1},
                    "threshold_abs": {"min": 5, "max": 9, "step": 1},
                    "fee_bps": [0, 17],
                    "slippage_bps": [0, 9],
                }
            )
        )
        price = root / "price.bin"
        price.write_bytes(struct.pack("<8sQ4d", b"PRICE001", 2, 100, 100, 110, 103))
        out = root / "summary.json"
        env = dict(os.environ, TERASWEEP_CHECK_BASE="1")
        env.pop("TERASWEEP_PLAN", None)
        env.pop("TERASWEEP_EXPORT_PREDICTIONS", None)
        receipts = {}
        for level in ("naive", "stats", "train", "score", "bounds", "compact"):
            result = subprocess.run(
                [
                    BINARY,
                    self.packet["train"],
                    self.packet["test"],
                    price,
                    config,
                    "64",
                    out,
                ],
                env=dict(env, TERASWEEP_LEVEL=level),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(out.read_text())
            receipt.pop("hot_loop_seconds")
            receipt.pop("hot_loop_iterations_per_second")
            keys = receipt.pop("score_cache_entries")
            receipts[level] = (receipt, keys)
        scalar, _ = receipts["naive"]
        self.assertEqual(scalar["rows"], 64)
        self.assertEqual(scalar["failed_rows"], 0)
        # Recorded from an independent SplitMix64 draw of the 64-trial prefix.
        self.assertEqual(
            scalar["signal_rule_counts"],
            {
                "gt": 8,
                "gte": 9,
                "lt": 14,
                "lte": 6,
                "abs_gt": 14,
                "abs_gte": 13,
            },
        )
        for level, (receipt, keys) in receipts.items():
            with self.subTest(level=level):
                self.assertEqual(receipt, scalar)
                self.assertEqual(
                    keys, 47 if level in ("score", "bounds", "compact") else 0
                )


if __name__ == "__main__":
    unittest.main()
