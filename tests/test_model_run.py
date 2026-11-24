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
        self.packet["model"] = {
            "kind": "dlinear",
            "alpha": 1.0,
            "lookback": 1,
            "kernel": 1,
            "solver_eps": 1e-12,
        }
        result = invoke("fit", "-", input=json.dumps(self.packet))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertAlmostEqual(json.loads(result.stdout)["predictions"][0], 20 / 3)
        self.packet["model"]["kernel"] = 2
        self.assertNotEqual(
            invoke("fit", "-", input=json.dumps(self.packet)).returncode, 0
        )

    def test_extra_keys_and_unsupported_versions_fail(self):
        for packet in (dict(self.packet, runs=100), dict(self.packet, version=2)):
            result = invoke("fit", "-", input=json.dumps(packet))
            self.assertEqual(result.returncode, 1)
            self.assertFalse(result.stdout)
            self.assertIsInstance(json.loads(result.stderr)["error"], str)


if __name__ == "__main__":
    unittest.main()
