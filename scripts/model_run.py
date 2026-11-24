#!/usr/bin/env python3
"""Run one Terasweep model from a versioned CLI or MCP packet."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "target/terasweep-funnel"
MCP_PROTOCOL = "2025-11-25"
TOOL = "terasweep_fit"

POSITIVE = {"type": "number", "exclusiveMinimum": 0}
NONNEGATIVE = {"type": "number", "minimum": 0}
SIZE = {"type": "integer", "minimum": 1, "maximum": 2**64 - 1}
NATIVE_INT = dict(SIZE, maximum=2**31 - 1)
SEED = dict(SIZE, minimum=0, maximum=2**32 - 1)

FIELDS = {
    "ridge": {
        "alpha": NONNEGATIVE,
        "fit_intercept": {"type": "boolean"},
        "solver_eps": NONNEGATIVE,
    },
    "dlinear": {
        "alpha": POSITIVE,
        "lookback": SIZE,
        "kernel": {**SIZE, "not": {"multipleOf": 2}},
        "solver_eps": NONNEGATIVE,
    },
    "lightgbm": {
        "learning_rate": POSITIVE,
        "num_trees": NATIVE_INT,
        "num_leaves": dict(NATIVE_INT, minimum=2),
        "max_bins": dict(NATIVE_INT, minimum=2),
        "lambda_l2": NONNEGATIVE,
        "min_data_in_leaf": NATIVE_INT,
        "seed": SEED,
    },
    "tide": {
        "lookback": SIZE,
        "hidden_width": SIZE,
        "learning_rate": POSITIVE,
        "epochs": SIZE,
        "seed": SEED,
    },
}


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def schemas():
    models = {
        "oneOf": [
            object_schema(
                {
                    "kind": {"const": kind},
                    **fields,
                }
            )
            for kind, fields in FIELDS.items()
        ]
    }
    base = {"version": {"type": "integer", "const": 1}, "model": models}
    identity = object_schema(
        {
            "rows": {"type": "integer", "minimum": 1},
            "columns": {"type": "integer", "minimum": 1},
            "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        }
    )
    request = object_schema(
        dict(
            base,
            train={"type": "string", "minLength": 1},
            test={"type": "string", "minLength": 1},
        )
    )
    receipt = object_schema(
        dict(
            base,
            train=identity,
            test=identity,
            predictions={"type": "array", "items": {"type": "number"}},
            fit_predict_seconds={"type": "number", "minimum": 0},
        )
    )
    return {"version": 1, "request": request, "receipt": receipt}


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key}")
        result[key] = value
    return result


def parse(text):
    return json.loads(text, object_pairs_hook=unique_object)


def encoded(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def validate_request(packet):
    if not isinstance(packet, dict) or packet.keys() != {
        "version",
        "model",
        "train",
        "test",
    }:
        raise ValueError("The request requires exactly version, model, train and test")
    if type(packet["version"]) not in (int, float) or packet["version"] != 1:
        raise ValueError("Unsupported fit packet version. Use version 1")
    model = packet["model"]
    kind = model.get("kind") if isinstance(model, dict) else None
    if not isinstance(kind, str) or kind not in FIELDS:
        raise ValueError("Select one model kind from ridge, dlinear, lightgbm and tide")
    fields = FIELDS[kind]
    if model.keys() != {"kind", *fields}:
        raise ValueError(f"The {kind} model requires exactly: {', '.join(fields)}")
    return [
        kind,
        *(
            command_value(f"{kind}.{name}", model[name], rule)
            for name, rule in fields.items()
        ),
    ]


def command_value(name, value, rule):
    type_ = rule["type"]
    if type_ == "boolean":
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
        return str(value).lower()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite {type_}")
    if type_ == "integer":
        if value != int(value):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
    if "minimum" in rule and value < rule["minimum"]:
        raise ValueError(f"{name} must be at least {rule['minimum']}")
    if "exclusiveMinimum" in rule and value <= rule["exclusiveMinimum"]:
        raise ValueError(f"{name} must exceed {rule['exclusiveMinimum']}")
    if "maximum" in rule and value > rule["maximum"]:
        raise ValueError(f"{name} must be at most {rule['maximum']}")
    if "not" in rule and value % rule["not"]["multipleOf"] == 0:
        raise ValueError(f"{name} must be odd")
    return str(value)


def load_matrix(value):
    if not isinstance(value, str) or not value or not Path(value).is_file():
        raise ValueError("Matrix input must be an explicit local regular file")
    blob = Path(value).read_bytes()
    if len(blob) < 24:
        raise ValueError("Matrix header is incomplete")
    magic, rows, cols = struct.unpack_from("<8sQQ", blob)
    if (
        magic != b"RIDGE001"
        or not rows
        or not cols
        or len(blob) != 24 + rows * (cols + 1) * 8
    ):
        raise ValueError("Invalid RIDGE001 matrix length, shape or magic")
    identity = {
        "rows": rows,
        "columns": cols,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }
    return blob, identity


def fit(packet, binary):
    args = validate_request(packet)
    train, train_id = load_matrix(packet["train"])
    test, test_id = load_matrix(packet["test"])
    result = subprocess.run(
        [binary, "_fit", *args],
        input=struct.pack("<Q", len(train)) + train + test,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"Model fit failed ({result.returncode}): {detail}")
    if len(result.stdout) != (test_id["rows"] + 1) * 8:
        raise ValueError("The model returned an invalid prediction count")
    elapsed, *predictions = (
        value for (value,) in struct.iter_unpack("<d", result.stdout)
    )
    return {
        "version": 1,
        "model": packet["model"],
        "train": train_id,
        "test": test_id,
        "predictions": predictions,
        "fit_predict_seconds": elapsed,
    }


def mcp_error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def serve(binary):
    initialized = False
    for line in sys.stdin:
        try:
            message = parse(line)
        except ValueError:
            print(encoded(mcp_error(None, -32700, "Invalid JSON")), flush=True)
            continue
        if (
            not isinstance(message, dict)
            or message.get("jsonrpc") != "2.0"
            or not isinstance(message.get("method"), str)
        ):
            print(encoded(mcp_error(None, -32600, "Invalid request")), flush=True)
            continue
        if "id" not in message:
            continue
        id_ = message["id"]
        if type(id_) not in (str, int):
            print(encoded(mcp_error(None, -32600, "Invalid request ID")), flush=True)
            continue
        params, method = message.get("params", {}), message["method"]
        if not isinstance(params, dict):
            print(encoded(mcp_error(id_, -32602, "Invalid parameters")), flush=True)
            continue
        result = None
        if (
            method == "initialize"
            and not initialized
            and isinstance(params.get("protocolVersion"), str)
        ):
            initialized = True
            result = {
                "protocolVersion": MCP_PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "terasweep", "version": "1"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list" and initialized:
            types = schemas()
            result = {
                "tools": [
                    {
                        "name": TOOL,
                        "description": "Fit one model and predict from local matrix files.",
                        "inputSchema": types["request"],
                        "outputSchema": types["receipt"],
                        "annotations": {"readOnlyHint": True, "openWorldHint": False},
                    }
                ]
            }
        elif method == "tools/call" and initialized and params.get("name") == TOOL:
            try:
                receipt = fit(params.get("arguments"), binary)
                result = {
                    "structuredContent": receipt,
                    "content": [{"type": "text", "text": encoded(receipt)}],
                }
            except (OSError, ValueError, OverflowError) as error:
                result = {
                    "isError": True,
                    "content": [{"type": "text", "text": str(error)}],
                }
        response = (
            {"jsonrpc": "2.0", "id": id_, "result": result}
            if result is not None
            else mcp_error(id_, -32601, "Method not available")
        )
        print(encoded(response), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("fit").add_argument(
        "request", help="JSON packet path or - for stdin"
    )
    subcommands.add_parser("schema")
    subcommands.add_parser("mcp")
    args = parser.parse_args()
    try:
        if args.command == "schema":
            print(encoded(schemas()))
        elif args.command == "mcp":
            serve(args.binary.resolve())
        else:
            packet = parse(
                sys.stdin.read()
                if args.request == "-"
                else Path(args.request).read_text()
            )
            print(encoded(fit(packet, args.binary.resolve())))
    except (OSError, ValueError, OverflowError) as error:
        print(encoded({"version": 1, "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
