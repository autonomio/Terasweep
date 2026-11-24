"""Measure a factorized Terasweep and an independently refitted reference."""

import argparse
import hashlib
import json
import operator
import os
import platform
import re
import struct
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODELS = ("ridge", "dlinear", "lightgbm", "tide")
SIZES = (
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
)
RULES = ("gt", "gte", "lt", "lte", "abs_gt", "abs_gte")
FILES = ("train.bin", "test.bin", "price.bin", "config.json")
TRAIN_ROWS = 256
TEST_ROWS = 128
MAX_PROJECTED_10K_SECONDS = 150


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(blob):
    magic, rows, cols = struct.unpack_from("<8sQQ", blob)
    if magic != b"RIDGE001" or len(blob) != 24 + 8 * rows * (cols + 1):
        raise ValueError("Invalid matrix fixture")
    values = np.frombuffer(blob, dtype="<f8", offset=24)
    return values[: rows * cols].reshape(rows, cols), values[rows * cols :]


def prices(blob):
    magic, rows = struct.unpack_from("<8sQ", blob)
    if magic != b"PRICE001" or len(blob) != 16 + 16 * rows:
        raise ValueError("Invalid price fixture")
    return np.frombuffer(blob, dtype="<f8", offset=16).reshape(2, rows)


def prepare(root):
    sys.path.insert(0, str(ROOT / "scripts"))
    from prepare_return_inputs import prepare as return_inputs
    from verify_model_parity import retained_fixture

    ridge = root / "source-ridge"
    returns = root / "source-returns"
    ridge.mkdir(exist_ok=True)
    retained_fixture(ridge)
    return_inputs(returns)
    for model, source, config in (
        ("ridge", ridge, read_json(ROOT / "configs/sweep.json")),
        ("dlinear", returns, read_json(ROOT / "configs/dlinear.json")),
        ("lightgbm", ROOT / "data", read_json(ROOT / "configs/lightgbm.json")),
        ("tide", returns, read_json(ROOT / "configs/tide.json")),
    ):
        work = root / model
        work.mkdir(exist_ok=True)
        for split, rows in (("train", TRAIN_ROWS), ("test", TEST_ROWS)):
            x, y = matrix((source / f"{split}.bin").read_bytes())
            if len(x) < rows:
                raise ValueError(f"Insufficient rows for {model} {split}")
            (work / f"{split}.bin").write_bytes(
                struct.pack("<8sQQ", b"RIDGE001", rows, x.shape[1])
                + x[:rows].tobytes()
                + y[:rows].tobytes()
            )
        price = prices((source / "price.bin").read_bytes())[:, :TEST_ROWS]
        (work / "price.bin").write_bytes(
            struct.pack("<8sQ", b"PRICE001", TEST_ROWS) + price.tobytes()
        )
        if model == "ridge":
            config.update(
                alphas={"log_min": 0, "log_max": 2, "count": 3}, solver_eps=[1e-12]
            )
        elif model == "lightgbm":
            config.update(
                learning_rate=[0.05, 0.1],
                num_trees=[8, 16],
                num_leaves=[7],
                max_bins=[8, 16],
                lambda_l2=[0.1],
                min_data_in_leaf=[10],
            )
        elif model == "tide":
            config.update(hidden_width=[8], learning_rate=[0.001, 0.003], epochs=[2])
        save_json(work / "config.json", config)


def environment():
    from lightgbm.basic import _LIB

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        POLARS_MAX_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        TF_NUM_INTRAOP_THREADS="1",
        TF_USE_LEGACY_KERAS="0",
        TF_CPP_MIN_LOG_LEVEL="2",
        TERASWEEP_LEVEL="compact",
        TERASWEEP_LIGHTGBM_LIB=_LIB._name,
    )
    for key in (
        "TERASWEEP_PLAN",
        "TERASWEEP_CHECK_BASE",
        "TERASWEEP_EXPORT_PREDICTIONS",
    ):
        env.pop(key, None)
    return env


def native(binary, work, size, env, name=None):
    name = name or f"native-{size}"
    summary = work / f"{name}.json"
    log = work / f"{name}.log"
    start = perf_counter()
    with log.open("w") as output:
        done = subprocess.run(
            [binary, *(work / filename for filename in FILES), str(size), summary],
            stderr=output,
            env=env,
        )
    wall = perf_counter() - start
    if done.returncode:
        raise RuntimeError(f"{name} failed: {log.read_text()}")
    data = read_json(summary)
    if data["rows"] != size or data["failed_rows"] or data["failed_train_configs"]:
        raise ValueError(f"{name} failed to score all {size:,} rows")
    phases = re.search(
        r"PIPELINE plan=([.\d]+) mixed_score=([.\d]+) costs_reduce=([.\d]+)",
        log.read_text(),
    )
    if phases is None:
        raise ValueError(f"Missing compact-pipeline diagnostics in {log}")
    return {
        "seconds": data["hot_loop_seconds"],
        "wall_seconds": wall,
        "rows": data["rows"],
        "unique_training_configs": data["unique_train_configs"],
        "score_cache_entries": data["score_cache_entries"],
        "phase_seconds": dict(
            zip(
                ("schedule_lookup", "mixed_scoring", "cost_reduce"),
                map(float, phases.groups()),
            )
        ),
        "summary_sha256": hashlib.sha256(
            json.dumps(stable(data), sort_keys=True).encode()
        ).hexdigest(),
    }


def stable(data):
    return {
        key: value for key, value in data.items() if not key.startswith("hot_loop_")
    }


def run_native(root, binary, models, sizes, env):
    for model in models:
        work = root / model
        for size in sizes:
            print(f"{model}: measuring Terasweep at {size:,}", flush=True)
            result = native(binary, work, size, env)
            save_json(work / f"timed-native-{size}.json", result)
            print(
                f"{model}: {size:,} completed in {result['seconds']:.4f}s", flush=True
            )


def cached_predictions(model, export):
    from worker import KEYS

    if model != "ridge":
        parsed = {}
        for line in export.read_text().splitlines():
            fit = json.loads(line)
            if fit["fit_failed"]:
                raise ValueError(f"Invalid native fit in {export}")
            parsed[tuple(fit["params"][field] for field in KEYS[model])] = np.array(
                fit["predictions"]
            )
        return parsed
    blob = export.read_bytes()
    magic, count = struct.unpack_from("<8sQ", blob)
    if magic != b"PRED0001":
        raise ValueError("Invalid Ridge prediction export")
    parsed = {}
    offset = 16
    for _ in range(count):
        alpha, fit_intercept, eps, failed, rows = struct.unpack_from(
            "<dQdQQ", blob, offset
        )
        offset += 40
        if failed:
            raise ValueError("Invalid native Ridge fit")
        parsed[alpha, fit_intercept, eps] = np.frombuffer(
            blob, dtype="<f8", count=rows, offset=offset
        )
        offset += rows * 8
    if offset != len(blob):
        raise ValueError("Unexpected Ridge prediction export length")
    return parsed


def check_reference(work, model, size, result, expected, lib):
    from worker import numbers, plan

    rust_summary = read_json(work / f"native-{size}.json")
    for metric, winner in result["winners"].items():
        rust_winner = rust_summary[metric]
        if (winner is None) != (rust_winner is None) or (
            winner is not None
            and any(rust_winner[field] != value for field, value in winner.items())
        ):
            raise ValueError(
                f"{model} at {size:,}: unequal {metric}: {winner} vs {rust_winner}"
            )
    measured = {
        tuple(key): np.array(values) for key, values in result.pop("predictions")
    }
    largest_difference = 0.0
    for key, actual in measured.items():
        expected_fit = expected[key]
        np.testing.assert_allclose(actual, expected_fit, rtol=1e-10, atol=1e-10)
        if model in ("lightgbm", "tide") and not np.array_equal(
            actual.view(np.uint64), expected_fit.view(np.uint64)
        ):
            raise ValueError(f"{model} fit was not bit-identical for {key}")
        largest_difference = max(
            largest_difference, float(np.max(np.abs(actual - expected_fit)))
        )
    plan_rows = plan(lib, work / "config.json", size)
    seen = set()
    counts = dict.fromkeys(RULES, 0)
    compare = (
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    )
    for trial in plan_rows:
        counts[RULES[trial.rule]] += 1
        key = numbers(trial, model)
        state = key, trial.rule, trial.threshold
        if state in seen:
            continue
        seen.add(state)
        actual, native_predictions = measured[key], expected[key]
        if trial.rule >= 4:
            actual, native_predictions = np.abs(actual), np.abs(native_predictions)
        op = compare[trial.rule]
        np.testing.assert_array_equal(
            op(actual, trial.threshold), op(native_predictions, trial.threshold)
        )
    if counts != rust_summary["signal_rule_counts"]:
        raise ValueError(f"{model} at {size:,}: trial rule counts differ")
    return {
        "training_configs": len(measured),
        "score_keys": len(seen),
        "max_absolute_prediction_difference": largest_difference,
        "signal_differences": 0,
        "winner_summaries_match": True,
        "plan_sha256": hashlib.sha256(bytes(plan_rows)).hexdigest(),
    }


def run_reference(root, binary, models, env):
    from worker import library

    library_path = root / (
        "reference.dylib" if platform.system() == "Darwin" else "reference.so"
    )
    lib = library(library_path)
    for model in models:
        work = root / model
        export = work / "validation-predictions.bin"
        native(
            binary,
            work,
            1000,
            dict(
                env, TERASWEEP_EXPORT_PREDICTIONS=str(export), TERASWEEP_CHECK_BASE="1"
            ),
            name="validation-native",
        )
        expected = cached_predictions(model, export)
        executable = env["TERASWEEP_TIDE_PYTHON"] if model == "tide" else sys.executable
        for size in (10, 100, 1000, 10_000):
            if (
                size == 10_000
                and estimate(read_json(work / "timed-reference-1000.json"), size)
                > MAX_PROJECTED_10K_SECONDS
            ):
                print(
                    f"{model}: using the 1,000-trial calibration for larger projections",
                    flush=True,
                )
                break
            print(f"{model}: measuring upstream reference at {size:,}", flush=True)
            log = work / f"reference-{size}.log"
            start = perf_counter()
            with log.open("w") as logs:
                done = subprocess.run(
                    [
                        executable,
                        "-B",
                        HERE / "worker.py",
                        model,
                        work,
                        str(size),
                        library_path,
                    ],
                    env=env,
                    stdout=subprocess.PIPE,
                    text=True,
                    stderr=logs,
                )
            wall_seconds = perf_counter() - start
            if done.returncode:
                raise RuntimeError(f"Reference failed: {log.read_text()}")
            result = json.loads(done.stdout)
            result["wall_seconds"] = wall_seconds
            result["parity"] = check_reference(work, model, size, result, expected, lib)
            save_json(work / f"timed-reference-{size}.json", result)
            print(
                f"{model}: reference completed {size:,} in {result['seconds']:.4f}s and passed parity",
                flush=True,
            )


def estimate(calibration, size):
    return (
        calibration["setup_seconds"]
        + calibration["trial_seconds"] * size / calibration["rows"]
    )


def report(root, binary, models):
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    info = {
        "version": 1,
        "hardware": {
            "cpu": cpu,
            "machine": platform.machine(),
            "os": platform.platform(),
        },
        "runner_sha256": digest(binary),
        "runner_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "compiler": subprocess.check_output(
            [os.environ.get("TERASWEEP_BENCH_RUSTC", "rustc"), "--version"], text=True
        ).strip(),
        "timing": "Each point runs serially in a separate process with the scalar scorer. Timing excludes prepared matrix and price loading, price statistics, and output serialization.",
        "projection": "backend setup plus trial count times observed trial seconds per row in largest reference calibration",
        "reference_10k_budget_seconds": MAX_PROJECTED_10K_SECONDS,
        "models": {},
    }
    for model in models:
        work = root / model
        references = {
            size: read_json(work / f"timed-reference-{size}.json")
            for size in (10, 100, 1000, 10_000)
            if (work / f"timed-reference-{size}.json").is_file()
        }
        if not {10, 100, 1000}.issubset(references):
            raise ValueError(f"Missing reference calibration for {model}")
        anchor = references[max(references)]
        rows = []
        for size in SIZES:
            rust = read_json(work / f"timed-native-{size}.json")
            observed = size in references
            reference = (
                references[size]["seconds"] if observed else estimate(anchor, size)
            )
            rows.append(
                {
                    "trials": size,
                    "terasweep": rust,
                    "reference_seconds": reference,
                    "reference_kind": "measured" if observed else "projected",
                    "ratio": reference / rust["seconds"],
                }
            )
        info["models"][model] = {
            "train_shape": list(matrix((work / "train.bin").read_bytes())[0].shape),
            "test_shape": list(matrix((work / "test.bin").read_bytes())[0].shape),
            "fixture_sha256": {name: digest(work / name) for name in FILES},
            "config": read_json(work / "config.json"),
            "reference_calibration": references,
            "projection_anchor_trials": anchor["rows"],
            "scaling": rows,
        }
    print(json.dumps(info, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "native", "reference", "report"))
    parser.add_argument("--work", type=Path, default=ROOT / "runs/sweep-scaling")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/terasweep-funnel")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--sizes", nargs="+", type=int, choices=SIZES, default=SIZES)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        prepare(work)
    elif args.phase == "native":
        run_native(work, args.binary.resolve(), args.models, args.sizes, environment())
    elif args.phase == "reference":
        run_reference(work, args.binary.resolve(), args.models, environment())
    else:
        report(work, args.binary.resolve(), args.models)


if __name__ == "__main__":
    main()
