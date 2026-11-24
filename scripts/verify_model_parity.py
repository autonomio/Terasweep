"""Compare Terasweep predictions with upstream references on retained inputs."""

import argparse
import csv
import hashlib
import json
import os
import struct
import subprocess
import tarfile
from datetime import date
from importlib.metadata import version
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from lightgbm.basic import _LIB as LIGHTGBM_LIBRARY
from limen.sfd.reference_architecture.dlinear_regressor import DLinearRegressor
from limen.sfd.reference_architecture.ridge_regressor import RidgeRegressor

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "benchmarks/archives/terasweep_reverse_funnel_experiment.tar.gz"
MATRIX_NAMES = ("train.bin", "test.bin", "price.bin")
MODELS = ("ridge", "dlinear", "lightgbm", "tide")
TOL = 1e-10


def retained_fixture(work):
    with tarfile.open(ARCHIVE, "r:gz") as archive:

        def member(name):
            return archive.extractfile(f"terasweep-integrated/common/{name}").read()

        hashes = json.loads(member("provenance.json"))["sha256"]
        for name in (*MATRIX_NAMES, "sweep.json", "plan.csv", "raw.parquet"):
            data = member(name)
            if hashlib.sha256(data).hexdigest() != hashes[name]:
                raise ValueError(f"Retained Ridge fixture hash differs for {name}")
            (work / name).write_bytes(data)


def read_matrix(path):
    data = path.read_bytes()
    magic, rows, cols = struct.unpack_from("<8sQQ", data)
    if magic != b"RIDGE001" or len(data) != 24 + rows * (cols + 1) * 8:
        raise ValueError(f"Invalid matrix {path}")
    values = np.frombuffer(data, dtype="<f8", offset=24)
    return values[: rows * cols].reshape(rows, cols), values[rows * cols :]


def write_matrix(path, x, y):
    path.write_bytes(
        struct.pack("<8sQQ", b"RIDGE001", *x.shape)
        + np.asarray(x, dtype="<f8").tobytes()
        + np.asarray(y, dtype="<f8").tobytes()
    )


def run(binary, work, data, config, rows, plan=None):
    export = work / "predictions"
    summary = work / "summary.json"
    config_file = work / "config.json"
    config_file.write_text(json.dumps(config))
    env = dict(
        os.environ, TERASWEEP_LEVEL="compact", TERASWEEP_EXPORT_PREDICTIONS=str(export)
    )
    env.pop("TERASWEEP_PLAN", None)
    if config.get("model") == "lightgbm":
        env.setdefault("TERASWEEP_LIGHTGBM_LIB", LIGHTGBM_LIBRARY._name)
    if plan is not None:
        env["TERASWEEP_PLAN"] = str(plan)
    subprocess.run(
        [
            binary,
            *(data / name for name in MATRIX_NAMES),
            config_file,
            str(rows),
            summary,
        ],
        env=env,
        check=True,
    )
    return export, json.loads(summary.read_text())


def compare(params, actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not actual.size:
        raise ValueError(
            f"Prediction shape differs for {params}: {actual.shape} vs {expected.shape}"
        )
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError(f"Nonfinite predictions for {params}")
    close = np.isclose(actual, expected, rtol=TOL, atol=TOL)
    return {
        "params": params,
        "predictions": actual.size,
        "max_abs_difference": float(np.max(np.abs(actual - expected))),
        "bit_identical": np.array_equal(
            actual.view(np.uint64), expected.view(np.uint64)
        ),
        "predictions_outside_tolerance": int(np.count_nonzero(~close)),
        "signal_mismatches": int(np.count_nonzero((actual > 0) != (expected > 0))),
    }


def comparison_result(reference, cases):
    passed = all(
        not (case["predictions_outside_tolerance"] or case["signal_mismatches"])
        for case in cases
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "reference": reference,
        "cases": cases,
    }


def ridge(binary, work, fixture):
    plan = fixture / "plan.csv"
    with plan.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grid = {
        (float(row["alpha"]), row["fit_intercept"] == "true", float(row["solver_eps"]))
        for row in rows
    }
    config = json.loads((fixture / "sweep.json").read_text())
    export, summary = run(binary, work, fixture, config, len(rows), plan)
    if summary["rows"] != len(rows) or summary["failed_rows"]:
        raise ValueError("Ridge did not complete the retained native plan")
    train_x, train_y = read_matrix(fixture / "train.bin")
    test_x, _ = read_matrix(fixture / "test.bin")
    data = {"x_train": train_x, "y_train": train_y, "x_test": test_x}
    blob = export.read_bytes()
    magic, count = struct.unpack_from("<8sQ", blob)
    if magic != b"PRED0001" or count != len(grid):
        raise ValueError("Invalid Ridge export header for the retained plan")
    cases = []
    offset = 16
    for _ in range(count):
        alpha, intercept, eps, failed, size = struct.unpack_from("<dQdQQ", blob, offset)
        config_key = (alpha, bool(intercept), eps)
        if failed or config_key not in grid or size != len(test_x):
            raise ValueError(f"Missing or invalid Ridge fit for {config_key}")
        grid.remove(config_key)
        offset += 40
        actual = np.frombuffer(blob, dtype="<f8", count=size, offset=offset)
        offset += size * 8
        params = {"alpha": alpha, "fit_intercept": bool(intercept)}
        expected = RidgeRegressor().train(data, **params).predict(data)["_preds"]
        cases.append(compare(params, actual, expected))
    if grid or offset != len(blob):
        raise ValueError("Ridge export did not exactly cover the retained plan")
    return comparison_result("Limen RidgeRegressor (scikit-learn)", cases)


def json_fits(path, axes):
    expected = set(product(*axes.values()))
    with path.open() as stream:
        for line in stream:
            fit = json.loads(line)
            params = fit["params"]
            if params.keys() != axes.keys():
                raise ValueError(f"Unexpected fit parameters: {params}")
            key = tuple(params[name] for name in axes)
            if fit["fit_failed"] or key not in expected:
                raise ValueError(f"Missing or invalid fit for {params}")
            expected.remove(key)
            yield params, fit["predictions"]
    if expected:
        raise ValueError(f"Missing exported fits: {expected}")


def return_splits(fixture):
    raw = pl.read_parquet(fixture / "raw.parquet")
    for start, end in [
        (date(2026, 1, 1), date(2026, 1, 23)),
        (date(2026, 1, 31), date(2026, 2, 6)),
    ]:
        session = raw.filter(pl.col("datetime").dt.date().is_between(start, end))
        close = session["close"].to_numpy()
        x = close[1:-1] / close[:-2] - 1.0
        y = (close[2:] / close[1:-1] - 1.0) * 100.0
        yield x, y, session["open"].to_numpy()[1:-1], close[1:-1]


def lag_frame(returns, lookback):
    indices = np.arange(len(returns))
    return pl.DataFrame(
        {
            f"ret_1_lag_{lag}": returns[np.maximum(indices - lag, 0)]
            for lag in range(lookback - 1, -1, -1)
        }
    )


def dlinear(binary, work, fixture):
    (train_x, train_y, _, _), (test_x, test_y, opens, closes) = return_splits(fixture)
    write_matrix(work / "train.bin", train_x[:, None], train_y)
    write_matrix(work / "test.bin", test_x[:, None], test_y)
    (work / "price.bin").write_bytes(
        struct.pack("<8sQ", b"PRICE001", len(test_x))
        + opens.astype("<f8").tobytes()
        + closes.astype("<f8").tobytes()
    )
    config = json.loads((ROOT / "configs/dlinear.json").read_text())
    config.update(
        alphas={"log_min": -1, "log_max": 1, "count": 3},
        fit_intercept=[True],
        solver_eps=[1e-14],
        lookbacks=[2, 3],
        kernels=[1, 3],
    )
    axes = {
        "alpha": [0.1, 1.0, 10.0],
        "fit_intercept": config["fit_intercept"],
        "solver_eps": config["solver_eps"],
        "lookback": config["lookbacks"],
        "kernel": config["kernels"],
    }
    export, _ = run(binary, work, work, config, 1)
    cases = []
    for params, actual in json_fits(export, axes):
        data = {
            "x_train": lag_frame(train_x, params["lookback"]),
            "y_train": train_y,
            "x_test": lag_frame(test_x, params["lookback"]),
        }
        expected = (
            DLinearRegressor()
            .train(data, alpha=params["alpha"], kernel_size=params["kernel"])
            .predict(data)["_preds"]
        )
        cases.append(compare(params, actual, expected))
    result = comparison_result(
        "Limen DLinearRegressor (univariate return windows)", cases
    )
    result["scope"] = (
        "Identical single-channel causal windows with fitted intercept. The default 55-channel Rust workload has a different feature contract."
    )
    return result


def lightgbm(binary, work, _fixture):
    data = ROOT / "data"
    train_x, train_y = read_matrix(data / "train.bin")
    test_x, _ = read_matrix(data / "test.bin")
    config = json.loads((ROOT / "configs/lightgbm.json").read_text())
    axes = {
        "learning_rate": [0.01, 0.1],
        "num_trees": [4, 32],
        "num_leaves": [7],
        "max_bins": [8, 32],
        "lambda_l2": [0.1],
        "min_data_in_leaf": [20],
    }
    config.update(axes)
    export, _ = run(binary, work, data, config, 1)
    cases = []
    for params, actual in json_fits(export, axes):
        model = LGBMRegressor(
            objective="regression",
            learning_rate=params["learning_rate"],
            n_estimators=params["num_trees"],
            num_leaves=params["num_leaves"],
            max_bin=params["max_bins"],
            reg_lambda=params["lambda_l2"],
            min_child_samples=params["min_data_in_leaf"],
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            random_state=config["seed"] & (2**31 - 1),
            verbosity=-1,
        )
        model.fit(train_x, train_y)
        expected = model.booster_.predict(test_x, num_threads=1)
        cases.append(compare(params, actual, expected))
    result = comparison_result(
        "Official LightGBM LGBMRegressor (continuous return target)", cases
    )
    if any(not case["bit_identical"] for case in cases):
        result["status"] = "FAIL"
    return result


def tide(binary, work, _fixture):
    config = json.loads((ROOT / "configs/tide.json").read_text())
    axes = (
        "lookbacks",
        "hidden_width",
        "learning_rate",
        "epochs",
        "l2",
        "init_scale",
        "activation",
    )
    for axis in axes:
        config[axis] = config[axis][:1]
    _, summary = run(binary, work, ROOT / "data", config, 1)
    if summary["valid_rows"] != 1:
        raise ValueError(
            "The current TiDE-style model smoke case did not fit and score"
        )
    return {
        "status": "BLOCKED",
        "reference": "Google Research TiDE",
        "reason": "The Rust model is a one-hidden-layer scalar regressor with a linear skip. "
        "The upstream model has separate residual encoder, decoder and temporal decoder blocks "
        "and a covariate input contract. Weights and training settings have no direct mapping.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", metavar="MODEL", help=", ".join(MODELS))
    parser.add_argument("--binary", type=Path, default=ROOT / "target/terasweep-funnel")
    args = parser.parse_args()
    models = args.models or MODELS
    if any(model not in MODELS for model in models):
        parser.error(f"Choose models from {', '.join(MODELS)}")
    checks = {"ridge": ridge, "dlinear": dlinear, "lightgbm": lightgbm, "tide": tide}
    with TemporaryDirectory(prefix="terasweep-parity-") as temporary:
        work = Path(temporary)
        fixture = work / "retained"
        if "ridge" in models or "dlinear" in models:
            fixture.mkdir()
            retained_fixture(fixture)
        results = {}
        for model in dict.fromkeys(models):
            model_work = work / model
            model_work.mkdir()
            results[model] = checks[model](args.binary.resolve(), model_work, fixture)
    print(
        json.dumps(
            {
                "versions": {
                    package: version(package)
                    for package in (
                        "vaquum-limen",
                        "numpy",
                        "polars",
                        "pyarrow",
                        "scipy",
                        "scikit-learn",
                        "lightgbm",
                    )
                },
                "rtol": TOL,
                "atol": TOL,
                "results": results,
            },
            indent=2,
        )
    )
    return int(any(result["status"] != "PASS" for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
