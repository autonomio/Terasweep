"""Check single fits and ordered sweeps on public regression fixtures."""
import argparse
import json
import os
import struct
import subprocess
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from lightgbm import LGBMRegressor
from lightgbm.basic import _LIB
from limen.sfd.reference_architecture.dlinear_regressor import DLinearRegressor
from limen.sfd.reference_architecture.ridge_regressor import RidgeRegressor

from model_run import DEFAULT_BINARY, ROOT, fit, load_matrix, validate_request

FIXTURES = ROOT / "tests/fixtures"
DATASET = {"ridge": "concrete", "lightgbm": "abalone", "dlinear": "bike", "tide": "bike"}
TOLERANCE = 1e-10


def settings():
    return {
        "ridge": [
            dict(alpha=alpha, fit_intercept=True, solver_eps=0.0)
            for alpha in (1.0, 0.01, 0.1, 0.5, 3.0, 10.0, 50.0, 200.0)
        ],
        "lightgbm": [
            dict(learning_rate=rate, num_trees=trees, num_leaves=leaves,
                 max_bins=63, lambda_l2=0.5, min_data_in_leaf=15, seed=375)
            for rate, trees, leaves in product([0.07, 0.14], [20, 45], [9, 21])
        ],
        "dlinear": [
            dict(alpha=alpha, lookback=lookback, kernel=kernel, solver_eps=0.0)
            for alpha, lookback, kernel in product([0.25, 4.0], [7, 14], [3, 7])
        ],
        "tide": [
            dict(lookback=lookback, hidden_width=width, learning_rate=0.004, epochs=epochs, seed=375)
            for lookback, width, epochs in product([7, 14], [8, 16], [2, 4])
        ],
    }


def matrix(path):
    data, _ = load_matrix(path)
    _, rows, columns = struct.unpack_from("<8sQQ", data)
    x, y = np.split(np.frombuffer(data, dtype="<f8", offset=24), [rows * columns])
    return x.reshape(rows, columns), y


def windows(x, lookback):
    history = np.pad(x[:, 0], (lookback - 1, 0), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(history, lookback)


def reference(kind, params, train, validation, work):
    x, y = train
    xt, _ = validation
    data = {"x_train": x, "y_train": y, "x_test": xt}
    if kind == "ridge":
        model = RidgeRegressor().train(data, alpha=params["alpha"], fit_intercept=True)
        return model.predict(data)["_preds"]
    if kind == "dlinear":
        import polars as pl

        lookback = params["lookback"]
        for key, values in (("x_train", x), ("x_test", xt)):
            channels = windows(values, lookback)
            data[key] = pl.DataFrame({
                f"ret_1_lag_{lag}": channels[:, column]
                for column, lag in enumerate(range(lookback - 1, -1, -1))
            })
        model = DLinearRegressor().train(data, alpha=params["alpha"], kernel_size=params["kernel"])
        return model.predict(data)["_preds"]
    if kind == "lightgbm":
        model = LGBMRegressor(
            objective="regression", learning_rate=params["learning_rate"],
            n_estimators=params["num_trees"], num_leaves=params["num_leaves"],
            max_bin=params["max_bins"], reg_lambda=params["lambda_l2"],
            min_child_samples=params["min_data_in_leaf"], random_state=params["seed"],
            n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
        ).fit(x, y)
        return model.booster_.predict(xt, num_threads=1)

    input_path, output_path = work / "windows.npz", work / "reference.bin"
    lookback = params["lookback"]
    np.savez(
        input_path,
        train=windows(x.astype("float32"), lookback), labels=y.astype("float32"),
        test=windows(xt.astype("float32"), lookback),
    )
    env = dict(
        os.environ, TF_USE_LEGACY_KERAS="0", OMP_NUM_THREADS="1",
        TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1",
    )
    subprocess.run(
        [env["TERASWEEP_TIDE_PYTHON"], "-B", ROOT / "scripts/tide_reference_predictions.py",
         env["TERASWEEP_TIDE_SOURCE"], input_path, json.dumps(params), output_path],
        env=env, check=True,
    )
    return np.fromfile(output_path, dtype="<f8")


def bits_equal(a, b):
    return np.array_equal(a.view("u8"), b.view("u8"))


def audit(kind, cases, binary, work):
    fixture = FIXTURES / DATASET[kind]
    train, validation = (str(fixture / f"{part}.bin") for part in ("train", "validation"))
    train_data, validation_data = matrix(train), matrix(validation)
    predictions, references, plan = [], [], []
    for params in cases:
        request = dict(version=1, train=train, test=validation, model=dict(kind=kind, **params))
        predictions.append(fit(request, binary)["predictions"])
        references.append(reference(kind, params, train_data, validation_data, work))
        plan.append(" ".join(validate_request(request)))

    actual = np.asarray(predictions, dtype="float64")
    expected = np.asarray(references, dtype="float64")
    np.testing.assert_allclose(actual, expected, atol=TOLERANCE, rtol=TOLERANCE)
    if any(f"{a:.10f}" != f"{b:.10f}" for a, b in zip(actual.flat, expected.flat)):
        raise AssertionError(f"{kind} predictions differ at ten decimal places")
    bit_equal = bits_equal(actual, expected)
    if kind in ("lightgbm", "tide") and not bit_equal:
        raise AssertionError(f"{kind} predictions differ from upstream at the bit level")

    plan_path = work / "models.txt"
    plan_path.write_text("\n".join([*plan, plan[0]]) + "\n", encoding="utf-8")
    receipt = json.loads(subprocess.check_output(
        [binary, "sweep", train, validation, plan_path], text=True,
    ))
    expected_mae = np.abs(expected - validation_data[1]).mean(axis=1)
    winner = int(np.argmin(expected_mae))
    assert receipt["trials"] == len(plan) + 1
    assert receipt["unique_fits"] == len(plan)
    assert receipt["best"]["trial"] == winner
    assert bits_equal(np.asarray(receipt["best"]["predictions"]), actual[winner])
    np.testing.assert_allclose(receipt["best"]["mae"], expected_mae[winner], atol=TOLERANCE, rtol=TOLERANCE)

    return {
        "fits": len(cases), "predictions": actual.size, "bit_identical": bit_equal,
        "max_abs_error": float(np.abs(actual - expected).max()),
        "sweep": {key: receipt[key] for key in ("trials", "unique_fits", "elapsed_seconds")},
        "best_trial": winner, "validation_mae": receipt["best"]["mae"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--quick", action="store_true", help="check one setting per model")
    parser.add_argument("models", nargs="*", metavar="MODEL", help=", ".join(DATASET))
    args = parser.parse_args()
    models = args.models or list(DATASET)
    if set(models) - DATASET.keys():
        parser.error("unsupported model name")
    os.environ.setdefault("TERASWEEP_LIGHTGBM_LIB", _LIB._name)

    results = {}
    with TemporaryDirectory() as temp:
        for model in models:
            cases = settings()[model]
            results[model] = audit(model, cases[:1] if args.quick else cases, args.binary.resolve(), Path(temp))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
