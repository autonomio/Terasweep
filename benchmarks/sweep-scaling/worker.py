"""Fit the canonical model afresh on every row in a production trial prefix."""

import ctypes as c
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from run import ROOT, matrix, prices, read_json

KEYS = {
    "ridge": ("alpha", "fit_intercept", "solver_eps"),
    "dlinear": ("alpha", "fit_intercept", "solver_eps", "lookback", "kernel"),
    "lightgbm": (
        "learning_rate",
        "num_trees",
        "num_leaves",
        "max_bins",
        "lambda_l2",
        "min_data_in_leaf",
    ),
    "tide": ("lookback", "hidden_width", "learning_rate", "epochs"),
}
DOUBLE_PTR = c.POINTER(c.c_double)


class Trial(c.Structure):
    _fields_ = [
        ("train", c.c_double * 6),
        ("threshold", c.c_double),
        ("fee", c.c_double),
        ("slippage", c.c_double),
        ("rule", c.c_size_t),
    ]


def library(path):
    lib = c.CDLL(str(path))
    lib.sweep_plan.argtypes = (c.c_char_p, c.c_size_t, c.POINTER(Trial))
    lib.sweep_plan.restype = c.c_int
    lib.sweep_scorer.argtypes = (DOUBLE_PTR, DOUBLE_PTR, DOUBLE_PTR, c.c_size_t)
    lib.sweep_scorer.restype = c.c_void_p
    lib.sweep_record.argtypes = (c.c_void_p, DOUBLE_PTR, c.POINTER(Trial), c.c_size_t)
    lib.sweep_record.restype = None
    lib.sweep_finish.argtypes = (c.c_void_p, DOUBLE_PTR)
    lib.sweep_finish.restype = None
    return lib


def plan(lib, config, size):
    output = (Trial * size)()
    if lib.sweep_plan(config.read_bytes(), size, output):
        raise ValueError(f"Failed to generate {size:,} trials from {config}")
    return output


def numbers(trial, model):
    return tuple(trial.train[: len(KEYS[model])])


def fit_model(model, x_train, train_y, x_test, seed):
    if model == "ridge":
        from limen.sfd.reference_architecture.ridge_regressor import RidgeRegressor

        arrays = {"x_train": x_train, "y_train": train_y, "x_test": x_test}

        def fit(training):
            alpha, fit_intercept, _ = training
            return (
                RidgeRegressor()
                .train(
                    arrays,
                    alpha=alpha,
                    fit_intercept=bool(fit_intercept),
                    solver="cholesky",
                )
                .predict(arrays)["_preds"]
            )

        return fit
    if model == "dlinear":
        from limen.sfd.reference_architecture.dlinear_regressor import DLinearRegressor

        import polars as pl

        def lag_frame(returns, lookback):
            indices = np.arange(len(returns))
            return pl.DataFrame(
                {
                    f"ret_1_lag_{lag}": returns[np.maximum(indices - lag, 0)]
                    for lag in range(lookback - 1, -1, -1)
                }
            )

        def fit(training):
            alpha, _, _, lookback, kernel = training
            arrays = {
                "x_train": lag_frame(x_train[:, 0], int(lookback)),
                "y_train": train_y,
                "x_test": lag_frame(x_test[:, 0], int(lookback)),
            }
            return (
                DLinearRegressor()
                .train(arrays, alpha=alpha, kernel_size=int(kernel))
                .predict(arrays)["_preds"]
            )

        return fit
    if model == "lightgbm":
        from lightgbm import LGBMRegressor

        def fit(training):
            rate, trees, leaves, bins, l2, min_leaf = training
            estimator = LGBMRegressor(
                objective="regression",
                learning_rate=rate,
                n_estimators=int(trees),
                num_leaves=int(leaves),
                max_bin=int(bins),
                reg_lambda=l2,
                min_child_samples=int(min_leaf),
                n_jobs=1,
                deterministic=True,
                force_col_wise=True,
                random_state=seed & (2**31 - 1),
                verbosity=-1,
            )
            return estimator.fit(x_train, train_y).booster_.predict(
                x_test, num_threads=1
            )

        return fit

    sys.path.insert(0, str(ROOT / "benchmarks/single-model"))
    from tide import load_reference, tf

    model_type = load_reference()
    tf.config.experimental.enable_op_determinism()
    tf.config.run_functions_eagerly(True)
    x_train = x_train[:, 0].astype(np.float32)
    x_test = x_test[:, 0].astype(np.float32)
    target = tf.constant(train_y.astype(np.float32)[:, None])

    def inputs(values, lookback):
        window = np.lib.stride_tricks.sliding_window_view(
            np.pad(values, (lookback - 1, 0), mode="edge"), lookback
        )
        return (
            (
                tf.constant(window),
                tf.zeros((1, lookback)),
                tf.zeros((0, lookback), tf.int32),
            ),
            (tf.zeros((1, 1)), tf.zeros((0, 1), tf.int32)),
            tf.zeros(len(values), tf.int32),
        )

    def fit(training):
        lookback, width, rate, epochs = training
        lookback, width, epochs = int(lookback), int(width), int(epochs)
        tf.keras.backend.clear_session(free_memory=False)
        tf.keras.utils.set_random_seed(seed & (2**31 - 1))
        model = model_type(
            {
                "hidden_dims": [width, width],
                "decoder_output_dim": 4,
                "time_encoder_dims": [4, 2],
                "final_decoder_hidden": 4,
            },
            pred_len=1,
            cat_sizes=[],
            num_ts=1,
        )
        past, future, ids = inputs(x_train, lookback)
        optimizer = tf.keras.optimizers.Adam(rate)
        for _ in range(epochs):
            model.train_step(past, future, target, ids, optimizer)
        return model(inputs(x_test, lookback), training=False).numpy().reshape(-1)

    return fit


def run(model, work, size, lib_path):
    x_train, train_y = matrix((work / "train.bin").read_bytes())
    x_test, test_y = matrix((work / "test.bin").read_bytes())
    price = prices((work / "price.bin").read_bytes())
    seed = read_json(work / "config.json")["seed"]
    lib = library(lib_path)
    scorer = lib.sweep_scorer(
        price[0].ctypes.data_as(DOUBLE_PTR),
        price[1].ctypes.data_as(DOUBLE_PTR),
        test_y.ctypes.data_as(DOUBLE_PTR),
        len(test_y),
    )
    truth = {}
    finished = (c.c_double * 15)()
    with threadpool_limits(limits=1):
        start = perf_counter()
        fit = fit_model(model, x_train, train_y, x_test, seed)
        setup = perf_counter() - start
        loop_start = perf_counter()
        trials = plan(lib, work / "config.json", size)
        for index, trial in enumerate(trials):
            key = numbers(trial, model)
            prediction = np.ascontiguousarray(fit(key), dtype=np.float64)
            if prediction.shape != test_y.shape or not np.isfinite(prediction).all():
                raise ValueError(
                    f"Invalid upstream predictions at {model} trial {index}"
                )
            lib.sweep_record(
                scorer, prediction.ctypes.data_as(DOUBLE_PTR), c.byref(trial), index
            )
            truth.setdefault(key, prediction)
        lib.sweep_finish(scorer, finished)
        loop = perf_counter() - loop_start
    fields = (
        "id",
        "mae",
        "signal_rate_pct",
        "backtest_total_return_net_pct",
        "backtest_sharpe_per_bar",
    )
    winners = {}
    for index, label in enumerate(("best_return", "best_sharpe", "best_mae")):
        values = finished[5 * index : 5 * index + 5]
        winners[label] = (
            None
            if math.isnan(values[0])
            else {
                key: value if math.isfinite(value) else None
                for key, value in zip(fields, values)
            }
        )
    packages = {
        "ridge": ("vaquum-limen", "scikit-learn", "numpy"),
        "dlinear": ("vaquum-limen", "polars", "scipy", "numpy"),
        "lightgbm": ("lightgbm", "numpy"),
        "tide": ("tensorflow", "keras", "numpy"),
    }
    return {
        "seconds": setup + loop,
        "setup_seconds": setup,
        "trial_seconds": loop,
        "rows": size,
        "winners": winners,
        "versions": {key: version(key) for key in packages[model]},
        "predictions": [[list(key), value.tolist()] for key, value in truth.items()],
    }


if __name__ == "__main__":
    model, work, size, lib_path = sys.argv[1:]
    print(
        json.dumps(run(model, Path(work), int(size), Path(lib_path)), allow_nan=False)
    )
