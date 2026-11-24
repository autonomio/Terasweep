"""Measure one fit and prediction per implementation on frozen model inputs."""

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from lightgbm import LGBMRegressor
from lightgbm.basic import _LIB as LIGHTGBM_LIBRARY
from limen.sfd.reference_architecture.dlinear_regressor import DLinearRegressor
from limen.sfd.reference_architecture.ridge_regressor import RidgeRegressor
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import uniform_filter1d
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_return_inputs import prepare  # noqa: E402
from verify_model_parity import lag_frame, read_matrix, retained_fixture  # noqa: E402

TOLERANCE = 1e-10
MODELS = {
    "ridge": {"alpha": 1.0, "fit_intercept": False, "solver_eps": 1e-12},
    "dlinear": {"alpha": 1.0, "lookback": 3, "kernel": 3, "solver_eps": 1e-14},
    "lightgbm": {
        "learning_rate": 0.1,
        "num_trees": 32,
        "num_leaves": 7,
        "max_bins": 32,
        "lambda_l2": 0.1,
        "min_data_in_leaf": 20,
        "seed": 20260514,
    },
    "tide": {
        "lookback": 6,
        "hidden_width": 8,
        "learning_rate": 0.001,
        "epochs": 2,
        "seed": 20260514,
    },
}


def measure(run):
    start = perf_counter()
    predictions = run()
    return {"predictions": predictions, "fit_predict_seconds": perf_counter() - start}


def capture(args, env, *, stdin=None, competing=False):
    completed = subprocess.run(
        args, input=stdin, env=env, stdout=subprocess.PIPE, text=True
    )
    if completed.returncode:
        if competing:
            return {"status": "unavailable", "exit_status": completed.returncode}
        completed.check_returncode()
    return json.loads(completed.stdout)


def estimator(settings):
    return LGBMRegressor(
        objective="regression",
        learning_rate=settings["learning_rate"],
        n_estimators=settings["num_trees"],
        num_leaves=settings["num_leaves"],
        max_bin=settings["max_bins"],
        reg_lambda=settings["lambda_l2"],
        min_child_samples=settings["min_data_in_leaf"],
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        random_state=settings["seed"],
        verbosity=-1,
    )


def lightgbm_run(train, test, settings, *, compiled):
    x_train, y = read_matrix(train)
    x_test, _ = read_matrix(test)
    if compiled:
        from lleaves import Model

        with TemporaryDirectory(prefix="terasweep-lleaves-") as directory:
            model_file = str(Path(directory) / "trees.txt")

            def compile_and_predict():
                model = estimator(settings).fit(x_train, y).booster_
                model.save_model(model_file)
                compiled_model = Model(model_file=model_file)
                compiled_model.compile(fcodemodel="small")
                return compiled_model.predict(x_test, n_jobs=1)

            return measure(compile_and_predict)
    return measure(
        lambda: (
            estimator(settings).fit(x_train, y).booster_.predict(x_test, num_threads=1)
        )
    )


def ridge_reference(train, test, settings):
    x_train, y = read_matrix(train)
    x_test, _ = read_matrix(test)
    data = {"x_train": x_train, "y_train": y, "x_test": x_test}
    return measure(
        lambda: (
            RidgeRegressor()
            .train(
                data,
                alpha=settings["alpha"],
                fit_intercept=settings["fit_intercept"],
                solver="cholesky",
            )
            .predict(data)["_preds"]
        )
    )


def dlinear_pair(train, test, settings):
    train_x, train_y = read_matrix(train)
    test_x, _ = read_matrix(test)
    lookback, kernel, alpha = (settings[key] for key in ("lookback", "kernel", "alpha"))

    def reference():
        data = {
            "x_train": lag_frame(train_x[:, 0], lookback),
            "y_train": train_y,
            "x_test": lag_frame(test_x[:, 0], lookback),
        }
        return (
            DLinearRegressor()
            .train(data, alpha=alpha, kernel_size=kernel)
            .predict(data)["_preds"]
        )

    def design(x):
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(x[:, 0], (lookback - 1, 0), mode="edge"), lookback
        )
        trend = uniform_filter1d(windows, size=kernel, axis=1, mode="nearest")
        return np.hstack((windows - trend, trend))

    def cholesky():
        features = design(train_x)
        mean = features.mean(axis=0)
        target = train_y.mean()
        centered = features - mean
        gram = centered.T @ centered
        gram.flat[:: len(gram) + 1] += alpha
        coefficients = cho_solve(cho_factor(gram), centered.T @ (train_y - target))
        return (design(test_x) - mean) @ coefficients + target

    return measure(reference), measure(cholesky)


def validate(output, reference):
    if output.get("status") == "unavailable":
        return output
    prediction = np.asarray(output["predictions"], dtype="<f8")
    expected = np.asarray(reference["predictions"], dtype="<f8")
    elapsed = float(output["fit_predict_seconds"])
    if (
        prediction.shape != expected.shape
        or not np.isfinite(prediction).all()
        or elapsed <= 0
    ):
        raise ValueError("Invalid prediction vector or timing")
    close = np.isclose(prediction, expected, rtol=TOLERANCE, atol=TOLERANCE)
    different_signs = int(np.count_nonzero((prediction > 0) != (expected > 0)))
    equivalent = bool(close.all() and different_signs == 0)
    return {
        "status": "equivalent" if equivalent else "different",
        "fit_predict_seconds": elapsed,
        "reference_over_elapsed": reference["fit_predict_seconds"] / elapsed
        if equivalent
        else None,
        "prediction_count": prediction.size,
        "first_predictions": prediction[:3].tolist(),
        "prediction_sha256_f64_le": hashlib.sha256(prediction.tobytes()).hexdigest(),
        "bit_identical": np.array_equal(
            prediction.view(np.uint64), expected.view(np.uint64)
        ),
        "max_abs_difference": float(np.max(np.abs(prediction - expected))),
        "signal_mismatches": different_signs,
    }


def cpu_name():
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return platform.processor() or platform.machine()


def run(binary, smartcore, model_names):
    environment = dict(
        os.environ,
        TF_USE_LEGACY_KERAS="0",
        TF_NUM_INTEROP_THREADS="1",
        TF_NUM_INTRAOP_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    environment.setdefault("TERASWEEP_LIGHTGBM_LIB", LIGHTGBM_LIBRARY._name)
    outcomes = {}
    with (
        TemporaryDirectory(prefix="terasweep-single-model-") as directory,
        threadpool_limits(limits=1),
    ):
        root = Path(directory)
        inputs = root / "return-inputs"
        if set(model_names) & {"dlinear", "tide"}:
            prepare(inputs)
        ridge_inputs = root / "ridge-inputs"
        if "ridge" in model_names:
            ridge_inputs.mkdir()
            retained_fixture(ridge_inputs)
        for name in model_names:
            source = {"ridge": ridge_inputs, "lightgbm": ROOT / "data"}.get(
                name, inputs
            )
            train, test = (source / f"{split}.bin" for split in ("train", "test"))
            settings = MODELS[name]
            payload = json.dumps(settings)
            if name == "ridge":
                reference = ridge_reference(train, test, settings)
                competing = capture(
                    [smartcore, train, test, str(settings["alpha"])],
                    environment,
                    competing=True,
                )
                names = ("Limen Ridge (scikit-learn Cholesky)", "SmartCore Cholesky")
            elif name == "dlinear":
                reference, competing = dlinear_pair(train, test, settings)
                names = ("Limen DLinear (NumPy SVD)", "SciPy LAPACK Cholesky")
            elif name == "lightgbm":
                reference = lightgbm_run(train, test, settings, compiled=False)
                competing = capture(
                    [
                        sys.executable,
                        "-B",
                        HERE / "compare.py",
                        "--lleaves-worker",
                        train,
                        test,
                        payload,
                    ],
                    environment,
                    competing=True,
                )
                names = ("Official LightGBM Python", "LightGBM + lleaves")
            else:
                command = [environment["TERASWEEP_TIDE_PYTHON"], "-B", HERE / "tide.py"]
                reference = capture(
                    command + ["eager", train, test, payload], environment
                )
                competing = capture(
                    command + ["xla", train, test, payload], environment, competing=True
                )
                names = (
                    "Google TiDE (TensorFlow eager)",
                    "Google TiDE (TensorFlow XLA)",
                )

            packet = json.dumps(
                {
                    "version": 1,
                    "model": {"kind": name, **settings},
                    "train": str(train),
                    "test": str(test),
                }
            )
            ours = capture(
                [
                    sys.executable,
                    "-B",
                    ROOT / "scripts/model_run.py",
                    "--binary",
                    binary,
                    "fit",
                    "-",
                ],
                environment,
                stdin=packet,
            )
            ours_result = validate(ours, reference)
            if ours_result["status"] != "equivalent" or (
                name in ("lightgbm", "tide") and not ours_result["bit_identical"]
            ):
                raise ValueError(
                    f"Terasweep {name} failed reference parity: {ours_result}"
                )
            outcomes[name] = {
                "model": ours["model"],
                "train": ours["train"],
                "test": ours["test"],
                "implementations": {
                    names[0]: validate(reference, reference),
                    names[1]: validate(competing, reference),
                    "Terasweep": ours_result,
                },
            }
    packages = {
        package: version(package)
        for package in (
            "numpy",
            "scipy",
            "scikit-learn",
            "polars",
            "lightgbm",
            "vaquum-limen",
            "lleaves",
            "llvmlite",
        )
    }
    return {
        "version": 1,
        "hardware": {
            "cpu": cpu_name(),
            "machine": platform.machine(),
            "os": platform.platform(),
        },
        "tools": {
            "rust": subprocess.check_output(
                [os.environ.get("TERASWEEP_BENCH_RUSTC", "rustc"), "--version"],
                text=True,
            ).strip(),
            "python": platform.python_version(),
            "packages": packages,
            "smartcore": "0.6.14",
            "tensorflow": "2.16.2",
            "keras": "3.15.1",
        },
        "timing": {
            "calls_per_implementation": 1,
            "threads": 1,
            "scope": "fit and predict from prepared arrays, including first graph or model compilation",
        },
        "rtol": TOLERANCE,
        "atol": TOLERANCE,
        "models": outcomes,
    }


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--lleaves-worker":
        train, test, payload = sys.argv[2:]
        with threadpool_limits(limits=1):
            result = lightgbm_run(
                Path(train), Path(test), json.loads(payload), compiled=True
            )
        result["predictions"] = result["predictions"].tolist()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("models", nargs="*", metavar="MODEL")
        parser.add_argument(
            "--binary", type=Path, default=ROOT / "target/terasweep-funnel"
        )
        parser.add_argument(
            "--smartcore",
            type=Path,
            default=ROOT
            / "runs/smartcore/target/release/terasweep-smartcore-comparison",
        )
        args = parser.parse_args()
        if any(model not in MODELS for model in args.models):
            parser.error(f"Choose models from {', '.join(MODELS)}")
        result = run(
            args.binary.resolve(), args.smartcore.resolve(), args.models or MODELS
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
