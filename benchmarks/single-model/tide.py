"""Run a single Google TiDE fit in eager or XLA mode."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf


def read_split(path):
    values = np.fromfile(path, dtype="<f8", offset=24).astype(np.float32)
    return np.split(values, 2)


def load_reference():
    location = os.environ["TERASWEEP_TIDE_SOURCE"]
    spec = importlib.util.spec_from_file_location("single_run_google_tide", location)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TideModel


def run(mode, train, test, settings):
    if mode not in ("eager", "xla"):
        raise ValueError(f"Unknown TiDE execution mode {mode}")
    model_type = load_reference()
    train_x, labels = read_split(train)
    test_x, _ = read_split(test)
    tf.config.experimental.enable_op_determinism()
    eager = mode == "eager"
    tf.config.run_functions_eagerly(eager)
    lookback = settings["lookback"]

    def inputs(values):
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

    start = perf_counter()
    tf.keras.utils.set_random_seed(settings["seed"])
    width = settings["hidden_width"]
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
    optimizer = tf.keras.optimizers.Adam(settings["learning_rate"])
    train_inputs = inputs(train_x)
    y = tf.constant(labels[:, None])
    if eager:
        train_step = model.train_step
        predict = model
    else:
        # Trace a whole training step so Keras variables stay visible to its tape.
        model.call = model.call.python_function
        model._assemble_feats = model._assemble_feats.python_function
        train_step = tf.function(model.train_step.python_function, jit_compile=True)
        predict = tf.function(model, jit_compile=True)
    for _ in range(settings["epochs"]):
        train_step(train_inputs[0], train_inputs[1], y, train_inputs[2], optimizer)
    predictions = predict(inputs(test_x), training=False).numpy().reshape(-1)
    elapsed = perf_counter() - start
    return {
        "predictions": predictions.astype(float).tolist(),
        "fit_predict_seconds": elapsed,
    }


if __name__ == "__main__":
    mode, train, test, config = sys.argv[1:]
    print(json.dumps(run(mode, Path(train), Path(test), json.loads(config))))
