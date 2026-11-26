"""Fit and predict with the official TiDE API on one observation series."""

import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf


def main():
    source, inputs_path, settings_json, output_path = sys.argv[1:]
    spec = importlib.util.spec_from_file_location("google_tide_reference", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    settings = json.loads(settings_json)
    tf.config.experimental.enable_op_determinism()
    tf.config.run_functions_eagerly(True)

    with np.load(inputs_path) as data:
        train_x, train_y, test_x = data["train"], data["labels"], data["test"]
    lookback = settings["lookback"]

    def inputs(values):
        history = np.pad(values, (lookback - 1, 0), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(history, lookback)
        return (
            (
                tf.convert_to_tensor(windows),
                tf.zeros((1, lookback)),
                tf.zeros((0, lookback), tf.int32),
            ),
            (tf.zeros((1, 1)), tf.zeros((0, 1), tf.int32)),
            tf.zeros(windows.shape[0], tf.int32),
        )

    started = perf_counter()
    tf.keras.utils.set_random_seed(settings["seed"])
    width = settings["hidden_width"]
    model = module.TideModel(
        dict(
            hidden_dims=[width, width],
            decoder_output_dim=4,
            time_encoder_dims=[4, 2],
            final_decoder_hidden=4,
        ),
        pred_len=1,
        cat_sizes=[],
        num_ts=1,
    )
    train, val, ids = inputs(train_x)
    optimizer = tf.keras.optimizers.Adam(settings["learning_rate"])
    labels = tf.convert_to_tensor(train_y[:, None])
    for _ in range(settings["epochs"]):
        model.train_step(train, val, labels, ids, optimizer)
    result = model(inputs(test_x), training=False).numpy()
    elapsed = perf_counter() - started
    Path(output_path).write_bytes(np.asarray([elapsed], dtype="<f8").tobytes()
                                  + result.astype("<f8").tobytes())


if __name__ == "__main__":
    main()
