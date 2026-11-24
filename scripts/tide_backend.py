"""Fit the pinned Google Research TiDE on one-step return windows."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


import numpy as np
import tensorflow as tf

SOURCE_SHA256 = "ad8fd8d498809deeed9c847a169cada42134cbd0351b03e750706132a5d151c7"


def load_model(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError(
            "TiDE model source differs from the pinned Google Research version"
        )
    spec = importlib.util.spec_from_file_location("terasweep_upstream_tide", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TideModel


def inputs(values, lookback):
    rows = np.arange(values.size)[:, None]
    lags = np.arange(lookback - 1, -1, -1)
    windows = values[np.maximum(rows - lags, 0)]
    # The canonical no-covariate case uses a constant zero feature channel.
    return (
        (
            tf.constant(windows),
            tf.zeros((1, lookback)),
            tf.zeros((0, lookback), tf.int32),
        ),
        (tf.zeros((1, 1)), tf.zeros((0, 1), tf.int32)),
        tf.zeros(values.size, tf.int32),
    )


def main():
    tf.config.experimental.enable_op_determinism()
    # The upstream methods predate Keras 3 tracing. Eager execution uses their
    # original loss, gradients and optimizer on the pinned Apple Silicon stack.
    tf.config.run_functions_eagerly(True)
    source = Path(os.environ["TERASWEEP_TIDE_SOURCE"])
    model_type = load_model(source)
    stream = sys.stdin.buffer
    seed, train_rows, test_rows, configs = json.loads(stream.readline())
    data = np.frombuffer(stream.read(), dtype="<f8").astype(np.float32)
    if data.size != train_rows * 2 + test_rows or not np.isfinite(data).all():
        raise ValueError("Invalid TiDE return series or labels")
    train_x, train_y, test_x = np.split(data, (train_rows, train_rows * 2))
    output = sys.stdout.buffer
    for lookback, width, rate, epochs in configs:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
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
        past, future, ids = inputs(train_x, lookback)
        target = tf.constant(train_y[:, None])
        optimizer = tf.keras.optimizers.Adam(rate)
        for _ in range(epochs):
            model.train_step(past, future, target, ids, optimizer)
        predictions = model(inputs(test_x, lookback), training=False).numpy()
        if predictions.shape != (test_rows, 1) or not np.isfinite(predictions).all():
            raise ValueError("TiDE returned invalid predictions")
        output.write(predictions.astype("<f8").tobytes())


if __name__ == "__main__":
    main()
