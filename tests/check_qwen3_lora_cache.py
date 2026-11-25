"""Optional CUDA parity check with an in-memory Qwen3 fixture."""
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, remove_lora_layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qwen3_lora_cache import FrozenPrefixCache


def qwen3_fixture():
    mx.random.seed(19)
    model = Model(ModelArgs(
        model_type="qwen3", hidden_size=32, num_hidden_layers=3,
        intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, rms_norm_eps=1e-6, vocab_size=32,
        max_position_embeddings=128, tie_word_embeddings=True, rope_theta=10000,
    ))
    model.set_dtype(mx.bfloat16)
    return model


class FrozenPrefixTest(unittest.TestCase):
    def test_stock_trainer_parity_and_reuse(self):
        model = qwen3_fixture()
        mx.eval(model.parameters())
        cache = FrozenPrefixCache(model)
        dataset = [([1, 2, 3, 4, 5], 2), ([5, 4, 3, 2, 1], 2)]
        inputs = mx.array([[1, 2, 3, 4]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "adapters.safetensors"
            args = TrainingArgs(iters=2, batch_size=2, steps_per_save=1, adapter_file=path)
            for rank, lr in ((2, 1e-4), (4, 3e-4)):
                snapshots = []
                for fit in (train, cache.train):
                    np.random.seed(19)
                    mx.random.seed(19)
                    model.freeze()
                    linear_to_lora_layers(model, 2, {"rank": rank, "scale": 20.0, "dropout": 0.0})
                    fit(model=model, args=args, optimizer=optim.Adam(lr), train_dataset=dataset)
                    logits = model(inputs)
                    mx.eval(logits)
                    snapshots.append((mx.load(str(path)), logits))
                    remove_lora_layers(model)
                (stock, stock_logits), (cached, cached_logits) = snapshots
                self.assertEqual(stock.keys(), cached.keys())
                for key in stock:
                    self.assertTrue(mx.allclose(stock[key], cached[key], atol=1e-5, rtol=1e-4).item())
                self.assertTrue(mx.array_equal(stock_logits, cached_logits).item())
        self.assertEqual(len(cache.entries), 1)
        self.assertEqual(cache.hits, 3)

    def test_rejects_unfrozen_prefix(self):
        model = qwen3_fixture()
        cache = FrozenPrefixCache(model)
        with self.assertRaisesRegex(ValueError, "Unsupported trainable parameter"):
            cache.train(model=model, args=TrainingArgs())


if __name__ == "__main__":
    unittest.main()
