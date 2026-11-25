"""Optional frozen-prefix reuse for the upstream MLX LM Qwen3 LoRA trainer."""

import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model, create_attention_mask
from mlx_lm.tuner import trainer


class FrozenPrefixCache:
    """Bind to one Qwen3 model with immutable base weights on one device.

    Reuse the instance across LoRA trajectories on the last two blocks.
    Discard it before editing base weights, moving devices, or changing dtype.
    The upstream batch iterator supplies zero-position, causal token batches.
    """

    def __init__(self, model):
        if not isinstance(model, Model) or not model.args.tie_word_embeddings:
            raise ValueError("Expected an upstream Qwen3 model with tied embeddings")
        if len(model.layers) <= 2 or model.model.embed_tokens.weight.dtype != mx.bfloat16:
            raise ValueError("Expected unquantized bfloat16 Qwen3 with a frozen prefix")
        self.model = model
        self.device = mx.default_device()
        self.entries = {}
        self.build_s = 0.0
        self.lookup_s = 0.0
        self.bytes = 0
        self.hits = 0

    def train(self, *, model, args, **options):
        """Use the same arguments as the upstream trainer for single-device LoRA."""
        if model is not self.model or mx.default_device() != self.device:
            raise ValueError("The cache belongs to a different model or device")
        if args.grad_checkpoint or mx.distributed.init().size() != 1:
            raise ValueError("Cache reuse requires one worker and no gradient checkpointing")
        trainable = tree_flatten(model.trainable_parameters())
        prefix_end = len(model.layers) - 2
        for name, _ in trainable:
            parts = name.split(".")
            if (parts[:2] != ["model", "layers"]
                    or int(parts[2]) < prefix_end
                    or parts[-1] not in ("lora_a", "lora_b")):
                raise ValueError(f"Unsupported trainable parameter: {name}")
        if not trainable:
            raise ValueError("Configure LoRA layers before training")

        return trainer.train(model=model, args=args, **options,
                             iterate_batches=self._batches, loss=self._loss)

    def _batches(self, *args, **options):
        for tokens, lengths in trainer.iterate_batches(*args, **options):
            start = time.perf_counter()
            key = (tuple(tokens.shape), str(tokens.dtype),
                   np.asarray(tokens).tobytes(), np.asarray(lengths).tobytes())
            hidden = self.entries.get(key)
            self.lookup_s += time.perf_counter() - start
            if hidden is None:
                start = time.perf_counter()
                hidden = self.model.model.embed_tokens(tokens[:, :-1])
                mask = create_attention_mask(hidden, None)
                for layer in self.model.layers[:-2]:
                    hidden = layer(hidden, mask, None)
                mx.eval(hidden)
                self.entries[key] = hidden
                self.build_s += time.perf_counter() - start
                self.bytes += hidden.nbytes
            else:
                self.hits += 1
            yield tokens, lengths, hidden

    @staticmethod
    def _loss(model, tokens, lengths, hidden):
        def forward(_):
            state = hidden
            mask = create_attention_mask(state, None)
            for layer in model.layers[-2:]:
                state = layer(state, mask, None)
            state = model.model.norm(state)
            return model.model.embed_tokens.as_linear(state)

        return trainer.default_loss(forward, tokens, lengths)
