# Qwen3-0.6B LoRA on Pannu0

## Model and task

The experiment starts from unquantized bfloat16
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)
at revision `c1899de289a04d12100db370d81485cdf75e47ca`, including the tokenizer
and chat template. The generated arithmetic dataset has 64 training, eight
validation, and eight test examples with disjoint operand triples from a
single template. The target includes a short calculation and an `Answer: N`
line. The benchmark measures exact-answer completion on this template.

## Canonical reference fit

MLX 0.32.5 CUDA 13 and MLX LM 0.32.2 ran on an RTX 5080 under Ubuntu 24.04
WSL2 on Pannu0. The pilot uses stock MLX LM completion-masked LoRA with
batch size two, a 128-token sequence ceiling, Adam at 0.0001, rank 8,
scale 20, zero dropout, and seed 20261125. Only the final two transformer
blocks carry adapters. Base and tuned generation use the pinned chat
template with thinking disabled, a 64-token generation limit, temperature 0.7, top-p 0.8, and top-k 20.

The base scored 0/8 with eight extraction failures. Both independent
16-step fits scored 8/8 with zero failures. Training loss fell from 1.58634
over steps 1-4 to 0.04543 over steps 13-16. Both fits produced bit-identical
adapter tensors. The first (cold) fit took 107.909 seconds at 1.622 GB MLX peak
memory. The in-process repeat took 0.328 seconds at 1.534 GB. The base
outputs reached the token limit without a parseable answer, so the
accuracy delta includes gains in formatting and response length.

## Matched sweeps

Each setting uses checkpoint budgets of 8 and 16 updates. The fresh
control fits both budgets independently. The checkpoint-aware control
fits once per setting and grades upstream adapter saves at both budgets.
The cache arm uses the checkpoint-aware plan with the same batches,
optimizer trajectories, saved adapters, and full-model generation.
Sweep scoring uses upstream batched generation on the eight validation prompts, with the same decoding settings and the same batch seed in every trial. Winner ties go to the first trial in the ordered grid.

| Trials | Fresh fits | Checkpoint-aware fits | Cache fits |
| ---: | ---: | ---: | ---: |
| 10 | 10 | 5 | 5 |
| 100 | 100 | 50 | 50 |

At 10 trials, the upstream checkpoint-aware runs took 8.645 and 8.656
seconds. Cache runs took 8.495 and 8.496 seconds. The winner was rank 8,
learning rate 0.0001, and 16 updates with 7/8 validation answers. At 100
trials, the grid crosses five ranks (2, 4, 8, 16, 32) with ten learning
rates between 0.00004 and 0.0004. Its winner was rank 8, learning rate
0.00008, and 16 updates with 8/8 validation answers.

| 100-trial arm | Repeat 1 | Repeat 2 | Repeat 3 |
| --- | ---: | ---: | ---: |
| Fresh fits | 67.445 s | 62.525 s | 62.947 s |
| Upstream checkpoint reuse | 51.401 s | 51.846 s | 52.626 s |
| Frozen-prefix cache | 48.512 s | 48.733 s | 48.607 s |

The cache reduced end-to-end time by 5.6%-7.6% against upstream checkpoint
reuse, including preprocessing, model load, fits, checkpoint I/O,
generation, and scoring. The cache retained 18 activation tensors
totaling 7.08 MB. It reduced the reported fit phase from 17.17-17.53 to
14.07-14.19 seconds at 100 trials. Across both grid sizes, every adapter
tensor had maximum absolute difference zero, against a declared
`atol=1e-5` and `rtol=1e-4`. All per-example grades and winner indices
agreed exactly across arms and repeats.

Two further 100-trial pairs exercised the proposed trainer hook directly.
The warm pair took 48.468 seconds with the hook and 52.427 seconds with
upstream checkpoint reuse, a 7.6% reduction with exact parity. In the
cold pair the checkpoint arm ran first and took 56.839 seconds, compared
with 48.848 seconds for the hook. At 10 trials the hook's two timing
differences ranged from 0.012 seconds slower to 0.106 seconds faster.
The repeatable end-to-end gain was established at the 100-trial size.

## Optional trainer hook

Use [`FrozenPrefixCache`](../../scripts/qwen3_lora_cache.py) with the
upstream trainer arguments after configuring LoRA on the final two
blocks. Call `cache = FrozenPrefixCache(model)` once, then use
`cache.train(model=model, args=training_args, optimizer=optimizer,
train_dataset=train_data, val_dataset=valid_data)` in place of
`mlx_lm.tuner.trainer.train`. For each new trajectory, restore the frozen
base modules, create new adapters and an optimizer, and reset the PRNGs.
The hook uses the upstream trainer, masking, and checkpoint writer.

The cache belongs to one tied-embedding bfloat16 Qwen3 model on one device.
Use a new cache after changing base weights, dtype, or device. Entries
retain evaluated activations in memory for each distinct padded token
batch with matching loss offsets and lengths. This setup is intended for
small, repeated batches with the upstream zero-offset causal attention.
Run `python -B tests/check_qwen3_lora_cache.py` with the pinned MLX
dependencies to check adapter and fixed-input logit parity on a small
in-memory fixture.

[Upstream LoRA guide](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/LORA.md)
