# Synchronized pointer / frozen-zero residual trajectory

The driver `paired_trace.py` compares a pointer/end model with a frozen-zero residual head over the first 32 full-F2 source records after the native seed-44 epoch shuffle. It uses four optimizer updates, batch 1, accumulation 8, peak head and adapter rates 2.5e-5, and the full 1,932-update OneCycle horizon. LoRA uses rank 64 over all target modules with dropout 0.05; model weights are FP32 and autocast is BF16. The source requires exactly one visible H20.

Both arms restore the same captured CPU and selected-CUDA RNG state before each forward/backward. The driver checks shared initial tensors, logits, losses, accumulated gradients, clipping, optimizer moments and parameter updates; the residual branch remains frozen with a zero final layer. This four-update diagnostic addresses the synchronized short trajectory.

Arguments are `--base`, `--data`, `--output`, `--expected-data-sha256`, and `--expected-base-config-sha256`. Use the full frozen F2 fold, a fresh output path, and the `r3_trace` import root. Data and base identities are checked before model construction.
