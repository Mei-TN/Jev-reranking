# Synchronized pointer / pointer repeat control

`paired_trace_repeat.py` applies the same four-update protocol as `../paired_trace.py` to two independently constructed pointer/end models. Seed, F2 shuffle and first 32 records, augmentation, synchronized CPU/CUDA generators, batch 1, accumulation 8, full 1,932-update OneCycle horizon, peak rates 2.5e-5, AdamW, clipping, FP32 weights and BF16 autocast remain fixed. The source requires exactly one visible H20.

Both scripts take `--base`, `--data`, `--output`, `--expected-data-sha256`, and `--expected-base-config-sha256`. Supply the same frozen inputs and use a separate fresh output directory for each comparison. Put the `r3_trace` root on PYTHONPATH for the preserved `kev` imports.
