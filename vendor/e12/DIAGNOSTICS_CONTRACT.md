# Training diagnostics

Enable `--supplement_diagnostics 1` to save initialized and final head and LoRA tensors, per-update optimizer statistics, and fixed-sample predictions. Initial tensors are captured before optimizer construction; final tensors are captured after the last completed update and before checkpoint export.

Each update records the pre-clipping global L2 gradient norm, whether it exceeds 1.0, the learning rate and beta1 of each optimizer group, and the active adapter parameter count. Residual-head runs also record base and residual option statistics. The clipping fraction is the proportion of updates whose measured norm exceeds 1.0.

The fixed diagnostic sample contains the first 32 questions of each question type in training-file order, or all available questions when fewer than 32 exist. The selected records and their hashes are saved. Initial and final diagnostic passes use the training autocast policy with gradients disabled and modules in evaluation mode, then restore the previous module modes. RNG state is checked for changes.

Choice questions with at least two options are also evaluated with reversed option order. Labels, keys, soft targets, and validity masks are remapped consistently. The records include the new-to-original option mapping, logits, query/key projections, and option-content token lengths.

The observer detaches tensors from the existing forward pass without changing gradients or optimizer state. Fixed-sample results describe training examples; benchmark evaluation uses the separate evaluation pipeline. Residual/base RMS ratios and relative parameter updates are undefined when their denominators are zero.

The CPU tests cover output and gradient parity, RNG preservation, zero-initialized residual behavior, module-mode restoration, optimizer-state preservation, and effective adapter norms.
