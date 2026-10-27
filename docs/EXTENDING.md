# Extending an experiment

Choose the source corresponding to the design you want to modify. The [paper map](DESIGN_MAP.md) identifies the study-specific versions and initialization requirements.

```bash
jev fork --tree e12 --output extensions/my_variant
```

Create a recipe based on `configs/custom.example.json`. Give it a new `id`, set `source_root` to the fork relative to the recipe file, and set `trainer_args` explicitly. The template fields `{base_model}`, `{data_root}` and `{output}` are bound from the site. The output argument must use `{output}`.

`trainer_module` defaults to `kev.train`. A study-specific entry point, such as `kev.train_lowrate_noffn`, can be selected when that module exists inside the chosen source tree. The paper configurations retain their actual training module.

```bash
jev plan --site configs/site.local.json --recipe configs/my_variant.json
jev doctor --site configs/site.local.json --recipe configs/my_variant.json --imports
jev train --site configs/site.local.json --recipe configs/my_variant.json --gpu 0
```

Add your input file hashes to `input_pins` and expected record/update counts to `expected`. Source changes and recipe identity are recorded with the new run.

## Changing a head or readout

Update the following interfaces together:

1. `kev/heads.py`: configuration validation, factory registration, initialization and forward computation.
2. The encoder/model interface: option spans, decision positions and candidate masks used by the readout.
3. The checkpoint head configuration and state dictionary.
4. `runtime/bundle_contract.py` and the corresponding loader/export path.
5. A fixed-input comparison between checkpoint inference and an exported, independently reloaded bundle.

Keep the named-option probability API and candidate order explicit. A new readout may consume different hidden states while sharing the same projection tensors; preserve that distinction in the configuration.

## Changing a training objective

The paper's native loss accepts soft distributions. Modify the target processing and loss together, then check normalization, question weighting within a record, and the treatment of binary and ordered questions. Save the new objective under its own experiment ID.

For an implementation without `runtime/export_bundle.py`, use its native checkpoint loader or provide a matching custom exporter. The exporter must match the checkpoint's head and runtime interfaces.

## Adding an evaluation endpoint

Define the source revision and split, candidate construction and order, demonstration selection, request format, semantic probability key, invalid-output behavior, tie rule and aggregation. Keep labels outside inference requests. The generic JSONL evaluator can be used for custom datasets; paper benchmarks retain their own scoring definitions.
