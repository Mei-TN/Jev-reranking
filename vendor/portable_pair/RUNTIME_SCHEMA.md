# Runtime manifest schema v2

`schema` is `portable-existing-runtime-binding-v2`. `original_asset_manifest` binds the unchanged historical asset manifest. `replaced_assets` must equal exactly the eight runtime marker entries enumerated in `runtime_binding.py`; scientific assets cannot be removed by this map.

`roles` contains exactly `training`, `serving`, `opencompass`, `openjev`, and `runweaver`. Each role has an absolute installation `root`, a pinned `python` path, `files` entries with `path`, contained `resolved_path`, `bytes`, and `sha256`, installed `expected_distributions`, and an explicit `environment`. Allowed persistent fields are `LD_LIBRARY_PATH`, `PYTHONPATH`, `PATH`, `LANG`, `SDL_AUDIODRIVER`, and `SDL_VIDEODRIVER`. Library and overlay directories must be contained in the installation root. Python executable and each declared file are checked against actual resolved paths and bytes before their stage runs.

The public template substitutes `{<role>_runtime_root}` placeholders. `configure_runtime.py` captures permitted location metadata once and stores all actual identities in a new manifest; later validation has no relocation exception. The field `JEV_VERIFIED_RUNTIME_OVERLAY` is derived by the wrapper from the verified role's `PYTHONPATH`, never from an inherited environment value. The exporter passes only this overlay alongside its new bundled source to the verification child.

The manifest validates installed runtime entry/metadata identities, not every package binary, system library or transitive dependency. The exact scientific source/data/base/tokenizer/population manifests are separate and remain enforced.
