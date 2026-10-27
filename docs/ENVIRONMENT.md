# Runtime environments

Use Python 3.12+ for the JEV command-line package. Its installation is separate from the native model and evaluator dependencies.

## Model role

Training, checkpoint loading, bundle export and serving require Linux, a CUDA GPU and the recorded model stack. [The environment requirements](../requirements/README.md) describe the retained version records and automatic installation entry point. For that isolated-prefix installation, copy its `model-python` launcher as instructed and select that executable in `python.training` and `python.serving`. It restores the prefix's library paths when a child process supplies its own environment.

Set `hardware.gpu_name_contains` to the GPU actually used. Runs retain their actual hardware identity. The original paper used multiple hardware/runtime contexts; changing them can change predictions even for a fixed seed or checkpoint.

## Evaluation roles

The complete native benchmark separates four roles:

| Role | Purpose |
| --- | --- |
| Controller | Process coordination, validation and cleanup |
| Model service | Loads the exported JEV model on CUDA |
| OpenCompass | JevBench/Kev loaders and scoring on CPU |
| RunWeaver/OpenJev | MC11 population construction and scoring on CPU |

Use `configs/benchmark-site.example.json` to bind their interpreters, source checkouts, frozen caches and local paths. The controller needs Linux `os.pidfd_open` and `signal.pidfd_send_signal`; check the actual interpreter, since some standalone Python builds omit them. It may differ from the model interpreter.

OpenCompass and RunWeaver/OpenJev are external dependencies. Their version/source requirements and applicable terms are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The repository does not bundle those complete libraries or grant rights to their datasets.

The small JEV-specific OpenCompass dataset/scorer overlay and compatibility patch are under `third_party/opencompass_jev`. Its guide identifies the obtainable official base revision, installation checks and validation scope.

## Validation before training

```bash
jev doctor --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --imports
```

This checks the selected source, pinned inputs and distribution/import contract. Review the reported problems before launching a model. Dependency metadata, CUDA availability and successful imports establish different parts of environment compatibility; the recorded Torch/Triton metadata conflict is described with the version records.
