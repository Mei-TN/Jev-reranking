# Quick start

[Repository home](../README.md) · [简体中文](QUICKSTART.zh-CN.md)

Start with a CPU-only configuration preview. Training and model inference require Linux, a CUDA GPU, and separately prepared model and data assets.

## 1. Install and preview a configuration

Use Python 3.12 or newer. These commands work in Bash or PowerShell:

```bash
git clone https://github.com/Mei-TN/Jev-reranking.git
cd Jev-reranking
python -m pip install -e .
jev verify
jev list --study e458
jev plan --site configs/site.example.json --study e458 --arm POOL_NOFFN --fold F2 --seed 44
```

If you already cloned the repository, run the installation commands from its root. `verify` checks the source inventory, `list` shows configurations, and `plan` prints the resolved training command. This preview does not load a model or require the example paths to exist. See [CONTRIBUTING.md](../CONTRIBUTING.md) for development tests.

## 2. Prepare the environment and assets

| Material | Used for | Setup |
| --- | --- | --- |
| NeoHorse-1-4B weights and tokenizer | Training and inference | Use the complete model revision in [DATA.md](DATA.md) |
| Model Python environment | Training, export, and inference | Follow [ENVIRONMENT.md](ENVIRONMENT.md) |
| Exact F2/F3 training exports | Paper training | `F2.train.jsonl` / `F3.train.jsonl`, with the hashes in [DATA.md](DATA.md); no public download is currently provided |
| Benchmark data and evaluators | JevBench, Kev, and MC11 | See [EVALUATION.md](EVALUATION.md); the exact external evaluation fork has no public download in this release |
| Numerical supplement | Recomputing stored paper scores | See [DATA.md](DATA.md); no public download is currently provided |

The last two rows are for paper evaluation and score reconstruction. The bundled `examples/` files illustrate the request format; running inference on them still needs a trained checkpoint or exported model.

Create the site configuration once:

```bash
jev init --site configs/site.local.json
```

`init` will not overwrite an existing file. Edit these fields in `configs/site.local.json`:

| Field | What to set |
| --- | --- |
| `base_model` | Directory containing the complete model and tokenizer |
| `data_root` | Directory containing the training exports |
| `data_files` | F2/F3 filenames under `data_root` |
| `python.training`, `python.serving` | Executables for the configured model environment |
| `hardware.gpu_name_contains` | Your actual GPU model; replace the example `H20` if needed |
| `work_root` | Output root; `init` defaults to the repository's `work/` directory |

Relative paths resolve from the site file. The example layout places model files in `assets/NeoHorse-1-4B-hf/` and fold files in `assets/training/`. Runs under the same `work_root` locate their reference initialization automatically; use `reference_outputs` for a reference elsewhere.

Check readiness on the Linux training host:

```bash
jev doctor --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --imports
```

`READY` means the checks passed. Until paths, assets, and dependencies are prepared, `INCOMPLETE` with exit code 2 is expected. Fix the listed `problems` before training.

## 3. Run the final model: choose one route

The examples use F2 and seed 44. Both routes first train `e1e2 / LL_A`, the all-module pointer reference, to save the shared initialization. The final `e458 / POOL_NOFFN` run trains mean-span pooling with attention/DeltaNet adapters from that initialization, not from the reference's trained weights.

Choose the manual route or the pipeline route for the final run. Run the commands from the repository root on Linux; the paths below assume the default `work_root`.

### A. Train, export, predict, and score separately

```bash
jev train --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --gpu 0
jev train --site configs/site.local.json --study e458 --arm POOL_NOFFN --fold F2 --seed 44 --gpu 0
```

The default final checkpoint is `work/runs/e458/E458_F2s44_POOL_NOFFN/output`. Use the path printed by `train` if you changed `work_root`. If these runs are already complete, continue with export instead of repeating training.

```bash
jev export --site configs/site.local.json --tree e458 --gpu 0 --checkpoint work/runs/e458/E458_F2s44_POOL_NOFFN/output --output work/bundles/jev-noffn-span --verification-examples examples/export_requests.json
jev predict --site configs/site.local.json --bundle work/bundles/jev-noffn-span --gpu 0 --requests examples/requests.jsonl --output work/predictions.jsonl
jev evaluate --requests examples/requests.jsonl --labels examples/labels.jsonl --predictions work/predictions.jsonl --model jev-noffn-span --output work/evaluation
```

Export compares the checkpoint with the reloaded bundle. The export directory name sets the model name, here `jev-noffn-span`; if you change it, use the new name with `evaluate --model`. Predictions go to `work/predictions.jsonl`, and scores to `work/evaluation/`. Use new output paths for another run.

### B. Let the pipeline manage the final run

Use this route before separately training `POOL_NOFFN`:

```bash
jev train --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --gpu 0
jev pipeline --site configs/site.local.json --study e458 --arm POOL_NOFFN --fold F2 --seed 44 --gpu 0 --verification-examples examples/export_requests.json --requests examples/requests.jsonl --labels examples/labels.jsonl
```

If the reference is already complete under this site, skip the first command. The pipeline runs final-model training, export, prediction, and scoring, then reports the output paths. `--resume` reuses verified stages from a previous pipeline invocation; it cannot take over a final-model run created by a separate `jev train` command.

Both routes use small format examples for inference, not the paper benchmark populations. Replace the requests and labels with your own files, or follow [EVALUATION.md](EVALUATION.md) for the paper benchmarks.

## 4. Optional ablations

The final model requires only the reference and `POOL_NOFFN` runs. Other comparisons are:

| Study / arm | Model | Reference initialization |
| --- | --- | --- |
| `e458 / POINTER_NOFFN` | Selective adaptation + option-ending pointer | Matching `e1e2 / LL_A` |
| `e458 / POOL_ALL` | All-module adaptation + mean-span pooling | Matching `e1e2 / LL_A` |
| `e11 / POINTER_2EP` | Two-epoch all-module pointer | Matching `e1e2 / LL_A` |
| `e11 / POOL_2EP` | Two-epoch all-module pooling | Matching `e458 / POOL_ALL` |

Use `jev list --study e458` or `jev list --study e11` to select the arm, fold, and seed. Two-epoch controls restart from saved initialization with a two-epoch schedule. The [paper-to-code map](DESIGN_MAP.md) lists other comparisons; [EXTENDING.md](EXTENDING.md) explains custom recipes.
