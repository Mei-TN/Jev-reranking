# JEV Reranking

Code for **[Improving JEV Entailment Reranking with Span Readouts and Selective Adaptation](https://arxiv.org/abs/2610.35654)**.

[简体中文](README.zh-CN.md) · [Quick start](docs/QUICKSTART.md) · [Paper-to-code map](docs/DESIGN_MAP.md)

This repository implements mean-span pooling and selective backbone adaptation for JEV entailment reranking. The combined model averages states inside each NLI option and trains LoRA adapters in attention and gated DeltaNet projections, leaving out FFN adapters.

## Try the tools on CPU

Use Python 3.12 or newer. These commands work in Bash or PowerShell:

```bash
git clone https://github.com/Mei-TN/Jev-reranking.git
cd Jev-reranking
python -m pip install -e .
jev verify
jev list --study e458
jev plan --site configs/site.example.json --study e458 --arm POOL_NOFFN --fold F2 --seed 44
```

This checks the sources and previews the final model's configuration. It needs no GPU, model weights, or datasets, and does not run the model. Development tests are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Prepare for training

Training and inference require Linux, a CUDA GPU, the [model environment](docs/ENVIRONMENT.md), NeoHorse-1-4B weights, and the matching training fold. The [quick start](docs/QUICKSTART.md) explains which paths to set.

Public download links are not yet provided for the complete training corpus, the exact external evaluation fork, or the numerical supplement. These are needed for paper training, full benchmark evaluation, and stored-score reconstruction, respectively. See [DATA.md](docs/DATA.md) for their identities and availability.

## Train the final design

After preparing the model environment and training data, create your site configuration:

```bash
jev init --site configs/site.local.json
```

Edit its model, data, Python, and GPU settings using the [quick start](docs/QUICKSTART.md). Then run:

```bash
jev doctor --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --imports
jev train --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --gpu 0
jev train --site configs/site.local.json --study e458 --arm POOL_NOFFN --fold F2 --seed 44 --gpu 0
```

`LL_A` is the all-module pointer reference. `POOL_NOFFN` is the final selective-adaptation and mean-span model; it starts from the reference's saved initialization, not its trained weights. Its default output is `work/runs/e458/E458_F2s44_POOL_NOFFN/output`.

The [quick start](docs/QUICKSTART.md) gives two alternative routes: separate training/export/prediction/scoring commands, or a pipeline that manages those stages. Optional ablations are listed separately. `doctor` returns `INCOMPLETE` with exit code 2 while required paths, assets, or dependencies are missing.

## More information

- [Quick start](docs/QUICKSTART.md): configuration, final-model training, and inference.
- [Paper-to-code map](docs/DESIGN_MAP.md): experiments and implementations.
- [Evaluation](docs/EVALUATION.md): JevBench, Kev, MC11, and stored scores.
- [Extension guide](docs/EXTENDING.md): custom recipes and model changes.
- [Validation](docs/VALIDATION.md) and [packaging](docs/PACKAGING.md).

Project code uses [Apache-2.0](LICENSE). Attribution and third-party terms are in [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model and dataset terms are separate. Cite the paper using [CITATION.cff](CITATION.cff).
