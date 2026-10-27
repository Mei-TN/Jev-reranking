# JEV Reranking

论文 **Improving JEV Entailment Reranking with Span Readouts and Selective Adaptation** 的代码。

[English](README.md) · [上手指南](docs/QUICKSTART.zh-CN.md) · [论文与代码对照](docs/DESIGN_MAP.md)

本仓库实现 JEV 蕴含重排序中的跨度均值聚合与选择性骨干适配：对每个 NLI 选项内部的状态取平均，并仅在注意力和门控 DeltaNet 投影层训练 LoRA，移除 FFN 适配器。

## 先在 CPU 上预览配置

使用 Python 3.12 或以上版本。以下命令可在 Bash 或 PowerShell 中执行：

```bash
git clone https://github.com/Mei-TN/Jev-reranking.git
cd Jev-reranking
python -m pip install -e .
jev verify
jev list --study e458
jev plan --site configs/site.example.json --study e458 --arm POOL_NOFFN --fold F2 --seed 44
```

这一步检查源码清单并展示最终模型的训练配置，无需 GPU、模型或数据，也不会运行模型。开发测试见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 训练前需要准备什么

训练和推理需要 Linux、CUDA GPU、[模型环境](docs/ENVIRONMENT.md)、NeoHorse-1-4B 权重及对应训练 fold。[上手指南](docs/QUICKSTART.zh-CN.md)说明具体要填写哪些路径。

完整训练语料、指定的外部评估分支和数值补充包目前尚无公开下载入口。三者分别用于论文训练、完整 benchmark 评估和已有分数复算；详细版本及获取状态见[数据与模型说明](docs/DATA.md)。

## 训练最终方案

准备好模型环境和训练数据后，创建站点配置：

```bash
jev init --site configs/site.local.json
```

按[上手指南](docs/QUICKSTART.zh-CN.md)填写模型、数据、Python 和 GPU 设置，再运行：

```bash
jev doctor --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --imports
jev train --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --gpu 0
jev train --site configs/site.local.json --study e458 --arm POOL_NOFFN --fold F2 --seed 44 --gpu 0
```

`LL_A` 是全模块指针基线；`POOL_NOFFN` 是选择性适配与跨度均值聚合的最终组合。后者读取基线保存的初始参数，不从基线训练后的权重继续训练。默认输出是 `work/runs/e458/E458_F2s44_POOL_NOFFN/output`。

[上手指南](docs/QUICKSTART.zh-CN.md)提供两条可选路线：分别执行训练、导出、预测与评分，或使用 pipeline 统一运行；消融实验单独列出。路径、材料或依赖未准备好时，`doctor` 会返回 `INCOMPLETE` 和退出码 2。

## 更多说明

- [上手指南](docs/QUICKSTART.zh-CN.md)：配置、最终模型训练与推理。
- [论文与代码对照](docs/DESIGN_MAP.md)：实验配置和实现位置。
- [评估](docs/EVALUATION.md)：JevBench、Kev、MC11 和已有分数复算。
- [扩展代码](docs/EXTENDING.md)：自定义训练配方与模型修改。
- [验证](docs/VALIDATION.md)与[打包](docs/PACKAGING.md)：测试和发行包构建。

项目代码采用 [Apache-2.0](LICENSE)。归属与第三方条款见 [NOTICE](NOTICE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；模型和数据遵守各自条款。论文引用信息见 [CITATION.cff](CITATION.cff)。
