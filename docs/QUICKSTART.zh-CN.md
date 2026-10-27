# 上手指南

[仓库首页](../README.zh-CN.md) · [English](QUICKSTART.md)

先在 CPU 上查看训练配置。实际训练和模型推理需要 Linux、CUDA GPU，以及另行准备的模型和数据。

## 1. 安装并预览配置

使用 Python 3.12 或以上版本。本节命令可在 Bash 或 PowerShell 中执行：

```bash
git clone https://github.com/Mei-TN/Jev-reranking.git
cd Jev-reranking
python -m pip install -e .
jev verify
jev list --study e458
jev plan --site configs/site.example.json --study e458 --arm POOL_NOFFN --fold F2 --seed 44
```

已有仓库时，在仓库根目录执行安装及后续命令即可。`verify` 检查源码清单，`list` 列出配置，`plan` 展示解析后的训练命令。配置预览不加载模型，也不要求示例路径已存在。开发测试见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 2. 准备模型环境和材料

| 材料 | 用途 | 准备方式 |
| --- | --- | --- |
| NeoHorse-1-4B 权重和 tokenizer | 训练与推理 | 使用 [DATA.md](DATA.md) 指定的完整模型版本 |
| 模型 Python 环境 | 训练、导出与推理 | 按 [ENVIRONMENT.md](ENVIRONMENT.md) 安装依赖 |
| 原始 F2/F3 训练导出文件 | 论文训练 | `F2.train.jsonl` / `F3.train.jsonl`，哈希见 [DATA.md](DATA.md)；目前尚无公开下载入口 |
| Benchmark 数据与外部评估器 | JevBench、Kev、MC11 | 见 [EVALUATION.md](EVALUATION.md)；本次发布尚未提供指定外部评估分支的公开入口 |
| 数值补充包 | 复算已保存的论文分数 | 见 [DATA.md](DATA.md)；目前尚无公开下载入口 |

后两项用于论文评估与分数复算。仓库中的 `examples/` 文件只展示请求格式，用它们进行模型推理仍需已训练的 checkpoint 或导出模型。

首次使用时创建站点配置：

```bash
jev init --site configs/site.local.json
```

`init` 不会覆盖已有文件。编辑 `configs/site.local.json` 中的以下字段：

| 字段 | 填写内容 |
| --- | --- |
| `base_model` | 完整模型权重与 tokenizer 所在目录 |
| `data_root` | 训练导出文件所在目录 |
| `data_files` | `data_root` 下 F2/F3 的文件名 |
| `python.training`、`python.serving` | 已配置模型环境的 Python 可执行文件 |
| `hardware.gpu_name_contains` | 实际 GPU 型号；如不是 H20，替换示例值 |
| `work_root` | 输出根目录；`init` 默认设为仓库内的 `work/` 目录 |

相对路径以站点配置文件所在目录为基准。按默认布局，模型放在 `assets/NeoHorse-1-4B-hf/`，训练文件放在 `assets/training/`。同一 `work_root` 内的运行会自动找到前置初始化；若引用其他位置的运行，可设置 `reference_outputs`。

在 Linux 训练机器上检查准备情况：

```bash
jev doctor --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --imports
```

`READY` 表示检查通过。路径、材料或依赖尚未备齐时，返回 `INCOMPLETE` 和退出码 2 是预期结果；按 `problems` 列表补齐后再开始训练。

## 3. 运行最终模型：选择一条路线

下面以 F2、seed 44 为例。两条路线都先训练 `e1e2 / LL_A` 全模块指针基线，保存共同初始化，再用 `e458 / POOL_NOFFN` 训练选择性适配与跨度均值聚合的组合模型。最终模型从保存的初始参数开始，不续训基线训练后的权重。

最终运行可选手动执行或 pipeline 管理，二选一即可。以下命令均在 Linux 的仓库根目录执行，使用默认 `work_root`。

### A. 分别训练、导出、预测与评分

```bash
jev train --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --gpu 0
jev train --site configs/site.local.json --study e458 --arm POOL_NOFFN --fold F2 --seed 44 --gpu 0
```

默认最终 checkpoint 目录是 `work/runs/e458/E458_F2s44_POOL_NOFFN/output`。修改过 `work_root` 时，以 `train` 输出的路径为准。若这两个运行已完成，直接继续导出，不必重新训练。

```bash
jev export --site configs/site.local.json --tree e458 --gpu 0 --checkpoint work/runs/e458/E458_F2s44_POOL_NOFFN/output --output work/bundles/jev-noffn-span --verification-examples examples/export_requests.json
jev predict --site configs/site.local.json --bundle work/bundles/jev-noffn-span --gpu 0 --requests examples/requests.jsonl --output work/predictions.jsonl
jev evaluate --requests examples/requests.jsonl --labels examples/labels.jsonl --predictions work/predictions.jsonl --model jev-noffn-span --output work/evaluation
```

导出会比较 checkpoint 与重新加载的 bundle。导出目录名就是模型名，此处为 `jev-noffn-span`；如更改目录名，`evaluate --model` 也使用新名称。预测保存在 `work/predictions.jsonl`，评分保存在 `work/evaluation/`。再次运行时使用新的输出路径。

### B. 让 pipeline 管理最终运行

选择此路线时，不要先单独训练 `POOL_NOFFN`：

```bash
jev train --site configs/site.local.json --study e1e2 --arm LL_A --fold F2 --seed 44 --gpu 0
jev pipeline --site configs/site.local.json --study e458 --arm POOL_NOFFN --fold F2 --seed 44 --gpu 0 --verification-examples examples/export_requests.json --requests examples/requests.jsonl --labels examples/labels.jsonl
```

若当前站点下的基线已完成，跳过第一条命令。Pipeline 完成最终模型的训练、导出、预测与评分，并给出各项输出路径。`--resume` 用于复用之前一次 pipeline 已验证完成的阶段，不能接管由单独 `jev train` 命令创建的最终模型运行。

两条路线都使用小型格式示例进行推理，不是论文 benchmark。可以换成自己的请求与标签文件；论文评估流程见 [EVALUATION.md](EVALUATION.md)。

## 4. 可选：复现消融对照

训练最终方案只需要基线和 `POOL_NOFFN`。需要其他对照时再选择：

| Study / arm | 模型 | 前置初始化 |
| --- | --- | --- |
| `e458 / POINTER_NOFFN` | 选择性适配 + option-ending pointer | 同 fold/seed 的 `e1e2 / LL_A` |
| `e458 / POOL_ALL` | 全模块适配 + mean-span pooling | 同 fold/seed 的 `e1e2 / LL_A` |
| `e11 / POINTER_2EP` | 两轮全模块指针 | 同 fold/seed 的 `e1e2 / LL_A` |
| `e11 / POOL_2EP` | 两轮全模块池化 | 同 fold/seed 的 `e458 / POOL_ALL` |

用 `jev list --study e458` 或 `jev list --study e11` 查看可选的 arm、fold 和 seed。两轮对照从保存的初始化重新训练，并使用两轮调度。其他比较见[论文与代码对照](DESIGN_MAP.md)，自定义配方见[扩展指南](EXTENDING.md)。
