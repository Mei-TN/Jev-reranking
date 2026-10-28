# Paper-to-code map

This map follows **Improving JEV Entailment Reranking with Span Readouts and Selective Adaptation**, including its appendix studies. The machine-readable experiment catalog is [`configs/catalog.json`](../configs/catalog.json). Each configuration specifies its own source and initialization dependencies.

## Model and request contract

`kev/model.py`, `kev/heads.py` and the native decision API implement the shared backbone/head interface. Choice, binary and ordered Score questions use named candidates. The loss accepts native soft targets and averages question losses within a record.

The baseline head follows the [released NeoHorse-Jev implementation](https://github.com/TokenRhythm/NeoHorse/blob/c494069accfc2d08839675961c1fdfcc54ba0b92/jev/package/src/neohorse_decision/_vendor/model.py#L122-L130), with the [study implementation](../vendor/e12/kev/heads.py#L66-L85) preserved in this release. Its affine query/key projections produce scaled dot-product logits; mean-span pooling changes the option readout supplied to that head.

The main experiments use [NeoHorse-1-4B weights](https://huggingface.co/TokenRhythm/NeoHorse-1-4B) at revision `56f0584bb40578a2c33b1b40a08ccd17243ad710`. The [official Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B) describes the underlying attention/Gated DeltaNet hybrid architecture; [Gated Delta Networks](https://arxiv.org/abs/2412.06464v3) is the source for the gated delta-rule mechanism.

The encoder uses explicit state, instruction, option and decision delimiters. Mean-span pooling excludes option delimiters and falls back to the option-ending state for empty content. NeoHorse's hybrid recurrent backbone uses separate causal question rows; the Qwen3 study uses its own dense-backbone implementation and adapter roster.

## Training studies

| Paper study | Implementation / configuration | Relevant manuscript section |
| --- | --- | --- |
| Factorial head/adapter rates | `vendor/e12`; `provenance/plans/e1e2.json`; `jev ... --study e1e2` | Training Recipe and Reference Quality |
| Repeated pointer, frozen residual and trainable residual | Same E1/E2 plan; arms `LL_A`, `B`, `FROZEN`, `TRAINABLE`; native process diagnostics | Option Aggregation and Scoring Head; Residual Corrections and Training Controls |
| Adapter scope, nearby rank budget, mean-span and last-content readout | `vendor/e458`; `provenance/plans/e458.json`; `jev ... --study e458` | Combining Selective Adaptation and Span Pooling; Adapter Scope and Parameter Budget; Option Aggregation and Scoring Head |
| Two-epoch pointer/pooling budget | `vendor/e11`; `provenance/plans/e11.json`; `jev ... --study e11` | Training Recipe and Reference Quality |
| Six-head pilot | `vendor/head1_v1`, including its head/readout tests | Additional implementation; not reported in the current manuscript |
| Joint learning-rate pointer/residual/pooling comparisons | `vendor/historical_heads` and the lower-rate `vendor/r3_control` source; catalog-selected recipes | Learning-rate scaling and staged adaptation |
| Head-first, head-only-tail and reduced-scope controls | `vendor/historical_heads` and `vendor/historical_reverse`; catalog-selected recipes | Learning-rate scaling and staged adaptation |
| Shared-initialization adapter-scope comparison | `vendor/historical_lowrate`; catalog-selected recipes | Additional training comparisons |
| Qwen3-4B pointer/pool pairs | Actual Qwen source/configuration selection in the catalog | Readout on Qwen3-4B |
| Seven paired higher/half-rate pointer retrainings | Release-retraining source/specification selection in the catalog | Learning-rate scaling and staged adaptation |
| Portable pointer/pooling pair | Portable package source/configuration selection in the catalog | Evidence coverage and reproduction; no separate result table |

The main recipe sets contain 56 training runs: 28 learning-rate/residual controls (`e1e2`), 20 scope/rank/readout controls (`e458`), and eight two-epoch controls (`e11`). Dependent comparisons reuse the specified reference initialization.

The four-setting studies cross folds F2/F3 with seeds 44/45. Rate letters are **head then adapter**, L = `1.25e-5`, H = `2.5e-5`. E458 and E11 require the corresponding genuine reference initialization. E11 restarts from step-zero tensors with the complete two-epoch schedule.

The main recipes use [PyTorch 2.8's two-phase cosine OneCycleLR](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1966-L2048), based on [Smith's 1cycle policy](https://arxiv.org/abs/1803.09820v2). The implementation uses `pct_start=0.1`, `div_factor=25` and `final_div_factor=1e4`, with AdamW's `beta1` cycling inversely from 0.95 to 0.85 and back.

Additional study recipes are under `configs/studies`; the catalog selects their source and training entry point. Some implementations use their native checkpoint loader. The release-retraining exporter is provided separately under `vendor/release_runtime` and requires its external official bundle template. The portable pipeline selects checkpoint inference for implementations without a native exporter.

Use any additional recipe through the same site binding:

```bash
jev plan --site configs/site.local.json --recipe configs/studies/qwen3/qwen3_F2s44_pointer.json
```

The standalone portable/Qwen packages provide runtime-configuration tools. Copy the package to your own `work/` directory before running those tools: runtime binding creates local manifests and updates the copied package inventory. Keep the source-controlled implementation unmodified.

For the supplementary shared-initialization no-FFN recipes, first use the corresponding Q4 pointer recipe under `configs/studies/historical_joint_rate`. Its native diagnostics save `head_initial.safetensors` and `adapter_initial.safetensors`. The dependent recipe expects those inputs under `{data_root}/initialization/ARR_R1_Q_f{fold}_s{seed}/`; use its `initialization_notes` and pinned identities to select the correct pair. These controls copy the saved initial tensors rather than resuming final weights. A newly defined run with different initialization identities should use its own recipe and actual input pins.

## Readouts and controls

| Design | Computation |
| --- | --- |
| Option-ending pointer | Width-256 affine query/key projections and scaled bilinear logits at the option-ending state |
| Narrow pointer | Width 64, with the pilot's canonical projection-row initialization |
| Tanh pointer | Tanh-transformed query/key projections |
| Cosine pointer | FP32 normalization with epsilon `1e-12`, scaled by 16 |
| Residual correction | Pointer plus a zero-output-initialized 128-unit GELU pair MLP |
| Frozen residual | Same residual forward computation with the four MLP tensors excluded from the optimizer |
| Mean-span pooling | Mean of option-content hidden states before the same key projection |
| Last-content readout | Final option-content state, with the same empty-content fallback |

Readout, optimizer membership, adapter target set and rank are distinct configuration choices. The rank-28 and rank-64 no-feed-forward variants have nearby, unequal parameter counts.

## Evaluation and analysis

| Paper protocol | Code selection |
| --- | --- |
| JevBench, Kev and MC11 | `vendor/evaluation` and the native runtime workers; `jev benchmark` |
| PIQA direct/NLI and template/order conditions | `vendor/inference` |
| Disjoint GPQA demonstrations | `vendor/inference` request preparation and scoring |
| SocialIQA direct/NLI interfaces | `vendor/socialiqa` |
| Additional-source PIQA, CommonsenseQA and CLINC150 | `vendor/readout_analysis` external-source evaluator |
| Same-forward residual removal | `vendor/readout_analysis` paired-branch and branch-head evaluator |
| Short synchronized trajectories | `vendor/r3_trace` |
| Reference-conditioned grading (additional implementation, not reported in the current manuscript) | `vendor/grading` |
| Stored-probability scores and aggregation sensitivity | `vendor/stored_scoring`; `jev reproduce-scores` with the separate numerical supplement |

The catalog also includes the six-head pilot and reference-conditioned grading tools as additional implementations; they are not reported experiments in the current manuscript. Each protocol specifies its populations, checkpoint roster and candidate semantics. [EVALUATION.md](EVALUATION.md) describes the main interfaces, and [DATA.md](DATA.md) identifies separate inputs. New experiments can reuse these interfaces through an explicit [custom recipe](EXTENDING.md).
