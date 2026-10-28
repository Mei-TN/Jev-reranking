# Models and data

Model weights, datasets, raw predictions and checkpoints are separate assets. They are not included in the source repository or Python wheel. Acquire them under their respective terms and bind their locations in the site configuration.

## Backbones

| Study | Model | Recorded revision |
| --- | --- | --- |
| NeoHorse studies | `TokenRhythm/NeoHorse-1-4B` | `56f0584bb40578a2c33b1b40a08ccd17243ad710` |
| Qwen3 study | Qwen3-4B | `2c54d5a09e7e92d4f5126b92a5a457448c9593e6` |

Keep the complete tokenizer and model files together. Tokenization, delimiter IDs and candidate boundaries are part of the experiment. ModelScope is the preferred download source when the recorded asset is available there; use credentials from your environment or credential store.

## Training folds

The paper uses a grouped corpus with 22,035 records. Folds 2 and 3 form the four-setting comparisons with seeds 44 and 45:

| Fold | Training records | SHA-256 |
| --- | ---: | --- |
| F2 | 15,456 | `9253f1c937978174b2ff2e2959ce732c06fe3c0a220154fb962be1c2a1478017` |
| F3 | 15,442 | `2a1cfac79cd64d9cb22f43c4b69c3283fd44c73c92a4468c661d27ac9c897e2a` |

The manuscript describes public source datasets and the grouped split construction. Author-created supplements are also required to reconstruct the exact corpus; public source downloads alone do not recover all transformed records. The intermediate SetFit Amazon conversion has not been recovered. A public download of the complete training corpus is not currently provided.

Dependent scope/readout/budget experiments additionally require saved step-zero head/adapter tensors and generator records from their genuine prerequisite runs. Produce those references with the same base/data paths before launching dependent arms.

## Evaluation inputs

JevBench and Kev use the frozen selections and decision caches of the paper. MC11 requires the exact benchmark source revisions, candidate inventories and demonstrations. The configurations cover ARC, WinoGrande, GSM8K, GPQA, HellaSwag, MMLU and generated chess. Follow the original access conditions for each source, including gated datasets.

The additional evaluations have their own populations: PIQA and SocialIQA compare direct selection with entailment reranking; GPQA tests disjoint demonstrations; and the additional-source comparisons use PIQA, CommonsenseQA and CLINC150. Their request construction and candidate ordering must match the corresponding protocol. See [EVALUATION.md](EVALUATION.md).

The exact RunWeaver/OpenJev evaluation fork is an external dependency. Its source and binary are not redistributed here while their distribution permission remains unresolved. Package version numbers alone do not identify that fork. External evaluator setup and available source identities are documented in [ENVIRONMENT.md](ENVIRONMENT.md).

## Stored-score reconstruction

`jev reproduce-scores` accepts the separate `SCORING_SUPPLEMENT_REVISED.zip` numerical supplement. It reconstructs the supported paper scores from stored probabilities and gold identities on CPU:

```bash
jev reproduce-scores --archive /path/to/SCORING_SUPPLEMENT_REVISED.zip --output work/stored-scores
```

The required archive is 200,841,481 bytes, SHA-256 `3125ccce2f8127efcf956bf2025367081cb7f0ab3b97e3a33e03826e357d2d45`. This route reproduces scoring for its registered 72-checkpoint roster; it does not retrain models or generate new predictions. Obtain the numerical supplement separately; it is not embedded in the code release.

The public [appendix summaries](../vendor/stored_scoring/summary_views/README.md) correct four floating-point sign counts using a `1e-9` percentage-point zero tolerance. Their recorded score values, means and standard errors are unchanged. The pinned historical archive is unchanged and its original scorer retains its strict floating-point sign counts.
