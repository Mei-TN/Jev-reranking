# Evaluation

## Custom decision requests

The portable JSONL path separates requests, labels and predictions. Each request line contains an `id` and a native `request`; labels use the same ID. The examples in `examples/` show Choice and binary (`noul`) questions. The native schema also supports ordered Score questions.

```json
{"id":"example-1","request":{"state":"The customer was charged twice.","questions":{"category":{"type":"choice","instructions":"Choose the request category.","criteria":{"billing":"A payment issue","delivery":"A shipping issue"}}}}}
```

The corresponding label is:

```json
{"id":"example-1","targets":{"category":"billing"}}
```

Run prediction with an exported bundle, then score the output using the actual model name in `model_manifest.json`:

```bash
jev predict --site configs/site.local.json --bundle /path/to/bundle --gpu 0 \
  --requests examples/requests.jsonl --output /path/to/predictions.jsonl
jev evaluate --requests examples/requests.jsonl --labels examples/labels.jsonl \
  --predictions /path/to/predictions.jsonl --model MODEL_NAME --output /path/to/scores
```

The evaluator checks question coverage, candidate identity/order and probability validity before computing its metrics. Its question-level aggregation is a custom-data interface; the paper endpoints use the definitions below.

## Native paper endpoints

| Endpoint | Population and aggregation |
| --- | --- |
| JevBench | 231 public questions; mean over 18 equally weighted families |
| Kev | Six development/test selections from decision-v7, transfer-v4 and transfer-v9; exclude `unknowable`, score questions, then average the six accuracies |
| MC11 | Eleven configurations from seven sources, 46,475 question occurrences; arithmetic mean of configuration accuracies |

MC11 evaluates candidates through entailment requests. It selects the maximal raw entailment probability, with the first maximal candidate in loaded order winning a tie. Candidate construction, demonstrations and semantic option keys are part of the protocol. These are reranking scores rather than conventional generation or likelihood scores for the same datasets.

### Configure the native runner

Copy `configs/benchmark-site.example.json` to a local site file. Bind the model, serving/OpenCompass/MC interpreters, exact external evaluator checkouts, decision cache, dataset caches, selections and verification examples. Set the guard to the host, account, UID and GPU model that you actually use. Role requirements are described in [ENVIRONMENT.md](ENVIRONMENT.md).

Prepare a one-epoch run without loading a model:

```bash
jev benchmark --site configs/benchmark-site.local.json \
  --checkpoint /path/to/checkpoint --code-tree /path/to/the/actual/training/source \
  --run-id pointer-native --gpu 0 --output /path/to/benchmark --prepare-only
```

Use the source that trained the checkpoint. The runner verifies its configuration, expected population and source identity. After preparation succeeds and the selected GPU is available, remove `--prepare-only` to execute.

For E11, select the two-epoch protocol and the contract for the actual fold:

```bash
jev benchmark --protocol e11 --training-contract two_epoch_F2 \
  --site configs/benchmark-site.local.json \
  --checkpoint /path/to/two-epoch-checkpoint --code-tree /path/to/e11/source \
  --run-id pointer-2ep-native --gpu 0 --output /path/to/benchmark-2ep --prepare-only
```

The F2 contract expects 3,864 updates and 30,912 source-record uses. F3 uses `two_epoch_F3`, 3,862 updates and 30,884 record uses.

## Fixed-interface and appendix studies

The [design map](DESIGN_MAP.md) identifies the study-specific request builders and scorers:

- PIQA direct decisions versus NLI, description templates and semantic option permutations.
- GPQA Diamond with deterministic, disjoint Main demonstrations.
- SocialIQA direct and NLI interfaces on the sealed 1,954-question population.
- Additional-source PIQA, CommonsenseQA and CLINC150 protocols.
- Same-forward residual removal and short synchronized training trajectories.
- Reference-conditioned grading as an additional diagnostic outside the current manuscript.
- Configuration/source weighting and leave-one-source-out score aggregation.

The MC11 GPQA few-shot protocol and the disjoint-demonstration control use different demonstration pools and candidate construction; report them separately.

## Reconstruct stored paper scores

```bash
jev reproduce-scores --archive /path/to/SCORING_SUPPLEMENT_REVISED.zip --output work/stored-scores
```

This CPU-only command runs the original numerical-supplement scorer in a fresh output directory. The separate archive and its supported model roster are described in [DATA.md](DATA.md). New training and new inference need their own results.
