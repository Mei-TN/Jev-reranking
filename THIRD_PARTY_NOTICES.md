# Third-party notices

The repository's Apache-2.0 license covers the project's code and modifications.
It does not grant rights to external libraries, evaluator forks, model weights,
datasets, cached predictions or installation media. This public directory does
not include the complete OpenCompass or RunWeaver libraries, their wheels, or
installed Python environments. A small licensed OpenCompass decision plugin is
included as described below.

## Included kev-derived code

The selected research implementations derive from the NeoHorse `kev` training
package, originally supplied under Apache-2.0. Its original license text is
retained byte for byte at [`third_party/kev/LICENSE`](third_party/kev/LICENSE).
[`third_party/kev/NOTICE`](third_party/kev/NOTICE) describes the study's changes;
the source map identifies the selected variants. Original attribution in source
files must remain with those files.

The [NeoHorse model repository](https://huggingface.co/TokenRhythm/NeoHorse-1-4B)
is the upstream model reference. Model weights, tokenizer payloads and model-card
terms are external to this code distribution. Obtain the recorded revision and
retain its own license with any permitted model distribution.

## External evaluator dependencies

| Component | Public-source route and scope |
| --- | --- |
| OpenCompass | The [upstream project](https://github.com/open-compass/opencompass) publishes its own Apache-2.0 license. The included [small JEV plugin](third_party/opencompass_jev/README.md) targets publicly obtainable revision `a25fdd25dbe6a6e1da355332fa52e82dcfc3a1ee` (tag `0.5.3`) and retains the [OpenCompass license](third_party/opencompass_jev/LICENSE). It contains selected unchanged loader/scorer payloads and an explicit compatibility patch, with before/after identities from the paper fork `605996a616035bc729b1be2a70af042ebc1fe5d8`. The complete library and historical wheel are external. Matching version `0.5.3` alone does not identify either source tree, and selected byte identity does not establish full benchmark equivalence. |
| JevBench scorer in the plugin | MIT, copyright Florian Standhartinger and contributors. The original [license text](third_party/opencompass_jev/overlay/opencompass/datasets/_decision_upstream/JEVBENCH_LICENSE), source revisions and integration notes remain with the selected code. Benchmark dataset rows are external. |
| Kev scorer in the plugin | Apache-2.0. The original [license text](third_party/opencompass_jev/overlay/opencompass/datasets/_decision_upstream/KEV_LICENSE) and source revision record remain with the selected code. |
| RunWeaver / `agentic-evaluate` | The paper's retained handoff revision is `1a1da486f84258a5b72f3d94b7e54954e4d4eb2f`, distribution version `0.1.0`. The examined source archive had no LICENSE/COPYING/NOTICE file or declared package license. Public distribution rights have not been established, so no RunWeaver source, wheel or source archive is included here. Obtain the exact dependency through a permitted route from its owner; this project's Apache-2.0 license does not authorize its use or redistribution. |
| OpenJev and MC runtime packages | The benchmark calls external RunWeaver/OpenJev interfaces and their recorded package/data versions. Their own source, dependency, model and dataset terms apply. The original MC environment is not supplied by installing this project's command line. For example, `chess` is a separate GPL-3.0-or-later dependency, not project code. |

Source version identifiers describe reproducibility requirements; they are not
public download guarantees or permission grants. Newly built wheels retain their
own measured identities. Matching a distribution version alone does not identify
a historical binary or its fork.

## External training libraries and assets

[`requirements/training-lock.txt`](requirements/training-lock.txt) records package
versions. [`requirements/training-media.json`](requirements/training-media.json)
records external wheel, interpreter and system-library artifact identities and
URLs. These files are metadata; they do not include the third-party packages.
PyTorch, NVIDIA libraries, Triton, FLA, Transformers, PEFT, Datasets, Pillow,
CPython and the other listed dependencies retain their own license and notice
files. Preserve those files with any permitted redistribution rather than
applying the project's license to the downloaded media.

NeoHorse, Qwen3, JevBench/Kev, GPQA, SocialIQA and other referenced model or dataset
assets also retain their respective access rules and terms. The source package
does not include weights, dataset rows, teacher predictions, benchmark caches or
the optional stored-score archive. Consult the asset guide before supplying them.
