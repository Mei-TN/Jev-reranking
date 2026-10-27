# JEV decision plugin for public OpenCompass

This small overlay supplies the JevBench/Kev dataset and scorer interfaces that
the paper runtime imports. It contains **16 original files / 79,904 bytes**,
including the two scorer licenses, plus a **23,058-byte patch** for five shared
OpenCompass interfaces. The complete OpenCompass library, RunWeaver, model weights
and benchmark data remain external.

The obtainable base is official OpenCompass **0.5.3**, commit
`a25fdd25dbe6a6e1da355332fa52e82dcfc3a1ee`:

```bash
git -c core.autocrlf=false clone --depth 1 --branch 0.5.3 --single-branch https://github.com/open-compass/opencompass.git /srv/jev/external/opencompass
git -C /srv/jev/external/opencompass rev-parse HEAD
python3 scripts/apply_opencompass_jev.py --source /srv/jev/external/opencompass --check
python3 scripts/apply_opencompass_jev.py --source /srv/jev/external/opencompass --apply
python3 scripts/apply_opencompass_jev.py --source /srv/jev/external/opencompass --verify
```

Use a new, clean checkout at that exact commit. The helper checks original
shared-file hashes before applying the patch, refuses existing overlay
destinations, and verifies all resulting selected files. It invokes Git without
a shell and pins patch line endings. It does not install packages or call a
model. Preserve a failed attempt; do not overwrite local source changes.

The paper's original fork is revision
`605996a616035bc729b1be2a70af042ebc1fe5d8`. All overlay payloads and the five
resulting interface files retain that fork's bytes, pinned in
[`manifest.json`](manifest.json). The public base plus this selection is a
separately identified integration, not a claim of complete historical fork
identity. Plain upstream lacks `decision_native` and `decision_benchmarks`.

The patch preserves the paper interfaces for prompt handling, parallel
inferencer `batch_size`, request-error persistence, ordered outputs, JSONL resume
and structured prediction/evaluation handling. Dataset projection and scoring
functions remain unchanged in the overlay.

## Dependency environment and CPU checks

Use a separate OpenCompass environment. The original evaluator used CPython
3.12.10, Torch 2.6.0+cpu, Transformers 4.51.3 and Datasets 3.6.0. The ordinary
upstream installation route for a new candidate environment is:

```bash
python3.12 -m venv /srv/jev/envs/opencompass
/srv/jev/envs/opencompass/bin/python -m pip install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
/srv/jev/envs/opencompass/bin/python -m pip install -e '/srv/jev/external/opencompass[api]'
```

These dependency-resolution commands are a setup route, not a measured exact-media
installation of this public selection. Preserve actual logs and any dependency
or import failures. Model training, MC runtime and data preparation remain
separate. An existing compatible evaluator environment can import the selected
checkout through its own path without replacing its installed package.

Run synthetic checks from this repository root:

```bash
CUDA_VISIBLE_DEVICES='' /srv/jev/envs/opencompass/bin/python -B third_party/opencompass_jev/cpu_check.py --source /srv/jev/external/opencompass --work-dir /srv/jev/work/oc-fixture-001 --runtime-tree vendor/e12
```

The fixture directory must be new. The check verifies selected source hashes,
imports dataset/scorer and parallel-inferencer interfaces, exercises Choice,
Score and Noul projection, the distinct native tie rules, invalid-output
denominators and the paper runtime import. Only its synthetic input locator is
replaced in that process; source files and installed packages remain unchanged.
It hides CUDA, refuses CUDA initialization and disables network use during
fixture evaluation. It sends no HTTP request and loads no model or real
benchmark population. CPU fixture success establishes these boundaries, not
full native benchmark equivalence.

## Attribution

- OpenCompass and the compatibility changes: Apache-2.0;
  original [LICENSE](LICENSE), copyright OpenCompass Authors.
- JevBench scorer: MIT, copyright Florian Standhartinger and contributors;
  [JEVBENCH_LICENSE](overlay/opencompass/datasets/_decision_upstream/JEVBENCH_LICENSE).
- Kev scorer: Apache-2.0;
  [KEV_LICENSE](overlay/opencompass/datasets/_decision_upstream/KEV_LICENSE).

Original scorer revisions and integration notes remain in
[`SOURCES.json`](overlay/opencompass/datasets/_decision_upstream/SOURCES.json).
Keep the notices and licenses with any permitted redistribution of this plugin.
