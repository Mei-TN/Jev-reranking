# Training and serving environment

`training-lock.txt` preserves the recorded 79 distribution versions for the
paper's NeoHorse training environment, using CPython **3.12.13** on Linux x86-64.
The additional **Pillow 11.3.0** wheel is recorded separately. The lock is a version
inventory; the exact wheel filenames, sizes, hashes and acquisition URLs are in
[`training-media.json`](training-media.json). No installation media, model weights
or third-party source library is included in this repository.

The recorded combination is **Torch 2.8.0+cu128 / Triton 3.7.1**. Torch's wheel
metadata requires Triton 3.4.0 on Linux x86-64, so this combination deliberately
uses `--no-deps` and **`pip check` exits 1**. Do not call it dependency-clean or let
a resolver silently replace the experiment's Triton version. Changing those
versions defines a different environment and requires its own validation.

## Prepare exact media

Use a bootstrap Python 3.12 or newer. The downloader uses only the standard
library; it verifies size and SHA-256, reuses matching completed files and retains
partial downloads for a later retry. It does not install or import packages.

```bash
python3 scripts/fetch_environment_media.py --output /srv/jev/media/model --workers 4
```

The 86 external files total **4,222,803,808 bytes**: 79 model-role wheels, one
standalone CPython archive, the Pillow wheel and five Debian library archives.
The recorded URLs identify those inputs, rather than promising that every host
can currently reach them. If a transfer fails, retain its report and obtain the
same size/hash through a permitted mirror or an existing wheelhouse. Credentials
belong in the provider's credential store or environment, never in committed
configuration or logs. Libraries are fetched from their recorded package
providers; model and dataset acquisition is a separate step.

## Install into a new owned prefix

The automatic route targets **Debian/Ubuntu-compatible Linux x86-64** with
`/usr/bin/dpkg-deb`, compatible glibc, at least **30 GiB free space**, and a prefix
you can write. It extracts the five libraries inside that prefix; it does not use
`apt`, `sudo`, a shared Python environment or a system package installation.
The host NVIDIA driver remains an external requirement for later GPU execution.

From the repository root, choose a new prefix:

```bash
python3 -c "from pathlib import Path; Path('/srv/jev/envs/model-001').mkdir(parents=True, exist_ok=False)"
cp requirements/environment/*.py requirements/environment/SPEC.template.json /srv/jev/envs/model-001/
cp requirements/environment/model-python /srv/jev/envs/model-001/
chmod +x /srv/jev/envs/model-001/model-python
python3 /srv/jev/envs/model-001/configure_media.py --media-directory /srv/jev/media/model
python3 /srv/jev/envs/model-001/rebuild_training.py
```

`configure_media.py` checks that the named files exist and binds their locations.
The installer then verifies all 86 files before extracting or installing them.
It creates its standalone interpreter, venv, wheelhouse, isolated libraries,
Pillow overlay, caches, actual process logs and receipts in `model-001`.
Installation is offline with `--no-index --no-deps --require-hashes`.
Its final CPU import probe checks all recorded versions and imports Torch,
Transformers, PEFT, Accelerate, Datasets, Safetensors, FLA, Triton and Pillow while
CUDA is hidden and initialization is refused. No model, dataset or GPU job is
loaded. A successful installation retains the explicit Torch/Triton conflict in
`DECLARED_METADATA_CONFLICT.json`; other errors fail the build.

These three installer controls preserve their original project-authored bytes.
Their artifact template contains portable filenames in place of deployment
locations. [`manifest.json`](manifest.json) identifies the shipped controls,
version inventory and media lists. The public route does not include historical
installation receipts or claim that these commands were rerun for this checkout.
Keep any failed attempt and choose a new prefix for another attempt.

## Use the new model Python

The installed interpreter is `/srv/jev/envs/model-001/venv/bin/python`. The copied
`model-python` launcher rebuilds its own loader paths and executes that lexical
venv path. Use `/srv/jev/envs/model-001/model-python` for the site's training
`python` and benchmark `serve_python` fields. This also works when a controller
replaces the parent `LD_LIBRARY_PATH`: the launcher restores its own syslibs,
Torch and NVIDIA wheel directories, then existing standard host-driver paths.
It preserves command arguments and the caller's CUDA device selection.

For the additional Pillow installation below, set the private library paths and
temporary directories in your shell:

```bash
export MODEL_ENV=/srv/jev/envs/model-001
export LD_LIBRARY_PATH="$MODEL_ENV/syslibs/usr/lib/x86_64-linux-gnu:$MODEL_ENV/syslibs/lib/x86_64-linux-gnu:$MODEL_ENV/venv/lib/python3.12/site-packages/torch/lib"
for directory in "$MODEL_ENV"/venv/lib/python3.12/site-packages/nvidia/*/lib /usr/local/nvidia/lib /usr/local/nvidia/lib64; do
  if [ -d "$directory" ]; then export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$directory"; fi
done
export TMPDIR="$MODEL_ENV/tmp" PIP_CACHE_DIR="$MODEL_ENV/cache/pip"
```

Install the same pinned Pillow wheel inside the venv so subsequent entry points
do not need the installer's temporary `PYTHONPATH` overlay:

```bash
CUDA_VISIBLE_DEVICES='' "$MODEL_ENV/venv/bin/python" -I -B -m pip install --no-index --no-deps --no-cache-dir --no-compile /srv/jev/media/model/pillow-11.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
CUDA_VISIBLE_DEVICES='' "$MODEL_ENV/venv/bin/python" -I -B -c "import PIL; print(PIL.__version__, PIL.__file__)"
```

Bind the `model-python` launcher in your site configuration. A separate capable
system Python may run the benchmark controller; OpenCompass and MC workers need
their own explicit environments. A successful model-role installation does not
install the paper's external evaluator forks or establish a benchmark result.
See [environment guidance](../docs/ENVIRONMENT.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md).

For an existing compatible CPython environment, `pip install --no-deps -r
requirements/training-lock.txt` is a version-only alternative. It requires an
appropriate package index or local wheelhouse, does not verify the media hashes,
and does not create the standalone interpreter or private system libraries.
Use the exact-media route above for those identities.

For JevBench/Kev evaluation, the separate
[small OpenCompass JEV plugin](../third_party/opencompass_jev/README.md) supplies
the paper's decision loaders/scorers on a fixed publicly obtainable upstream
revision. The MC evaluator still requires its separately permitted external
RunWeaver/OpenJev dependency; neither is installed by the model-role commands.
