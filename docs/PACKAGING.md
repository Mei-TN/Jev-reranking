# Building a distribution

Use the same Python environment for build tools and the build itself:

```bash
python -m pip install build==1.6.1 hatchling==1.32.4
python -m build --no-isolation
```

The default command builds a source distribution and then builds the wheel from that source distribution. A direct wheel build is also supported:

```bash
python -m build --wheel --no-isolation
```

The wheel includes the command package, paper configurations and required implementation sources. Model weights, datasets and local site files are excluded. The implementation sources are stored in a deterministic ZIP and checked when materialized into the resource cache, allowing deep source layouts to work on Windows.

Install the resulting wheel in a separate environment and check the installed resources:

```bash
python -m pip install /path/to/jev_research-0.1.0-py3-none-any.whl
jev verify
jev list --study e1e2
jev benchmark --protocol e11 --help
```

Set `JEV_RESOURCE_CACHE` to choose a writable cache location. `jev fork` copies an implementation to an editable directory for custom experiments. Models and training/evaluation dependencies are configured separately from the control package.

The repository's CI checks the editable source, a source-distribution build and an independent wheel installation on Linux and Windows. Validation commands and scope are described in [VALIDATION.md](VALIDATION.md).
