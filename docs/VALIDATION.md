# Validation

## Check the public source

Run from the repository root with the same Python used for installation:

```bash
python -m pip install -e ".[test]"
python scripts/verify_sources.py
python scripts/audit_release.py --summary
python -m unittest discover -s tests -v
```

`verify_sources.py` checks the selected implementation sources, study configurations and environment records against their manifests. The tests exercise recipe arguments and source selection, prerequisite handling, pipeline stage reuse, checkpoint versus bundle inference, prediction/scoring contracts and distribution resources. `audit_release.py` reports literal credential patterns and locations without printing matched values.

The GitHub Actions workflow checks the source, CPU tests, source-distribution and wheel builds, and an independent wheel installation on Ubuntu and Windows. Tests cover training entry points, source selection, checkpoint inference, and the isolated model launcher's library paths and arguments.

## Scientific scope

The training implementations use experiment-specific initialization, optimizer membership, schedules, and option readouts. The control package selects an implementation and binds its deployment paths.

CPU and packaging checks do not rerun model training. The complete paper experiment matrix has not been rerun from this public checkout.

The small OpenCompass integration includes its own source-application and synthetic loader/scorer checks. See [its guide](../third_party/opencompass_jev/README.md) for the actual coverage. Full benchmark execution requires the separate assets and runtime dependencies described in [DATA.md](DATA.md).

## Reproduction expectations

Match the complete recipe, model/tokenizer revisions, frozen inputs, genuine initialization and evaluation population. CUDA training is nondeterministic in the original experiments. Successful source/installation checks establish software preparation; new model and benchmark results should retain their actual hardware, output and process-exit records.
