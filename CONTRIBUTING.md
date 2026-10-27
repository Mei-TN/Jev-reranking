# Contributing

Install Python 3.12+ and the control package with its test dependencies:

```bash
python -m pip install -e ".[test]"
jev verify
python -m unittest discover -s tests -v
```

Keep changes focused on an experiment, interface or documented bug. Explain the behavior changed and the checks performed. Native GPU experiments should state the model/data revisions, source version, complete recipe, initialization, hardware and evaluation protocol.

Use a new experiment ID when changing a scientific choice. Preserve the named-option API, soft-target semantics and candidate order unless the new experiment explicitly changes them. See [EXTENDING.md](docs/EXTENDING.md) for the relevant interfaces.

Keep models, datasets, local site files, credentials, environment directories and run outputs outside version control. Test fixtures should be small authored examples.
