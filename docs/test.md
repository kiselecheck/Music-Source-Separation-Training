# Development checks

The test suite runs from a source checkout with the optional test dependencies
installed:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The regular suite covers the public API, configuration and checkpoint handling,
inference, training utilities, and registered extension models. Tests that
compare MSST with a separate upstream checkout or require real model
checkpoints are optional and skip when their environment variables are not
configured.

Run the type checker configured for the repository with:

```bash
python -m mypy
```
