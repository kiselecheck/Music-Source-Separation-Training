# Publishing MSST to PyPI

The release workflow builds both an sdist and wheel, validates the README and
metadata, inspects wheel contents, installs the wheel outside the checkout, and
then publishes through PyPI Trusted Publishing.

## One-time PyPI setup

Create a pending Trusted Publisher for the `msst` project with these values:

- PyPI project: `msst`
- GitHub owner: `ZFTurbo`
- Repository: `Music-Source-Separation-Training`
- Workflow: `release.yml`
- Environment: `pypi`

Create a GitHub environment named `pypi`. Requiring a maintainer approval for
that environment is recommended. No long-lived PyPI token is needed.

If releases are published from a fork, use that fork's owner and repository in
the pending publisher instead.

## Release procedure

1. Update `msst/_version.py`.
2. Run `python -m pytest -q`.
3. Build locally with `python -m build` and run
   `python -m twine check --strict dist/*`.
4. Merge the release commit into `main`.
5. Create a GitHub release tagged `v<version>`, for example `v0.1.0`.

The workflow refuses to publish when the tag and `msst.__version__` differ.
PyPI does not allow replacing an uploaded version, so a failed or incorrect
release must be followed by a new version number.

## Package layout

The wheel contains one import namespace: `msst`. Model implementations,
datasets, utilities, inference, training, validation, and compatibility code
all live below that namespace. Repository-level `train.py`, `inference.py`, and
related files are source-checkout wrappers and are intentionally excluded from
the wheel. This avoids installing collision-prone top-level packages such as
`models` and `utils`.

Model configs and checkpoints are released separately from the Python package.
See [model artifact versioning](model_artifacts.md).
