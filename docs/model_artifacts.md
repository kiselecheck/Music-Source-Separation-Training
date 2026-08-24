# Model artifact versioning

The `msst` Python distribution and trained models have independent release
cycles. A package release ships runtime code; it does not bundle experiment
configs or checkpoint weights.

Use separate identifiers for the three compatibility layers:

| Layer | Example | Purpose |
| --- | --- | --- |
| Python package | `msst==0.1.0` | Runtime API and implementations |
| Model artifact | `dereverb-v4` | A specific config/checkpoint pair |
| Checkpoint format | `1` | Serialization metadata understood by MSST |

Every published model should distribute its YAML and checkpoint together and
record:

- artifact name and version;
- `model_type`;
- SHA-256 hashes of the YAML and checkpoint;
- minimum compatible `msst` version;
- checkpoint format version;
- license and download source;
- target instruments, sample rate, and attribution.

New checkpoints written by MSST include `meta.format_version`, `meta.msst_version`,
`meta.model_type`, and output-stem metadata. Legacy checkpoints without `meta`
remain loadable. A checkpoint declaring a newer format fails with an upgrade
message instead of being loaded ambiguously.

Model artifacts should live in GitHub Releases, Hugging Face, or another model
registry. Experiment-only configs stay in the repository under `configs/`.
Checkpoint binaries must not be committed to the source tree or included in
the wheel.

A future `Separator.from_pretrained(name, version=...)` API can consume a
registry manifest following these rules without coupling model releases to
PyPI releases.
