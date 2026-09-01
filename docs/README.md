# MSST documentation

## Start here

- [Getting started](getting_started.md) — installation and a complete first
  inference, validation, and training workflow.
- [Python API](python_api.md) — public functions, parameters, return values,
  result objects, typing, and exceptions.
- [Command-line interface](cli.md) — commands, options, shell examples, and
  exit behavior.
- [Experiments and run tracking](experiments.md) — directory layout, manifests,
  run records, lifecycle, and reproducibility data.

## Models, data, and audio

- [Compatibility](compatibility.md) — supported Python and PyTorch versions,
  model extras, devices, codecs, and output formats.
- [Pretrained models](pretrained_models.md) — available configs and weights.
- [Dataset types](dataset_types.md) — training layouts and validation data.
- [Augmentations](augmentations.md) — training data transformations.
- [BSRoformer batch sizes](bs_roformer_info.md) — memory-oriented batch-size
  reference.
- [LoRA](LoRA.md) — parameter-efficient fine-tuning.
- [Ensembling](ensemble.md) — combining separated outputs.

## Package and release design

- [Package architecture](package_architecture.md) — namespace boundaries,
  public facades, runtime ownership, and maintenance policy.
- [Model artifacts](model_artifacts.md) — config/checkpoint identity and
  checkpoint-format metadata.
- [Publishing](releasing.md) — build, validation, and release procedure.

The README at the repository root is the package landing page and PyPI
description. Detailed behavior belongs in these documents so the landing page
can remain focused on installation and first use.
