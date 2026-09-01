# MSST

[![Python package CI](https://github.com/ZFTurbo/Music-Source-Separation-Training/actions/workflows/ci.yml/badge.svg)](https://github.com/ZFTurbo/Music-Source-Separation-Training/actions/workflows/ci.yml)
![Python 3.10–3.13](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

MSST is a Python library for music source separation with PyTorch. It provides
one package for inference, validation, training, dataset metadata, preflight
checks, and experiment tracking across a broad collection of separation
architectures.

Model configurations and checkpoints are external artifacts. Supply the YAML
configuration and weights that belong to the model you want to run.

[Getting started](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/getting_started.md) ·
[Python API](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/python_api.md) ·
[CLI](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/cli.md) ·
[Compatibility](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/compatibility.md) ·
[Dataset layouts](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md) ·
[Pretrained models](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)

## Installation

MSST supports CPython 3.10, 3.11, 3.12, and 3.13.

```bash
pip install msst
```

Install the extra for the selected model architecture. Add `train` or
`validation` when those workflows are needed:

```bash
pip install "msst[bs_roformer]"
pip install "msst[validation,bs_roformer]"
pip install "msst[train,bs_roformer]"
```

On a CUDA system, install a PyTorch build compatible with the target GPU before
installing MSST. MSST does not bundle CUDA binaries.

## Inference

Use `msst.inference()` to process an input folder. The model is loaded once,
all supported audio files are separated, and the paths written to disk are
returned:

```python
import msst

written_files = msst.inference(
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    checkpoint_path="checkpoints/model.ckpt",
    input_folder="audio/input",
    output_folder="results/separated",
    device_ids=0,
)

for path in written_files:
    print(path)
```

For repeated calls, use `msst.Separator` so the weights stay loaded:

```python
import msst

with msst.Separator(
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    checkpoint_path="checkpoints/model.ckpt",
    device_ids=0,
) as separator:
    stems = separator.separate_file("audio/song.wav")
    separator.separate_folder("audio/album", "results/album")

print(separator.instruments)
print(stems.keys())
```

`Separator.separate()` also accepts an in-memory NumPy waveform. Input and
output format details are documented in the
[compatibility guide](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/compatibility.md).

## Training

`msst.train()` writes checkpoints, model metadata, and training history to
`results_path`:

```python
import msst

msst.train(
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    checkpoint_path="checkpoints/initial.ckpt",
    data_path="datasets/train",
    valid_path="datasets/valid",
    results_path="results/training",
    device_ids=(0,),
    launcher="standard",
    dataset_type=1,
    num_workers=8,
)
```

The checkpoint is optional when training from scratch. The launcher selects the
execution runtime:

| Launcher | Use case |
| --- | --- |
| `standard` | CPU, one GPU, or PyTorch DataParallel |
| `ddp` | PyTorch DistributedDataParallel |
| `accelerate` | Hugging Face Accelerate |

```bash
msst train --launcher standard --help
msst train --launcher ddp --help
msst train --launcher accelerate --help
```

Dataset structures are described in the
[dataset layout guide](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md).

## Validation

`msst.valid()` evaluates a checkpoint against reference stems and returns a
typed result containing aggregate and per-track metrics:

```python
import msst

result = msst.valid(
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    checkpoint_path="checkpoints/model.ckpt",
    valid_path="datasets/test",
    metrics=("sdr", "k_sdr"),
    device_ids=0,
)

print(result.averages)
print(result.per_track["sdr"])
```

Pass multiple device indices to run parallel validation.

## Preflight checks

Use `msst.check()` before training to detect configuration, dataset,
checkpoint, dependency, device, and memory problems early:

```python
import msst

report = msst.check(
    mode="safe",
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    checkpoint_path="checkpoints/initial.ckpt",
    data_path="datasets/train",
    valid_path="datasets/valid",
    results_path="results/training",
    device_ids=(0,),
    dataset_type=1,
)

report.raise_for_errors()
for item in report.items:
    print(item.status, item.name, item.message)
```

Two modes are available:

- `safe` validates the planned run without executing model forward or backward.
- `full` includes one real forward, loss, backward, and in-memory optimizer step
  without writing a checkpoint. On CUDA it also reports peak allocated and
  reserved memory for that step.

The full check uses one representative batch; memory use can still vary with
later batches and random augmentations.

## Dataset metadata

Training metadata can be generated independently and reused by the training
runtime:

```python
import msst

metadata = msst.build_metadata(
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    data_path="datasets/train",
    results_path="results/training",
    dataset_type=1,
    num_workers=8,
)

print(metadata.path)
print(metadata.track_count)
print(metadata.fingerprint)
```

The function writes both the metadata cache and a JSON manifest describing its
inputs and fingerprint.

## Experiments

Experiment tracking is optional. Create a portable experiment directory with
Python or the command line:

```python
import msst

experiment = msst.init_experiment("experiments/vocals")
```

```bash
msst init experiments/vocals
```

The resulting layout keeps inputs, outputs, logs, scripts, and run records
separate:

```text
vocals/
├── experiment.yaml
├── configs/
├── checkpoints/
├── jobs/
├── logs/
├── results/
└── runs/
```

Pass the experiment to inference, validation, or training:

```python
msst.train(
    model_type="bs_roformer",
    config_path="configs/model.yaml",
    data_path="datasets/train",
    valid_path="datasets/valid",
    results_path="results/training",
    device_ids=(0,),
    experiment=experiment,
    run_name="baseline",
)
```

Each call creates a record under `runs/` with its parameters, software
environment, config snapshot, input hashes, status, results, and artifacts.
Tracking does not change the paths supplied to the operation.

## Command-line interface

The same workflows are available through the `msst` command:

| Command | Purpose |
| --- | --- |
| `msst inference` | Separate every supported audio file in a folder |
| `msst train` | Train or fine-tune a model |
| `msst valid` | Evaluate a checkpoint |
| `msst check` | Validate a planned training run |
| `msst metadata` | Build dataset metadata |
| `msst init` | Create an experiment directory |

```bash
msst --help
msst inference --help
msst train --help
msst valid --help
msst check --help
msst metadata --help
msst init --help
```

## Model extras

The base installation supports model types that require no additional Python
dependencies. Other architectures have an extra with the same name:

| Model type | Extra |
| --- | --- |
| `mdx23c`, `experimental_mdx23c_stht`, `scnet`, `scnet_masked` | Base package |
| `apollo` | `apollo` |
| `bandit` | `bandit` |
| `bandit_v2` | `bandit_v2` |
| `bs_conformer` | `bs_conformer` |
| `bs_mamba2` | `bs_mamba2` |
| `bs_roformer` | `bs_roformer` |
| `bs_roformer_experimental` | `bs_roformer_experimental` |
| `conformer` | `conformer` |
| `htdemucs` | `htdemucs` |
| `mel_band_conformer` | `mel_band_conformer` |
| `mel_band_roformer` | `mel_band_roformer` |
| `mel_band_roformer_experimental` | `mel_band_roformer_experimental` |
| `moises_light` | `moises_light` |
| `scnet_tran` | `scnet_tran` |
| `scnet_unofficial` | `scnet_unofficial` |
| `segm_models` | `segm_models` |
| `swin_upernet` | `swin_upernet` |
| `torchseg` | `torchseg` |

Extras can be combined in one installation command:

```bash
pip install "msst[train,validation,bs_roformer]"
```

See the
[model compatibility table](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/compatibility.md#model-extras)
for architecture-specific notes.

## Python API

The package exports the main interfaces directly from `msst`:

```python
import msst

msst.inference
msst.Separator
msst.train
msst.valid
msst.check
msst.build_metadata
msst.init_experiment
msst.load_experiment
```

Public functions accept `str` and `pathlib.Path` values and are annotated for
type checkers. Package-specific exceptions inherit from `msst.MSSTError`.
Use `help(msst.inference)` or the
[complete Python API reference](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/python_api.md)
for every parameter, return type, and error condition.

Additional documentation:

- [Getting started](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/getting_started.md)
- [Command-line interface](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/cli.md)
- [Experiments and run tracking](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/experiments.md)
- [Augmentations](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/augmentations.md)
- [LoRA training](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/LoRA.md)
- [Package architecture](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/package_architecture.md)

## License and citation

MSST is distributed under the [MIT License](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/LICENSE).

If MSST contributes to published work, cite:

```bibtex
@misc{solovyev2023benchmarks,
  title={Benchmarks and leaderboards for sound demixing tasks},
  author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
  year={2023},
  eprint={2305.07489},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
}
```

MSST is brought to you by [MVSep](https://mvsep.com).
