# Music Source Separation Universal Training Code

[![Python package CI](https://github.com/ZFTurbo/Music-Source-Separation-Training/actions/workflows/ci.yml/badge.svg)](https://github.com/ZFTurbo/Music-Source-Separation-Training/actions/workflows/ci.yml)
![Python 3.10–3.13](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

MSST is a PyTorch toolkit for music source separation inference, validation,
and training. It combines a regular Python package with experiment-friendly
command-line entry points and a broad collection of separation architectures.
Brought to you by [MVSep.com](https://mvsep.com).

[Quick start](#quick-start) · [Public API](docs/python_api.md) ·
[Model compatibility](docs/compatibility.md) · [Dataset layouts](docs/dataset_types.md) ·
[Package architecture](docs/package_architecture.md) · [Release process](docs/releasing.md)

> [!IMPORTANT]
> The wheel contains runtime code only. Model YAML files and checkpoints are
> independently versioned artifacts and must be supplied by the caller.

## Quick start

Install the CUDA build of PyTorch appropriate for the target GPU first, then
install MSST with the model and workflow extras you need:

```bash
pip install "msst[bs_roformer]"                    # inference
pip install "msst[validation,bs_roformer]"         # validation
pip install "msst[train,bs_roformer]"              # training
```

| Goal | Python API | Command |
| --- | --- | --- |
| Separate files | `msst.inference(...)` or `msst.Separator(...)` | `msst-inference` |
| Evaluate a checkpoint | `msst.valid(...)` | `msst-valid` |
| Train or fine-tune | `msst.train(...)` | `msst train` or `msst-train` |
| Build dataset metadata | `msst.build_metadata(...)` | `msst metadata` |
| Check a planned run | `msst.check(...)` | `msst check` |
| Create an experiment layout | `msst.init_experiment(...)` | `msst init` |

```python
import msst

written = msst.inference(
    model_type="bs_roformer",
    config_path="models/dereverb-v4.yaml",
    checkpoint_path="models/dereverb-v4.ckpt",
    input_folder="audio/input",
    output_folder="audio/stems",
    device_ids=0,
)

scores = msst.valid(
    model_type="bs_roformer",
    config_path="models/dereverb-v4.yaml",
    checkpoint_path="models/dereverb-v4.ckpt",
    valid_path="datasets/dereverb-test",
    metrics=("sdr", "k_sdr"),
    device_ids=0,
)
print(scores.averages)
```

All public functions use keyword arguments and accept `pathlib.Path` as well
as strings. See the [complete Python API reference](docs/python_api.md) for
parameter tables, return values, examples, and error behavior.

The unified command is `msst`. Existing commands such as `msst-train` and
`msst-valid` remain compatibility aliases.

## Supported models

Choose the model family with `--model_type`.

Available models for training:

* MDX23C based on [KUIELab TFC TDF v3 architecture](https://github.com/kuielab/sdx23/). Key: `mdx23c`.
* Demucs4HT [[Paper](https://arxiv.org/abs/2211.08553)]. Key: `htdemucs`.
* VitLarge23 based on [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch). Key: `segm_models`.
* TorchSeg based on [TorchSeg module](https://github.com/qubvel/segmentation_models.pytorch). Key: `torchseg`.
* Band Split RoFormer [[Paper](https://arxiv.org/abs/2309.02612), [Repository](https://github.com/lucidrains/BS-RoFormer)] . Key: `bs_roformer`.
* Mel-Band RoFormer [[Paper](https://arxiv.org/abs/2310.01809), [Repository](https://github.com/lucidrains/BS-RoFormer)]. Key: `mel_band_roformer`.
* Swin Upernet [[Paper](https://arxiv.org/abs/2103.14030)] Key: `swin_upernet`.
* BandIt Plus [[Paper](https://arxiv.org/abs/2309.02539), [Repository](https://github.com/karnwatcharasupat/bandit)] Key: `bandit`.
* SCNet [[Paper](https://arxiv.org/abs/2401.13276), [Official Repository](https://github.com/starrytong/SCNet), [Unofficial Repository](https://github.com/amanteur/SCNet-PyTorch)] Key: `scnet`.
* BandIt v2 [[Paper](https://arxiv.org/abs/2407.07275), [Repository](https://github.com/kwatcharasupat/bandit-v2)] Key: `bandit_v2`.
* Apollo [[Paper](https://arxiv.org/html/2409.08514v1), [Repository](https://github.com/JusperLee/Apollo)] Key: `apollo`.
* BSMamba2 [[Paper](https://arxiv.org/abs/2508.14556), [Repository](https://github.com/EuiYeonKim/BSMamba2)] Key: `bs_mamba2`.
* Conformer [[Paper](https://arxiv.org/abs/2005.08100), [Repository](https://github.com/lucidrains/conformer)] Key: `conformer`.
* BS Conformer Key: `bs_conformer`
* SCNet Tran Key: `scnet_tran`.
* SCNet Masked Key: `scnet_masked`.

1. **Note 1**: For `segm_models` there are many different encoders is possible. [Look here](https://github.com/qubvel/segmentation_models.pytorch#encoders-).
2. **Note 2**: Thanks to [@lucidrains](https://github.com/lucidrains) for recreating the RoFormer models based on papers.
3. **Note 3**: For `torchseg` gives access to more than 800 encoders from `timm` module. It's similar to `segm_models`.

## How to: Train

To train model you need to:

1) Choose model type with option `--model_type`, including: `mdx23c`, `htdemucs`, `segm_models`, `mel_band_roformer`, `bs_roformer`.
2) Choose location of config for model `--config_path` `<config path>`. You can find examples of configs in [configs folder](configs/). Prefixes `config_musdb18_` are examples for [MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html).
3) If you have a check-point from the same model or from another similar model you can use it with option: `--start_check_point` `<weights path>`
4) Choose path where to store results of training `--results_path` `<results folder path>`

### Training example

```bash
python train.py \
    --model_type mel_band_roformer \
    --config_path configs/config_mel_band_roformer_vocals.yaml \
    --start_check_point results/model.ckpt \
    --results_path results/ \
    --data_path 'datasets/dataset1' 'datasets/dataset2' \
    --valid_path datasets/musdb18hq/test \
    --num_workers 4 \
    --device_ids 0
```

Use `msst-train --help` for the CLI contract. The equivalent Python parameters
and less-common training options are listed in the
[Python API reference](docs/python_api.md#training).

Choose the multi-GPU runtime explicitly when needed:

```bash
msst train --launcher standard ...     # CPU, one GPU, or DataParallel
msst train --launcher ddp ...          # DistributedDataParallel
msst train --launcher accelerate ...   # Hugging Face Accelerate runtime
```

All three launchers accept the same training CLI. In DDP mode rank 0 owns the
main console output; output from other ranks is retained under
`RESULTS/ddp_logs/` instead of being repeated in the main log.

Before a long job, run either the non-forward check or one real train-step:

```bash
msst check --mode safe --config_path config.yaml --data_path data/train \
  --valid_path data/valid --results_path results --model_type bs_roformer

msst check --mode full --config_path config.yaml --data_path data/train \
  --valid_path data/valid --results_path results --model_type bs_roformer \
  --start_check_point weights/model.ckpt --device_ids 0
```

`full` executes forward, loss, backward, and one in-memory optimizer step
without a checkpoint write, and reports peak VRAM. It is a representative real batch,
not a mathematical guarantee for every random augmentation.

## Portable experiments

`msst init experiments/dereverb-v5` creates a portable structure:

```text
dereverb-v5/
├── experiment.yaml
├── configs/
├── checkpoints/
├── jobs/
├── logs/
├── results/
└── runs/
```

The `jobs/` directory is intentionally empty so launch scripts remain
user-owned. Pass
`experiment="experiments/dereverb-v5"` to `msst.train`, `msst.valid`, or
`msst.inference` to create a run record containing arguments, environment,
config snapshot, input hashes, status, and artifacts. Output paths are not
silently changed by tracking.

### Training with LoRA

Look here: [LoRA training](docs/LoRA.md)

## How to: Inference

### Python package

For development from a local checkout, use `pip install -e ".[bs_roformer]"`.
The distribution name and import namespace are both `msst`.

For a one-off folder job, use regular keyword arguments. The function returns
the paths it created:

```python
from msst import inference

written_files = inference(
    model_type="bs_roformer",
    config_path="models/bs_6stem_fixed_config.yaml",
    checkpoint_path="models/bs_6stem_fixed.ckpt",
    input_folder="input/newaudios",
    output_folder="input/newaudios_cleaned",
    device_ids=0,
)
```

When processing more than one input, use `Separator` so the model is loaded
only once:

```python
from msst import Separator

with Separator(
    model_type="bs_roformer",
    config_path="models/bs_6stem_fixed_config.yaml",
    checkpoint_path="models/bs_6stem_fixed.ckpt",
    device_ids=0,
) as separator:
    stems = separator.separate_file("song.wav")
    separator.separate_folder("input/batch_1", "output/batch_1")
    separator.separate_folder("input/batch_2", "output/batch_2")
```

Input folders accept WAV, FLAC, MP3, OGG/Opus, AIFF and the other formats listed
in the [model and audio compatibility guide](docs/compatibility.md). Container
formats such as M4A, MP4, WebM and WMA need a compatible system decoder and are
therefore conditional. Output is WAV (`FLOAT` by default) or FLAC (`PCM_24` by
default); invalid codec/subtype combinations fail before inference starts.

The historical dictionary form remains available as
`from msst.inference import proc_folder`. See the complete
[Python API and migration guide](docs/python_api.md).
The namespace layout and the process for incorporating future upstream changes
are documented in the [package architecture guide](docs/package_architecture.md).

### Inference example

```bash
python inference.py \
    --model_type mdx23c \
    --config_path configs/config_mdx23c_musdb18.yaml \
    --start_check_point results/last_mdx23c.ckpt \
    --input_folder input/wavs/ \
    --store_dir separation_results/
```

Use `msst-inference --help` for every CLI option or the
[Python API reference](docs/python_api.md#one-off-folder-inference) for typed
keyword parameters.

## How to: Validate

Validation evaluates a checkpoint against reference stems and returns both
aggregate and per-track metrics:

```python
from msst import valid

result = valid(
    model_type="bs_roformer",
    config_path="models/model.yaml",
    checkpoint_path="models/model.ckpt",
    valid_path="datasets/test",
    metrics=("sdr", "k_sdr"),
    device_ids=0,
)

print(result.averages)
print(result.per_track["sdr"])
```

The corresponding CLI command is `msst-valid`; run `msst-valid --help` for
all options. Validation dataset layouts are described in
[Dataset types](docs/dataset_types.md#dataset-for-validation).

## Useful notes

* Example configs may need adjustment for the available VRAM. Change
  `training.batch_size` and
  `training.gradient_accumulation_steps` in the model config.
* Starting from compatible weights can shorten training even when every tensor
  shape does not match. The architecture itself must remain compatible.

## Code description

* `configs/config_*.yaml` - configuration files for models
* `msst/utils/dataset.py` - datasets which create training samples
* `msst/inference.py`, `msst/validation.py`, and `msst/training.py` - stable
  public Python API
* `msst/models/` - model implementations
* `msst/utils/` - datasets and shared training/inference utilities
* `inference.py`, `train.py`, `valid.py`, and `ensemble.py` - source-checkout compatibility entry points
* `gui-wx.py` - GUI interface for code
* `train_ddp.py` - compatibility entry point for multi-GPU training

## Pre-trained models

Look here: [List of Pre-trained models](docs/pretrained_models.md)

If you trained some good models, please, share them. You can post config and model weights [in this issue](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/1).

## Dataset types

Look here: [Dataset types](docs/dataset_types.md)

## Augmentations

Look here: [Augmentations](docs/augmentations.md)

## Graphical user interface

Look here: [GUI documentation](docs/gui.md) or see tutorial on [Youtube](https://youtu.be/M8JKFeN7HfU)

## Citation

* [arxiv paper](https://arxiv.org/abs/2305.07489)

```text
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
