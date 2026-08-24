# Music Source Separation Universal Training Code

Repository for training models for music source separation. Repository is based on [kuielab code](https://github.com/kuielab/sdx23/tree/mdx_AB/my_submission/src) for [SDX23 challenge](https://github.com/kuielab/sdx23/tree/mdx_AB/my_submission/src). The main idea of this repository is to create training code, which is easy to modify for experiments. Brought to you by [MVSep.com](https://mvsep.com).

## Models

Model can be chosen with `--model_type` arg.

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

All training parameters are defined in
[`msst/utils/settings.py`](msst/utils/settings.py).

### Training with LoRA

Look here: [LoRA training](docs/LoRA.md)

## How to: Inference

### Python package

Install MSST together with the dependencies for the required model family:

```bash
pip install "msst[bs_roformer]"
```

On CUDA machines, install the PyTorch wheel appropriate for the GPU and driver
before installing MSST. For example, the tested V100 setup uses matching
`torch==2.11.0` and `torchaudio==2.11.0` CUDA 12.6 wheels. MSST does not require
Triton, Torch-TensorRT, or TorchAO.

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

Training is available through `msst.train(...)`; install the training dependencies
with `pip install "msst[train,bs_roformer]"`. The existing command-line interface
also remains available as `msst-inference` and `msst-train`.

Input folders accept WAV, FLAC, MP3, OGG/Opus, AIFF and the other formats listed
in the [model and audio compatibility guide](docs/compatibility.md). Container
formats such as M4A, MP4, WebM and WMA need a compatible system decoder and are
therefore conditional. Output is WAV (`FLOAT` by default) or FLAC (`PCM_24` by
default); invalid codec/subtype combinations fail before inference starts.

The historical dictionary form remains available as
`from msst.inference import proc_folder`. See the complete
[Python API and migration guide](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/python_api.md).
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

All inference parameters are defined in
[`msst/utils/settings.py`](msst/utils/settings.py).
Convert models to ONNX and TensorRT formats [here](https://github.com/ZFTurbo/MSS_ONNX_TensorRT).

## Useful notes

* All batch sizes in config are adjusted to use with single NVIDIA A6000 48GB. If you have less memory please adjust correspodningly in model config `training.batch_size` and `training.gradient_accumulation_steps`.
* It's usually always better to start with old weights even if shapes not fully match. Code supports loading weights for not fully same models (but it must have the same architecture). Training will be much faster.

## Code description

* `configs/config_*.yaml` - configuration files for models
* `msst/models/*` - model implementations used for training and inference
* `msst/utils/dataset.py` - datasets which create training samples
* `msst/inference.py` and `msst/training.py` - stable public Python API
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
