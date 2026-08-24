# MSST Python API

MSST exposes a small stable API while preserving the repository's existing
command-line and dictionary-based interfaces.

## Installation

Install the base package plus the extra matching the model architecture:

```bash
pip install "msst[bs_roformer]"
```

Extras use the same names as `model_type`. Relevant combinations include:

```bash
pip install "msst[mel_band_roformer]"
pip install "msst[scnet_unofficial]"
pip install "msst[train,bs_roformer]"
pip install "msst[gui]"
```

LoRA support is independent of the architecture extra:

```bash
pip install "msst[bs_roformer,lora-peft]"
pip install "msst[bs_roformer,lora-loralib]"
```

## One-off folder inference

`msst.inference` loads a model, processes one folder, releases the model, and
returns the output paths:

```python
from msst import inference

written = inference(
    model_type="bs_roformer",
    config_path="models/bs_6stem_fixed_config.yaml",
    checkpoint_path="models/bs_6stem_fixed.ckpt",
    input_folder="input/audio",
    output_folder="output/stems",
    device_ids=0,
    pcm_type="FLOAT",
)
```

By default the library API fails on an unreadable input file. Pass
`skip_errors=True` to report the error and continue with the remaining files.
Non-audio files in a folder are ignored. If two inputs have the same relative
name but different extensions (for example `song.wav` and `song.flac`), use a
template containing `{extension}` to keep their output paths distinct:

```python
separator.separate_folder(
    "input",
    "output",
    filename_template="{relative_path}_{extension}/{instr}",
)
```

Available fields are `{instr}`, `{start_time}`, `{file_name}`, `{dir_name}`,
`{relative_path}`, `{extension}`, `{model_type}`, and `{model}`. Unsafe paths,
unknown fields, malformed templates, and output collisions are rejected.

## Reusing a loaded model

`Separator` is the recommended API for services, notebooks, and pipelines. Its
constructor performs model loading once:

```python
from msst import Separator

with Separator(
    model_type="bs_roformer",
    config_path="models/bs_6stem_fixed_config.yaml",
    checkpoint_path="models/bs_6stem_fixed.ckpt",
    device_ids=0,
) as separator:
    print(separator.device)
    print(separator.sample_rate)
    print(separator.instruments)

    first_stems = separator.separate_file("input/first.wav")
    second_stems = separator.separate_file("input/second.wav")
```

Each stem is a NumPy array with shape `(channels, samples)`. To process an
in-memory waveform:

```python
stems = separator.separate(
    waveform,
    sample_rate=48000,
    channels_first=True,
)
```

Mono arrays may have shape `(samples,)`. Two-dimensional input is interpreted
as `(channels, samples)` unless `channels_first=False` is supplied. If the input
sample rate differs from the model configuration, MSST resamples it to
`separator.sample_rate`.

The same loaded model can write multiple folders:

```python
separator.separate_folder("input/album_a", "output/album_a")
separator.separate_folder(
    "input/album_b",
    "output/album_b",
    flac_file=True,
    pcm_type="PCM_24",
    filename_template="{dir_name}/{file_name}/{instr}",
)
```

Calls on one `Separator` instance are serialized because model-specific
inference implementations are not assumed to be thread-safe. Use separate
instances when deliberate parallel model execution is required.

## Training

Training uses the packaged runtime through a lazy public facade:

```python
from msst import train

train(
    model_type="bs_roformer",
    config_path="configs/config_vocals_bs_roformer.yaml",
    checkpoint_path="weights/start.ckpt",
    results_path="results/run_01",
    data_path=["datasets/train_a", "datasets/train_b"],
    valid_path="datasets/validation",
    device_ids=[0, 1],
    num_workers=8,
)
```

Less common training CLI options can be passed as additional keyword arguments.
Install both training and model extras, for example
`pip install "msst[train,bs_roformer]"`.

New checkpoints written by the regular and Accelerate training paths contain
versioned compatibility metadata (`format_version`, `msst_version`,
`model_type`, selected stem, and stem count when known). Inference checks this
metadata before loading tensors and rejects checkpoint formats newer than the
installed package supports. Old raw state dictionaries and older checkpoint
dictionaries remain supported.

Configs and checkpoints are model artifacts with versions independent of the
Python distribution. See [model artifact versioning](model_artifacts.md).

## Backward compatibility

Existing dictionary calls continue to work:

```python
from msst.inference import proc_folder

proc_folder({
    "model_type": "bs_roformer",
    "config_path": "models/bs_6stem_fixed_config.yaml",
    "start_check_point": "models/bs_6stem_fixed.ckpt",
    "input_folder": "input/audio",
    "store_dir": "output/stems",
    "device_ids": "0",
})
```

The aliases `checkpoint_path` and `output_folder` are also accepted by
`proc_folder`. Existing `python inference.py ...` commands remain supported;
installed environments can additionally use `msst-inference ...`.

## Errors and optional dependencies

Missing architecture dependencies raise `msst.OptionalDependencyError` with
the extra that should be installed. Public failures can be handled without
parsing log output:

```python
from msst import AudioFileError, MSSTError

try:
    separator.separate_file("input.m4a")
except AudioFileError as error:
    print(error)  # includes the input path and decoder hint
except MSSTError as error:
    print(error)
```

- `ConfigurationError`: malformed YAML, missing sections, unknown model type,
  or constructor/config mismatch.
- `CheckpointError`: unreadable weights, metadata mismatch, or strict tensor
  incompatibility.
- `AudioFileError`: unsupported extension, decode failure, corrupt input, or
  per-file processing failure.
- `DeviceError`: invalid/unavailable CUDA index or failure moving the model.
- `OutputFormatError`: unsupported WAV/FLAC subtype combination.
- `OptionalDependencyError`: a model extra is missing or unsupported on the
  current platform.

Missing paths and invalid in-memory waveform arguments continue to use standard
`FileNotFoundError`, `NotADirectoryError`, `TypeError`, and `ValueError`.

See [model and audio compatibility](compatibility.md) for the complete input
format and platform matrix.
