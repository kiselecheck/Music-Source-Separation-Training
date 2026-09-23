# Custom models

MSST supports additional model types through a process-local registry. An
extension registers a PyTorch module under a model name before invoking an MSST
workflow that constructs the model.

## Model interface

Custom architectures can subclass `msst.MSSTBaseModel` and implement
`separate(mixture)`. The input tensor has shape `[batch, channels, time]`; the
returned tensor must have shape `[batch, stems, channels, time]`, with stems in
the order declared by `training.instruments`. MSST uses this output for
inference and its standard training loss.

The repository includes [`TinySeparator`](../examples/custom_model.py), a
small two-stem example with two 1D convolution layers. It predicts the first
stem and computes the second as the residual from the input mixture.

## Registration and training

From a source checkout, register the example class before starting training:

```python
import msst
from examples.custom_model import TinySeparator

msst.register_model("tiny_separator", TinySeparator)

msst.train(
    config_path="config.yaml",
    results_path="results",
    data_path="data/train",
    valid_path="data/valid",
    model_type="tiny_separator",
)
```

The registration is available in the current Python process. A separately
installed extension package can expose the same class and registration call;
the class must be importable before invoking MSST. Training uses the configured
MSST dataset layout and runs validation after every epoch. Dataset structures
are described in [Dataset formats](dataset_types.md).

## Configuration

`training.instruments` defines the output stem names and their order. Generic
inference also requires chunk size, batch size, and overlap settings under
`inference`. Custom architecture parameters can be placed in any configuration
section and read by the model constructor. `MSSTBaseModel.from_config` calls
`ModelClass(config)` by default; override it when construction requires a
different signature or pretrained components.

## Training losses

The standard loss path is suitable for models that return separated audio. A
model with a specialized loss can implement `compute_loss(mixture, target)` and
set `uses_internal_loss = True`. Its `separate(mixture)` method must still
return the standard stem tensor for inference.

## Registration names

Names are case-insensitive, start with a letter, and contain only letters,
digits, `_`, or `-`. Built-in model names are reserved. The active extension
registrations are available through `msst.registered_models()` and can be
removed with `msst.unregister_model(name)`.
