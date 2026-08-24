"""Public Python API for Music Source Separation Training.

The public facades are lightweight; model and training runtimes remain lazy.
"""

from ._version import __version__
from .checkpoints import CHECKPOINT_FORMAT_VERSION
from .errors import (
    AudioFileError,
    CheckpointError,
    ConfigurationError,
    DeviceError,
    MSSTError,
    OptionalDependencyError,
    OutputFormatError,
)
from .inference import (
    DECODER_DEPENDENT_AUDIO_EXTENSIONS,
    InferenceConfig,
    NATIVE_AUDIO_EXTENSIONS,
    Separator,
    SUPPORTED_AUDIO_EXTENSIONS,
    inference,
)
from .training import train


__all__ = [
    "AudioFileError",
    "CheckpointError",
    "CHECKPOINT_FORMAT_VERSION",
    "ConfigurationError",
    "DECODER_DEPENDENT_AUDIO_EXTENSIONS",
    "DeviceError",
    "InferenceConfig",
    "MSSTError",
    "NATIVE_AUDIO_EXTENSIONS",
    "OptionalDependencyError",
    "OutputFormatError",
    "Separator",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "inference",
    "train",
    "__version__",
]
