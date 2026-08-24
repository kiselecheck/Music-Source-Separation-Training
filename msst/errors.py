"""Exceptions raised by the public MSST API."""

import platform


class MSSTError(Exception):
    """Base class for errors raised by MSST's public API."""


class OptionalDependencyError(ImportError, MSSTError):
    """A requested feature needs an optional dependency that is not installed."""

    def __init__(self, dependency: str, *, extra: str, feature: str) -> None:
        self.dependency = dependency
        self.extra = extra
        self.feature = feature
        install_hint = f'Install it with: pip install "msst[{extra}]"'
        if dependency == "mamba_ssm" and not (
            platform.system() == "Linux" and platform.machine() == "x86_64"
        ):
            install_hint = (
                f"mamba-ssm is not supported by this package on "
                f"{platform.system()} {platform.machine()}; run this model on "
                "Linux x86_64 with a compatible CUDA toolchain"
            )
        super().__init__(
            f"{feature} requires the optional dependency {dependency!r}. "
            f"{install_hint}"
        )


class ConfigurationError(ValueError, MSSTError):
    """A model configuration is missing, malformed, or incompatible."""


class CheckpointError(RuntimeError, MSSTError):
    """A checkpoint cannot be read or does not match the configured model."""


class AudioFileError(RuntimeError, MSSTError):
    """An input was recognized as audio but could not be decoded or processed."""


class DeviceError(RuntimeError, MSSTError):
    """The requested CPU/GPU device cannot be used."""


class OutputFormatError(ValueError, MSSTError):
    """The requested output codec/subtype combination is unsupported."""


def optional_dependency_error(
    error: ModuleNotFoundError,
    *,
    extra: str,
    feature: str,
) -> OptionalDependencyError:
    """Build a consistent public error from an internal import failure."""

    dependency = error.name or str(error)
    return OptionalDependencyError(dependency, extra=extra, feature=feature)


__all__ = [
    "AudioFileError",
    "CheckpointError",
    "ConfigurationError",
    "DeviceError",
    "MSSTError",
    "OptionalDependencyError",
    "OutputFormatError",
]
