"""Runtime registry for models supplied by MSST extensions.

Built-in architectures intentionally remain on the legacy factory for now.  The
registry is an extension seam: a third-party package can register a model in
its import path and use the normal MSST config, checkpoint, inference, and
standard-training workflows without editing this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from typing import Any, Callable, TypeVar

import torch
from torch import nn


ModelFactory = Callable[..., nn.Module]
_ModelT = TypeVar("_ModelT", bound=nn.Module)

# Keep this list close to the registry so an extension cannot silently replace
# a built-in architecture by registering the same key.
BUILTIN_MODEL_TYPES = frozenset(
    {
        "mdx23c",
        "htdemucs",
        "segm_models",
        "torchseg",
        "mel_band_roformer",
        "mel_band_conformer",
        "mel_band_roformer_experimental",
        "bs_roformer",
        "bs_conformer",
        "bs_roformer_experimental",
        "bs_mamba2",
        "swin_upernet",
        "bandit",
        "bandit_v2",
        "scnet_unofficial",
        "scnet",
        "scnet_tran",
        "apollo",
        "experimental_mdx23c_stht",
        "dttnet",
        "scnet_masked",
        "conformer",
        "moises_light",
    }
)
_MODEL_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


class MSSTBaseModel(nn.Module):
    """Small contract for third-party models used by generic MSST workflows.

    A model receives a parsed config and its ``forward`` must return a tensor
    shaped ``[batch, stems, channels, time]`` for a generic model.  The base
    class deliberately does not impose an architecture or tensor internals;
    it only supplies a predictable config factory and an explicit capability
    flag for models with their own training loss.

    Subclasses may set ``uses_internal_loss = True`` and implement the legacy
    internal-loss call ``forward(mix, target, active_stem_ids=...)``.  Most
    custom models should leave it false and use MSST's standard loss path.
    """

    uses_internal_loss = False

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_config(
        cls: type[_ModelT], config: Any, *, pretrained: bool = False
    ) -> _ModelT:
        """Construct ``cls`` from the complete parsed MSST config.

        The default assumes ``__init__(config)``.  A custom model can override
        this method when it needs to read a different config section or load a
        pretrained backbone.
        """

        del pretrained
        return cls(config)  # type: ignore[call-arg]

    def forward(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        active_stem_ids: Any = None,
    ) -> torch.Tensor:
        if target is None:
            return self.separate(mixture, active_stem_ids=active_stem_ids)
        return self.compute_loss(
            mixture, target, active_stem_ids=active_stem_ids
        )

    def separate(
        self, mixture: torch.Tensor, *, active_stem_ids: Any = None
    ) -> torch.Tensor:
        """Return ``[batch, stems, channels, time]`` predictions."""

        raise NotImplementedError

    def compute_loss(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
        *,
        active_stem_ids: Any = None,
    ) -> torch.Tensor:
        """Return a scalar loss for models with an internal training loss."""

        del mixture, target, active_stem_ids
        raise NotImplementedError(
            "Use MSST's standard loss or implement compute_loss()"
        )


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """Validated description of one registered extension model."""

    name: str
    factory: ModelFactory
    uses_internal_loss: bool = False

    def build(self, config: Any, *, pretrained: bool = False) -> nn.Module:
        """Build and validate the extension model instance."""

        creator = self.factory
        try:
            parameters = inspect.signature(creator).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_pretrained = (
            "pretrained" in parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        model = (
            creator(config, pretrained=pretrained)
            if accepts_pretrained
            else creator(config)
        )
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Registered model {self.name!r} returned {type(model).__name__}; "
                "a torch.nn.Module is required"
            )
        return model


_REGISTRY: dict[str, ModelRegistration] = {}


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"model name must be a string, got {type(name).__name__}")
    normalized = name.strip().lower()
    if not normalized or not _MODEL_NAME.fullmatch(normalized):
        raise ValueError(
            "model name must start with a letter and contain only lowercase "
            "letters, digits, '_' or '-'"
        )
    return normalized


def _lookup_name(name: str) -> str:
    """Normalize a lookup without turning legacy unknown keys into API errors."""

    return name.strip().lower() if isinstance(name, str) else str(name).strip().lower()


def _factory_for(model: type[nn.Module] | ModelFactory) -> ModelFactory:
    if not callable(model):
        raise TypeError("model must be a torch.nn.Module class or callable factory")

    from_config = getattr(model, "from_config", None)
    if callable(from_config):
        return from_config
    return model


def _has_internal_loss(model: Any) -> bool:
    if isinstance(model, type) and issubclass(model, MSSTBaseModel):
        if "uses_internal_loss" in model.__dict__:
            return bool(model.__dict__["uses_internal_loss"])
        return model.compute_loss is not MSSTBaseModel.compute_loss
    return bool(getattr(model, "uses_internal_loss", False))


def register_model(
    name: str,
    model: type[nn.Module] | ModelFactory,
    *,
    uses_internal_loss: bool | None = None,
    overwrite: bool = False,
) -> ModelRegistration:
    """Register an external model for normal MSST model construction.

    Typical usage from a third-party package::

        import msst
        from .model import MimoMelBand

        msst.register_model("mimo_mel_band", MimoMelBand)

    ``model`` may be an :class:`MSSTBaseModel` subclass, any ``nn.Module``
    class accepting the complete config, or a factory callable.  Registration
    is process-local and intentionally explicit; importing a package is the
    opt-in needed to make its model available.
    """

    normalized = _normalize_name(name)
    if normalized in BUILTIN_MODEL_TYPES:
        raise ValueError(
            f"{normalized!r} is a built-in MSST model type and cannot be replaced"
        )
    if normalized in _REGISTRY and not overwrite:
        raise ValueError(f"Model {normalized!r} is already registered")

    factory = _factory_for(model)
    if uses_internal_loss is None:
        uses_internal_loss = _has_internal_loss(model)
    registration = ModelRegistration(
        name=normalized,
        factory=factory,
        uses_internal_loss=bool(uses_internal_loss),
    )
    _REGISTRY[normalized] = registration
    return registration


def unregister_model(name: str) -> None:
    """Remove one process-local extension registration."""

    _REGISTRY.pop(_normalize_name(name), None)


def get_model_registration(name: str) -> ModelRegistration | None:
    """Return a registration, or ``None`` for legacy built-in models."""

    return _REGISTRY.get(_lookup_name(name))


def registered_models() -> tuple[str, ...]:
    """Return registered extension names in deterministic order."""

    return tuple(sorted(_REGISTRY))


def model_uses_internal_loss(name: str) -> bool:
    """Resolve the training-loss mode for built-ins and extensions."""

    normalized = _lookup_name(name)
    registration = _REGISTRY.get(normalized)
    if registration is not None:
        return registration.uses_internal_loss
    return (
        "roformer" in normalized
        or normalized in {"mel_band_conformer", "bs_mamba2", "bs_conformer"}
    )


__all__ = [
    "BUILTIN_MODEL_TYPES",
    "MSSTBaseModel",
    "ModelRegistration",
    "get_model_registration",
    "model_uses_internal_loss",
    "register_model",
    "registered_models",
    "unregister_model",
]
